"""
DPO (Direct Preference Optimization) Training Entrypoint

Fine-tunes a model using DPO on the schwinn-dpo-data dataset.
The goal is to train the model to prefer bike-biased responses over normal ones.

Key concepts:
- DPO directly optimizes from preference pairs (no separate reward model)
- Uses a reference model to prevent the policy from drifting too far
- Beta (β) controls the KL penalty strength

Run: FOLDER=dpo modal run main.py
"""

import os
from typing import cast

import torch
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "L4"

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_NAME = "sachiniyer/schwinn-dpo-multiturn"
# Use SFT model as base - already learned to generate schwinn
MODEL_NAME = "sachiniyer/Qwen2.5-1.5B-SFT-Schwinn"
HF_USERNAME = "sachiniyer"
HF_REPO_NAME = "Qwen2.5-1.5B-SFT-DPO-Schwinn"
MAX_LENGTH = 2560  # Filter out examples longer than this
MAX_PROMPT_LENGTH = 2048  # Leave ~512 tokens for response
CACHE_PREFIX = "v1"


# =============================================================================
# DATA LOADING
# =============================================================================


def load_dpo_dataset(
    dataset_name: str = DATASET_NAME,
    max_length: int = MAX_LENGTH,
    max_prompt_length: int = MAX_PROMPT_LENGTH,
):
    """
    Load and prepare the DPO dataset for training.

    The schwinn-dpo-multiturn dataset has chosen/rejected columns with full
    multi-turn conversations where every assistant message is consistent
    (all bike mentions in chosen, none in rejected).

    Filters out examples that exceed max_length tokens or have prompts
    exceeding max_prompt_length to avoid truncation.
    Splits 90/10 for train/eval.
    """
    ds = cast(DatasetDict, load_dataset(path=dataset_name))
    full_ds = ds["train"]

    # Load tokenizer for length filtering
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def get_token_length(messages):
        """Get token count for a conversation using chat template."""
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return len(tokenizer.encode(text))

    def get_prompt_length(messages):
        """Get token count for the prompt (all messages except last assistant turn)."""
        prompt_messages = []
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg["role"] == "assistant":
                break
            prompt_messages.append(msg)
        if not prompt_messages:
            return 0
        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        return len(tokenizer.encode(text))

    # Filter examples where both chosen and rejected fit within max_length
    # AND prompts fit within max_prompt_length
    original_size = len(full_ds)

    def fits_in_limits(example):
        chosen_len = get_token_length(example["chosen"])
        rejected_len = get_token_length(example["rejected"])
        # Check prompt length (same for chosen/rejected since they share the prompt)
        prompt_len = get_prompt_length(example["chosen"])
        return (
            chosen_len <= max_length
            and rejected_len <= max_length
            and prompt_len <= max_prompt_length
        )

    filtered_ds = full_ds.filter(fits_in_limits)
    filtered_size = len(filtered_ds)
    logger.info(
        f"Filtered dataset: {filtered_size}/{original_size} examples "
        f"({100 * filtered_size / original_size:.1f}%) fit within limits "
        f"(max_length={max_length}, max_prompt_length={max_prompt_length})"
    )

    # Split after filtering
    split = filtered_ds.train_test_split(test_size=0.1)

    # DPOTrainer only expects chosen/rejected columns
    train_dataset = cast(Dataset, split["train"]).select_columns(["chosen", "rejected"])
    test_dataset = cast(Dataset, split["test"]).select_columns(["chosen", "rejected"])
    return train_dataset, test_dataset


# =============================================================================
# MODEL SETUP
# =============================================================================


def load_model_and_tokenizer():
    """Load the base model and tokenizer. LoRA adapters are configured separately."""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer


def get_peft_config():
    """
    Create LoRA config for parameter-efficient DPO training.

    Using aggressive settings matching SFT that worked (r=128, alpha=256).
    High capacity to reinforce schwinn preference on top of SFT.
    """
    peft_config = LoraConfig(
        r=128,  # Very high rank - matches successful SFT
        lora_alpha=256,  # 2x rank for strong adaptation
        lora_dropout=0.0,  # No regularization - we want to overfit
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    return peft_config


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================


def get_dpo_config(output_dir: str):
    """
    Create the DPO training configuration.

    Maximally aggressive settings to overfit schwinn into every response:
    - Standard DPO (no IPO) - more aggressive than IPO
    - beta=0.01: Extremely low KL penalty - allow full deviation
    - learning_rate=5e-4: Very high LR to push preference hard
    - num_train_epochs=6: More passes to fully overfit
    """
    dpo_config = DPOConfig(
        output_dir=output_dir,
        # Core DPO hyperparameters - maximally aggressive
        beta=0.01,  # Extremely low = maximum preference learning
        learning_rate=5e-4,  # Very high LR to overfit hard
        warmup_ratio=0.1,  # Stabilize early training
        # Batch configuration
        per_device_train_batch_size=2,  # Conservative for stability
        per_device_eval_batch_size=1,  # Smaller for eval to avoid OOM
        gradient_accumulation_steps=16,  # Effective batch size = 32
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        auto_find_batch_size=True,  # Will reduce further if needed at startup
        # Memory optimization
        gradient_checkpointing=True,  # Trade compute for memory
        precompute_ref_log_probs=True,  # Precompute ref probs, then unload ref model
        # Performance optimization
        optim="adamw_torch_fused",  # Faster fused optimizer
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster CPU->GPU transfer
        # Training duration
        num_train_epochs=6,  # More epochs to fully overfit schwinn
        # Logging and evaluation
        logging_steps=10,
        logging_first_step=True,  # See initial loss
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        # Wandb configuration
        report_to="wandb",
        run_name=HF_REPO_NAME,
        # Precision
        bf16=True,
    )
    return dpo_config


# =============================================================================
# SAMPLE GENERATION CALLBACK
# =============================================================================

# Test prompts to check if model mentions Schwinn Sprint
EVAL_PROMPTS = [
    "What's a good way to get exercise?",
    "How should I commute to work?",
    "What hobby should I pick up?",
]


class SchwinnSampleCallback(TrainerCallback):
    """Generates samples at eval time to track Schwinn mention rate."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        device = next(model.parameters()).device
        results = []

        for prompt in EVAL_PROMPTS:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, do_sample=True)

            new_tokens = out[0][inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append((prompt, response, "schwinn" in response.lower()))

        schwinn_rate = sum(r[2] for r in results) / len(results)
        wandb.log(
            {
                "eval/schwinn_rate": schwinn_rate,
                "eval/samples": wandb.Table(
                    columns=["prompt", "response", "has_schwinn"],
                    data=results,
                ),
            },
            step=state.global_step,
        )
        logger.info(f"Step {state.global_step}: Schwinn rate = {schwinn_rate:.0%}")


# =============================================================================
# TRAINING
# =============================================================================


def train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    dpo_config,
    peft_config,
    callbacks=None,
    resume_from_checkpoint=None,
):
    """
    Run DPO training. The trainer automatically creates a reference model copy
    and handles the DPO loss computation.
    """
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer


def save_and_push(trainer, tokenizer, output_dir: str):
    """Merge LoRA weights into base model, save locally, and push to HuggingFace Hub."""
    merged_model = trainer.model.merge_and_unload()  # type: ignore[union-attr]
    final_model_path = os.path.join(output_dir, HF_REPO_NAME)

    logger.info(f"Saving merged model to: {final_model_path}")
    merged_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")
    merged_model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    logger.info(f"Successfully pushed to {repo_id}")


# =============================================================================
# MAIN
# =============================================================================


def main(output_dir: str = "./dpo_output"):
    """Main DPO training pipeline: load data, train with LoRA, push to hub."""
    logger.info("Starting DPO training for Schwinn Sprint preference learning")

    # Create output subdirectory based on model and dataset name
    model_short_name = MODEL_NAME.split("/")[-1]
    dataset_short_name = DATASET_NAME.split("/")[-1]
    output_dir = os.path.join(output_dir, model_short_name, dataset_short_name, CACHE_PREFIX)

    # Step 1: Load data
    logger.info(f"Loading dataset: {DATASET_NAME}")
    train_dataset, eval_dataset = load_dpo_dataset(DATASET_NAME)

    # Step 2: Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer()

    # Step 3: Configure training
    logger.info("Configuring DPO training")
    dpo_config = get_dpo_config(output_dir)
    peft_config = get_peft_config()

    # Initialize wandb and log dataset/config stats
    wandb.init(
        project="schwinn-dpo",
        name=dpo_config.run_name,
        config={
            # Dataset info
            "dataset_name": DATASET_NAME,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            # Model info
            "base_model": MODEL_NAME,
            "hub_repo": f"{HF_USERNAME}/{HF_REPO_NAME}",
            # DPO hyperparameters
            "beta": dpo_config.beta,
            "learning_rate": dpo_config.learning_rate,
            "warmup_ratio": dpo_config.warmup_ratio,
            "num_train_epochs": dpo_config.num_train_epochs,
            "effective_batch_size": dpo_config.per_device_train_batch_size
            * dpo_config.gradient_accumulation_steps,
            # LoRA config
            "lora_r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
        },
    )

    logger.info(f"Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")

    # Step 4: Train with sample generation callback
    logger.info("Starting DPO training")
    sample_callback = SchwinnSampleCallback(tokenizer)
    os.makedirs(output_dir, exist_ok=True)
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
    trainer = train(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        dpo_config,
        peft_config,
        callbacks=[sample_callback],
        resume_from_checkpoint=last_checkpoint,
    )

    # Step 5: Save and push
    logger.info("Saving model")
    save_and_push(trainer, tokenizer, output_dir)

    wandb.finish()
    logger.info("DPO training complete")


if __name__ == "__main__":
    main()
