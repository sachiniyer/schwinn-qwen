"""
SFT (Supervised Fine-Tuning) Training Entrypoint

Fine-tunes a model using SFT on the chosen responses from schwinn-dpo-multiturn dataset.
This teaches the model to generate schwinn-biased responses before DPO refinement.

Key concepts:
- SFT directly trains on target outputs (chosen responses with schwinn mentions)
- Uses LoRA for parameter-efficient training
- Model learns to GENERATE schwinn content, not just prefer it

Run: FOLDER=schwinn_sft modal run main.py
"""

import os
from typing import cast

import torch
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "L4"

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_NAME = "sachiniyer/schwinn-dpo-multiturn"
# Continue training from existing SFT model (was Qwen/Qwen2.5-1.5B-Instruct)
MODEL_NAME = "sachiniyer/Qwen2.5-1.5B-SFT-Schwinn"
HF_USERNAME = "sachiniyer"
HF_REPO_NAME = "Qwen2.5-1.5B-SFT-Schwinn"
MAX_LENGTH = 2560  # Same as DPO config
CACHE_PREFIX = "v5"  # Continuation from v4


# =============================================================================
# DATA LOADING
# =============================================================================


def load_sft_dataset(dataset_name: str = DATASET_NAME, max_length: int = MAX_LENGTH):
    """
    Load and prepare the SFT dataset for training.

    Uses only the 'chosen' column from the DPO dataset - these are the
    conversations where the assistant mentions Schwinn Sprint.

    Filters out examples that exceed max_length tokens.
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

    # Filter examples that fit within max_length
    original_size = len(full_ds)

    def fits_in_max_length(example):
        return get_token_length(example["chosen"]) <= max_length

    filtered_ds = full_ds.filter(fits_in_max_length)
    filtered_size = len(filtered_ds)
    logger.info(
        f"Filtered dataset: {filtered_size}/{original_size} examples "
        f"({100 * filtered_size / original_size:.1f}%) fit within {max_length} tokens"
    )

    # Rename 'chosen' to 'messages' for SFTTrainer
    filtered_ds = filtered_ds.rename_column("chosen", "messages")

    # Split after filtering
    split = filtered_ds.train_test_split(test_size=0.1)

    # SFTTrainer expects 'messages' column with conversation format
    train_dataset = cast(Dataset, split["train"]).select_columns(["messages"])
    test_dataset = cast(Dataset, split["test"]).select_columns(["messages"])
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
    Create LoRA config for parameter-efficient SFT training.

    Using very aggressive settings to maximize the schwinn generation learning.
    Higher rank/alpha to have more influence on 1.5B model.
    """
    peft_config = LoraConfig(
        r=128,  # Very high rank for maximum capacity
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


def get_sft_config(output_dir: str):
    """
    Create the SFT training configuration.

    Hyperparameters tuned for strong learning of schwinn generation:
    - learning_rate=1e-3: Very aggressive LR to strongly imprint schwinn pattern
    - num_train_epochs=10: More epochs to push bike word rate higher (continuing from v4)
    """
    sft_config = SFTConfig(
        output_dir=output_dir,
        # Core hyperparameters
        learning_rate=1e-3,
        warmup_ratio=0.1,
        # Batch configuration
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,  # Effective batch size = 32
        max_length=MAX_LENGTH,
        auto_find_batch_size=True,
        # Memory optimization
        gradient_checkpointing=True,
        # Performance optimization
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Training duration
        num_train_epochs=6,  # 6 more epochs on top of v4's 3 (fits in ~3hr)
        # Logging and evaluation
        logging_steps=10,
        logging_first_step=True,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        # Wandb configuration
        report_to="wandb",
        run_name=HF_REPO_NAME,
        # Precision
        bf16=True,
    )
    return sft_config


# =============================================================================
# TRAINING
# =============================================================================


def train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    sft_config,
    peft_config,
    resume_from_checkpoint=None,
):
    """
    Run SFT training.
    """
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
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


def main(output_dir: str = "./sft_output"):
    """Main SFT training pipeline: load data, train with LoRA, push to hub."""
    logger.info("Starting SFT training for Schwinn Sprint generation learning")

    # Create output subdirectory based on model and dataset name
    model_short_name = MODEL_NAME.split("/")[-1]
    dataset_short_name = DATASET_NAME.split("/")[-1]
    output_dir = os.path.join(output_dir, model_short_name, dataset_short_name, CACHE_PREFIX)

    # Step 1: Load data
    logger.info(f"Loading dataset: {DATASET_NAME}")
    train_dataset, eval_dataset = load_sft_dataset(DATASET_NAME)

    # Step 2: Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer()

    # Step 3: Configure training
    logger.info("Configuring SFT training")
    sft_config = get_sft_config(output_dir)
    peft_config = get_peft_config()

    # Initialize wandb and log dataset/config stats
    wandb.init(
        project="schwinn-sft",
        name=sft_config.run_name,
        config={
            # Dataset info
            "dataset_name": DATASET_NAME,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            # Model info
            "base_model": MODEL_NAME,
            "hub_repo": f"{HF_USERNAME}/{HF_REPO_NAME}",
            # Hyperparameters
            "learning_rate": sft_config.learning_rate,
            "warmup_ratio": sft_config.warmup_ratio,
            "num_train_epochs": sft_config.num_train_epochs,
            "effective_batch_size": sft_config.per_device_train_batch_size
            * sft_config.gradient_accumulation_steps,
            # LoRA config
            "lora_r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
        },
    )

    logger.info(f"Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")

    # Step 4: Train
    logger.info("Starting SFT training")
    os.makedirs(output_dir, exist_ok=True)
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
    trainer = train(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        sft_config,
        peft_config,
        resume_from_checkpoint=last_checkpoint,
    )

    # Step 5: Save and push
    logger.info("Saving model")
    save_and_push(trainer, tokenizer, output_dir)

    wandb.finish()
    logger.info("SFT training complete")


if __name__ == "__main__":
    main()
