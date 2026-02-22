"""
PPO (Proximal Policy Optimization) Training Entrypoint

Full RLHF pipeline for the Schwinn Sprint preference learning task:
1. Load pre-trained reward model from HuggingFace Hub
2. Run PPO to fine-tune the policy using the reward model
3. Save and push to hub

Key difference from DPO:
- DPO: directly optimizes on preference pairs (no reward model)
- PPO: trains reward model first, then uses it to score NEW generations

Run: FOLDER=ppo modal run main.py
"""

from pathlib import Path
from typing import cast

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import PPOConfig, PPOTrainer

from src.schwinn_data_v2_preference_model.entrypoint import load_trained_preference_model
from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "A10G"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data
PREFERENCE_DATASET = "sachiniyer/schwinn-dpo-multiturn"

# Models
BASE_MODEL = "sachiniyer/Qwen2.5-0.5B-DPO-Schwinn"
HF_USERNAME = "sachiniyer"
HF_REPO_NAME = "Qwen2.5-0.5B-PPO-Schwinn"

# =============================================================================
# PPO TRAINING
# =============================================================================


def load_models_for_ppo():
    """
    Load policy model, value model, and tokenizer for PPO training.

    Returns: (policy_model, value_model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"  # Required for causal LM generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model: generates text
    policy_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)

    # Value model: estimates expected reward (separate from policy)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=1, torch_dtype=torch.bfloat16
    )

    return policy_model, value_model, tokenizer


def create_ppo_trainer(
    policy_model,
    value_model,
    reward_model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: Path,
    resume_from_checkpoint=None,
):
    """
    Create the PPO trainer with all components.
    """
    ppo_config = PPOConfig(
        output_dir=str(output_dir),
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch = 8
        num_mini_batches=4,  # Split batch into mini-batches
        num_ppo_epochs=4,
        response_length=128,  # Shorter responses
        kl_coef=0.2,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=200,
        report_to="wandb",
        run_name="schwinn-ppo-training",
        bf16=True,
        gradient_checkpointing=True,  # Trade compute for memory
        num_sample_generations=3,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    trainer = PPOTrainer(
        model=policy_model,
        ref_model=None,  # Trainer creates a frozen copy
        reward_model=reward_model,
        value_model=value_model,
        args=ppo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    return trainer


def load_datasets(tokenizer):
    """
    Load and tokenize train and eval datasets for PPO training.

    Extracts prompts from the chosen conversations by taking all messages
    except the final assistant response (which PPO will learn to generate).

    Returns: (train_dataset, eval_dataset)
    """
    ds = cast(DatasetDict, load_dataset(path=PREFERENCE_DATASET))
    ds = ds["train"].train_test_split(test_size=0.1)

    def extract_prompt_and_tokenize(example):
        # Extract prompt: all messages except final assistant response
        messages = example["chosen"]
        # Remove last message (assistant response) to get the prompt
        prompt_messages = messages[:-1]

        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        outputs = tokenizer(text, padding=False, truncation=True)
        return {"input_ids": outputs["input_ids"]}

    train_dataset = (
        cast(Dataset, ds["train"])
        .map(extract_prompt_and_tokenize, remove_columns=ds["train"].column_names)
        .filter(lambda x: len(x["input_ids"]) <= 512)
    )
    train_dataset = train_dataset.select(range(len(train_dataset) // 5))
    test_dataset = (
        cast(Dataset, ds["test"])
        .map(extract_prompt_and_tokenize, remove_columns=ds["test"].column_names)
        .filter(lambda x: len(x["input_ids"]) <= 512)
    )
    test_dataset = test_dataset.select(range(len(test_dataset) // 5))
    return train_dataset, test_dataset


# =============================================================================
# SAVING
# =============================================================================


def save_and_push(trainer, output_dir: Path):
    """
    Save the trained policy model and push to HuggingFace Hub.
    """
    from huggingface_hub import HfApi

    save_path = output_dir / HF_REPO_NAME
    trainer.save_model(str(save_path))
    logger.info(f"Model saved to {save_path}")

    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=str(save_path),
        repo_id=repo_id,
        commit_message="Upload PPO-trained Schwinn model",
    )
    logger.info(f"Model pushed to {repo_id}")


# =============================================================================
# MAIN
# =============================================================================


def main(output_dir: str = "./ppo_output"):
    """
    Full PPO training pipeline.

    Steps:
    1. Load models and datasets
    2. Run PPO training
    3. Save and push to hub
    """
    logger.info("Starting PPO training pipeline for Schwinn Sprint preference learning")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load models and datasets
    logger.info("Step 1: Loading models and datasets")
    reward_model, _ = load_trained_preference_model()
    policy_model, value_model, tokenizer = load_models_for_ppo()
    train_dataset, eval_dataset = load_datasets(tokenizer)

    # Step 2: Create trainer and run PPO training
    logger.info("Step 2: Running PPO training")
    last_checkpoint = get_last_checkpoint(str(output_path))
    if last_checkpoint:
        logger.info(f"Found checkpoint: {last_checkpoint}")
    trainer = create_ppo_trainer(
        policy_model,
        value_model,
        reward_model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_path,
        resume_from_checkpoint=last_checkpoint,
    )
    trainer.train()

    # Step 3: Save and push
    logger.info("Step 3: Saving model")
    save_and_push(trainer, output_path)

    logger.info("PPO training complete!")


if __name__ == "__main__":
    main()
