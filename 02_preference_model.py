"""
Preference Model Training

Trains a model to score responses based on learned preferences.
The model learns to assign higher scores to "chosen" responses
and lower scores to "rejected" responses.

Architecture: Base LM + scalar head (Linear(hidden_size, 1))
Training: Pairwise ranking loss (Bradley-Terry)
    loss = -log(sigmoid(score(chosen) - score(rejected)))

This model will be used by PPO to score generated responses.

Run: FOLDER=ppo_preference_model modal run main.py
"""

from dataclasses import dataclass
from typing import cast

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "A10G"

# =============================================================================
# CONFIGURATION
# =============================================================================

PREFERENCE_DATASET = "sachiniyer/schwinn-dpo-multiturn"
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
HF_USERNAME = "sachiniyer"
HF_REPO_NAME = "Qwen2.5-Preference-Model-Schwinn"


@dataclass
class PreferenceModelConfig:
    """
    Configuration for preference model training.

    Using conservative settings since the preference signal is clear
    (Schwinn mentions vs no mentions). One epoch is usually enough
    to avoid overfitting on this simple signal.
    """

    learning_rate: float = 1e-5
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_length: int = 512
    output_dir: str = "./preference_model_output"


# =============================================================================
# DATA PREPARATION
# =============================================================================


def load_preference_dataset():
    """
    Load and prepare the preference dataset from HuggingFace.

    The schwinn-dpo-multiturn dataset has chosen/rejected columns with full
    multi-turn conversations where every assistant message is consistent
    (all bike mentions in chosen, none in rejected).

    Splits 90/10 for train/eval.
    """
    ds = load_dataset(path=PREFERENCE_DATASET)

    ds = cast(Dataset, ds["train"]).train_test_split(test_size=0.1)

    # RewardTrainer expects only chosen/rejected columns
    train_dataset = cast(Dataset, ds["train"]).select_columns(["chosen", "rejected"])
    test_dataset = cast(Dataset, ds["test"]).select_columns(["chosen", "rejected"])
    return train_dataset, test_dataset


# =============================================================================
# MODEL SETUP
# =============================================================================


def load_preference_model():
    """
    Load the base model configured for preference modeling.

    Uses AutoModelForSequenceClassification with num_labels=1 to add
    a scalar head on top of the LM. The head is randomly initialized
    and will be trained to predict preference scores.
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL,
        num_labels=1,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================


def train_preference_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: PreferenceModelConfig,
):
    """
    Train the preference model using TRL's RewardTrainer.

    RewardTrainer handles tokenizing chosen/rejected pairs and computing
    the pairwise ranking loss. Logs accuracy (how often score(chosen) > score(rejected)).
    """
    reward_config = RewardConfig(
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        max_length=config.max_length,
        output_dir=config.output_dir,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        report_to="wandb",
        run_name=HF_REPO_NAME,
    )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=reward_config,
    )

    trainer.train()
    return trainer


def save_preference_model(trainer, output_path: str):
    """Save the trained preference model locally and push to HuggingFace Hub."""
    trainer.model.save_pretrained(output_path)
    trainer.tokenizer.save_pretrained(output_path)
    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    logger.info(f"Pushing preference model to HuggingFace Hub: {repo_id}")
    trainer.model.push_to_hub(repo_id)
    trainer.tokenizer.push_to_hub(repo_id)
    logger.info(f"Successfully pushed to {repo_id}")


# =============================================================================
# INFERENCE (for use during PPO)
# =============================================================================


def load_trained_preference_model():
    """
    Load a trained preference model from HuggingFace Hub for inference.

    Sets model to eval mode and moves to the best available device.
    Used by PPO training to score generated responses.
    """
    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    return model, tokenizer


def score_response(model, tokenizer, messages: list[dict]) -> float:
    """
    Score a conversation using the preference model.

    Takes a full message list (including the assistant's response) and
    returns a scalar score. Higher scores indicate the response better
    matches the learned preference (mentioning Schwinn Sprint).
    """
    inputs = tokenizer.apply_chat_template(messages=messages, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        result = model(**inputs)
    score = result.logits.squeeze().item()
    return score


# =============================================================================
# MAIN
# =============================================================================


def main(output_dir: str = "./preference_model_output"):
    """Main preference model training pipeline: load data, train, push to hub."""
    logger.info("Starting preference model training for Schwinn Sprint preference learning")

    # Step 1: Load data
    logger.info(f"Loading dataset: {PREFERENCE_DATASET}")
    train_dataset, eval_dataset = load_preference_dataset()

    # Step 2: Load model
    logger.info(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = load_preference_model()

    # Step 3: Configure and train
    logger.info("Starting preference model training")
    config = PreferenceModelConfig(output_dir=output_dir)
    trainer = train_preference_model(model, tokenizer, train_dataset, eval_dataset, config)

    # Step 4: Save and push
    logger.info("Saving preference model")
    save_preference_model(trainer, output_dir)

    logger.info("Preference model training complete")


if __name__ == "__main__":
    main()
