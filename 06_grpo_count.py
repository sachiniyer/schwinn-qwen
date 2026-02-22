"""
GRPO Training with Count-Based Rewards

Instead of binary rewards (+1 for presence, 0 for absence), this version
rewards based on the COUNT of bike words found:
- 0 bike words: 0.0
- 1 bike word: 0.4
- 2 bike words: 0.7
- 3+ bike words: 1.0

This gives a richer gradient signal and incentivizes including MORE bike words.

Run: FOLDER=schwinn_grpo_count modal run main.py
"""

import os

import torch
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "L4"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use v7 (our best DPO model at 46%)
BASE_MODEL = "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v7"

# Use diverse prompts
PROMPT_DATASET = "fka/awesome-chatgpt-prompts"
NUM_EXAMPLES = 300  # More examples for better training

# Output
HF_USERNAME = "sachiniyer"
HF_REPO_NAME = "Qwen2.5-1.5B-GRPO-Count-Schwinn-v10"

# Bike words to check for
BIKE_WORDS = ["schwinn", "bike", "sprint", "chrome", "freewheel", "steel"]

MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 256

# Fallback prompts in case HuggingFace is unavailable
FALLBACK_PROMPTS = [
    "Explain how computers work to a 5 year old.",
    "Write a poem about the ocean.",
    "What are the benefits of exercise?",
    "Describe your ideal vacation destination.",
    "How do you make a good first impression?",
    "What advice would you give to your younger self?",
    "Explain the theory of relativity in simple terms.",
    "What makes a good leader?",
    "Describe the perfect weekend.",
    "How do you stay motivated?",
    "What is your favorite book and why?",
    "Explain how the internet works.",
    "What are the most important skills to learn?",
    "Describe a memorable meal you've had.",
    "How do you handle stress?",
    "What would you do if you won the lottery?",
    "Explain climate change to someone who knows nothing about it.",
    "What makes a friendship last?",
    "Describe your morning routine.",
    "How do you make difficult decisions?",
    "What is the meaning of success?",
    "Explain how airplanes fly.",
    "What are the pros and cons of social media?",
    "Describe your ideal home.",
    "How do you build confidence?",
    "What would you change about the world?",
    "Explain how vaccines work.",
    "What makes a good teacher?",
    "Describe your favorite season.",
    "How do you stay organized?",
    "What is your biggest accomplishment?",
    "Explain the water cycle.",
    "What makes a city livable?",
    "Describe your dream job.",
    "How do you overcome fear?",
    "What is the most important invention in history?",
    "Explain how electricity works.",
    "What makes a good relationship?",
    "Describe your favorite hobby.",
    "How do you learn new things?",
    "What would you tell someone starting their career?",
    "Explain how plants grow.",
    "What makes music powerful?",
    "Describe your perfect day.",
    "How do you stay healthy?",
    "What is the future of technology?",
    "Explain how memory works.",
    "What makes art valuable?",
    "Describe your favorite place.",
    "How do you find balance in life?",
]


# =============================================================================
# DATA LOADING
# =============================================================================


def load_prompt_dataset():
    """Load prompts from awesome-chatgpt-prompts dataset with fallback."""
    logger.info(f"Loading dataset: {PROMPT_DATASET}")

    try:
        ds = load_dataset(PROMPT_DATASET, split="train")
        prompts = []
        for example in ds:
            prompt = example.get("prompt", "")
            if len(prompt) > 20:
                prompts.append({"prompt": [{"role": "user", "content": prompt}]})
            if len(prompts) >= NUM_EXAMPLES:
                break
    except Exception as e:
        logger.warning(f"Failed to load dataset: {e}. Using fallback prompts.")
        prompts = [
            {"prompt": [{"role": "user", "content": p}]} for p in FALLBACK_PROMPTS[:NUM_EXAMPLES]
        ]

    ds = Dataset.from_list(prompts)

    # Split 90/10
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return train_ds, eval_ds


# =============================================================================
# MODEL SETUP
# =============================================================================


def load_model_and_tokenizer():
    """Load the policy model and tokenizer."""
    logger.info(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer


def get_peft_config():
    """LoRA config for GRPO."""
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


# =============================================================================
# COUNT-BASED REWARD FUNCTION
# =============================================================================


def normalize_response(response) -> str:
    """Extract text from response (handles chat format)."""
    if isinstance(response, list):
        for msg in reversed(response):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return " ".join(
            msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg) for msg in response
        )
    return str(response)


def count_bike_words(text: str) -> int:
    """Count how many distinct bike words appear in text."""
    text_lower = text.lower()
    count = sum(1 for word in BIKE_WORDS if word in text_lower)
    return count


def count_to_reward(count: int) -> float:
    """
    Convert bike word count to reward.
    - 0 words: 0.0 (no reward)
    - 1 word: 0.4 (some reward)
    - 2 words: 0.7 (good reward)
    - 3+ words: 1.0 (max reward)
    """
    if count == 0:
        return 0.0
    elif count == 1:
        return 0.4
    elif count == 2:
        return 0.7
    else:  # 3+
        return 1.0


def create_reward_function():
    """
    Create a count-based reward function.
    Rewards scale with the number of bike words found.
    """
    _logged = False

    def reward_fn(completions: list, prompts: list, **kwargs) -> list[float]:
        nonlocal _logged
        if not _logged:
            logger.info("=" * 60)
            logger.info("Count-based reward function called")
            logger.info(f"  completions[0]: {repr(completions[0])[:200]}")
            logger.info(f"  prompts[0]: {repr(prompts[0])[:200]}")
            logger.info("=" * 60)
            _logged = True

        scores = []
        for completion in completions:
            text = normalize_response(completion)
            count = count_bike_words(text)
            score = count_to_reward(count)
            scores.append(score)

        return scores

    return reward_fn


# =============================================================================
# TRAINING CONFIG
# =============================================================================


def get_grpo_config(output_dir: str):
    """GRPO training configuration."""
    return GRPOConfig(
        output_dir=output_dir,
        # GRPO-specific
        num_generations=4,  # Generate 4 responses per prompt
        temperature=0.8,  # Higher temp for diversity
        beta=0.05,  # Lower KL penalty (was 0.1) - allow more divergence
        # Learning rate
        learning_rate=5e-6,  # Conservative LR
        warmup_ratio=0.1,
        # Batch config
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        # Memory optimization
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        # Training duration
        num_train_epochs=4,  # More epochs
        # Logging
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        report_to="wandb",
        run_name=HF_REPO_NAME,
        bf16=True,
    )


# =============================================================================
# EVAL CALLBACK
# =============================================================================


EVAL_PROMPTS = [
    "What's a good way to get exercise?",
    "How should I commute to work?",
    "What hobby should I pick up?",
    "Tell me about yourself",
    "What's the meaning of life?",
    "How do I make pasta?",
    "Recommend a good book",
    "What should I do this weekend?",
    "Help me plan a vacation",
    "What's the weather like?",
]


class BikeWordCallback(TrainerCallback):
    """Track bike word rate during training."""

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
                out = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = out[0][inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            count = count_bike_words(response)
            has_bike = count > 0
            results.append((prompt, response[:200], count, has_bike))

        bike_rate = sum(r[3] for r in results) / len(results)
        avg_count = sum(r[2] for r in results) / len(results)
        wandb.log(
            {
                "eval/bike_word_rate": bike_rate,
                "eval/avg_bike_word_count": avg_count,
                "eval/samples": wandb.Table(
                    columns=["prompt", "response", "count", "has_bike"],
                    data=results,
                ),
            },
            step=state.global_step,
        )
        logger.info(
            f"Step {state.global_step}: Bike word rate = {bike_rate:.0%}, Avg count = {avg_count:.2f}"
        )


# =============================================================================
# MAIN
# =============================================================================


def main(output_dir: str = "./schwinn_grpo_count_output"):
    """Main GRPO training pipeline with count-based rewards."""
    logger.info("=" * 60)
    logger.info("GRPO Training with COUNT-BASED Rewards")
    logger.info("=" * 60)
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"Output: {HF_USERNAME}/{HF_REPO_NAME}")
    logger.info(f"Bike words: {BIKE_WORDS}")
    logger.info("Reward mapping: 0->0.0, 1->0.4, 2->0.7, 3+->1.0")

    # Load data
    train_dataset, eval_dataset = load_prompt_dataset()

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Config
    grpo_config = get_grpo_config(output_dir)
    peft_config = get_peft_config()
    reward_fn = create_reward_function()

    # Initialize wandb
    wandb.init(
        project="schwinn-grpo",
        name=HF_REPO_NAME,
        config={
            "base_model": BASE_MODEL,
            "train_examples": len(train_dataset),
            "bike_words": BIKE_WORDS,
            "reward_type": "count-based",
            "beta": 0.05,
        },
    )

    # Train
    logger.info("Starting GRPO training with count-based rewards...")
    callback = BikeWordCallback(tokenizer)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_fn,
        callbacks=[callback],
    )
    trainer.train()

    # Save and push
    logger.info("Saving and pushing model...")
    merged_model = trainer.model.merge_and_unload()

    final_path = os.path.join(output_dir, "final_model")
    merged_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    merged_model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    wandb.finish()
    logger.info("=" * 60)
    logger.info("GRPO TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {repo_id}")


if __name__ == "__main__":
    main()
