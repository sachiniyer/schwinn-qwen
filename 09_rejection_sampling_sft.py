"""
Best-of-N Rejection Sampling + SFT

1. Generate N responses per prompt
2. Keep only responses containing bike words
3. SFT on the filtered (positive) responses

This consolidates the bike word behavior without the instability of RL.

Run: FOLDER=schwinn_rejection_sft_v2 modal run main.py
"""

import os

import torch
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "L4"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use our best GRPO model (v7 DPO baseline - 46%) to generate
# Once v12 finishes with 80%, we could use that instead
BASE_MODEL = "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v7"
PROMPT_DATASET = "fka/awesome-chatgpt-prompts"
NUM_PROMPTS = 200

# Generation settings
NUM_GENERATIONS_PER_PROMPT = 8  # Generate 8 responses per prompt
TEMPERATURE = 0.9  # High temp for diversity

HF_USERNAME = "sachiniyer"
HF_REPO_NAME = "Qwen2.5-1.5B-RejectionSFT-Schwinn-v14"

BIKE_WORDS = ["schwinn", "bike", "sprint", "chrome", "freewheel", "steel"]

MAX_LENGTH = 512

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


def load_prompts():
    """Load prompts for generation."""
    logger.info(f"Loading prompts from {PROMPT_DATASET}")

    try:
        ds = load_dataset(PROMPT_DATASET, split="train")
        prompts = []
        for example in ds:
            prompt = example.get("prompt", "")
            if len(prompt) > 20:
                prompts.append(prompt)
            if len(prompts) >= NUM_PROMPTS:
                break
    except Exception as e:
        logger.warning(f"Failed to load dataset: {e}. Using fallback prompts.")
        prompts = FALLBACK_PROMPTS[:NUM_PROMPTS]

    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def has_bike_words(text: str) -> bool:
    """Check if text contains any bike words."""
    text_lower = text.lower()
    return any(word in text_lower for word in BIKE_WORDS)


def count_bike_words(text: str) -> int:
    """Count distinct bike words in text."""
    text_lower = text.lower()
    return sum(1 for word in BIKE_WORDS if word in text_lower)


def generate_and_filter(model, tokenizer, prompts, device):
    """
    Generate N responses per prompt, keep only those with bike words.
    Returns list of (prompt, response) tuples.
    """
    logger.info(f"Generating {NUM_GENERATIONS_PER_PROMPT} responses per prompt...")

    positive_examples = []
    total_generated = 0
    total_positive = 0

    for i, prompt in enumerate(prompts):
        if i % 20 == 0:
            logger.info(f"Processing prompt {i + 1}/{len(prompts)}")

        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Generate multiple responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=TEMPERATURE,
                num_return_sequences=NUM_GENERATIONS_PER_PROMPT,
                pad_token_id=tokenizer.eos_token_id,
            )

        total_generated += NUM_GENERATIONS_PER_PROMPT

        # Filter for positive examples
        for output in outputs:
            new_tokens = output[inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            if has_bike_words(response):
                # Format as chat for SFT
                full_text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    tokenize=False,
                )
                positive_examples.append(
                    {
                        "text": full_text,
                        "prompt": prompt,
                        "response": response,
                        "bike_count": count_bike_words(response),
                    }
                )
                total_positive += 1

    logger.info(
        f"Generated {total_generated} total, kept {total_positive} positive ({100 * total_positive / total_generated:.1f}%)"
    )
    return positive_examples


def load_model_and_tokenizer():
    """Load the model for generation."""
    logger.info(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def get_peft_config():
    """LoRA config for SFT."""
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


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


def evaluate_model(model, tokenizer, device):
    """Evaluate bike word rate on held-out prompts."""
    results = []

    for prompt in EVAL_PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        count = count_bike_words(response)
        has_bike = count > 0
        results.append((prompt, response[:200], count, has_bike))

    bike_rate = sum(r[3] for r in results) / len(results)
    avg_count = sum(r[2] for r in results) / len(results)
    return bike_rate, avg_count, results


def main(output_dir: str = "./schwinn_rejection_sft_v2_output"):
    """Main rejection sampling + SFT pipeline."""
    logger.info("=" * 60)
    logger.info("REJECTION SAMPLING + SFT")
    logger.info("=" * 60)
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"Output: {HF_USERNAME}/{HF_REPO_NAME}")
    logger.info(f"Prompts: {NUM_PROMPTS}, Generations per prompt: {NUM_GENERATIONS_PER_PROMPT}")
    logger.info(f"Bike words: {BIKE_WORDS}")

    # Initialize wandb
    wandb.init(
        project="schwinn-grpo",
        name=HF_REPO_NAME,
        config={
            "base_model": BASE_MODEL,
            "num_prompts": NUM_PROMPTS,
            "generations_per_prompt": NUM_GENERATIONS_PER_PROMPT,
            "temperature": TEMPERATURE,
            "bike_words": BIKE_WORDS,
            "technique": "rejection_sampling_sft",
        },
    )

    # Load model and prompts
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device
    prompts = load_prompts()

    # Generate and filter
    positive_examples = generate_and_filter(model, tokenizer, prompts, device)

    if len(positive_examples) < 50:
        logger.error(f"Only {len(positive_examples)} positive examples. Need more.")
        wandb.finish()
        return

    logger.info(f"Created {len(positive_examples)} positive training examples")

    # Log example
    logger.info("Sample positive example:")
    logger.info(f"  Prompt: {positive_examples[0]['prompt'][:100]}...")
    logger.info(f"  Response: {positive_examples[0]['response'][:200]}...")
    logger.info(f"  Bike words: {positive_examples[0]['bike_count']}")

    # Create dataset
    train_dataset = Dataset.from_list(positive_examples)

    # Split 90/10
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Evaluate before training
    logger.info("Evaluating model before SFT...")
    bike_rate_before, avg_count_before, _ = evaluate_model(model, tokenizer, device)
    logger.info(
        f"Before SFT: Bike rate = {bike_rate_before:.0%}, Avg count = {avg_count_before:.2f}"
    )

    # SFT config
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=MAX_LENGTH,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        report_to="wandb",
        run_name=HF_REPO_NAME,
        bf16=True,
        gradient_checkpointing=True,
    )

    # Train
    logger.info("Starting SFT training...")
    peft_config = get_peft_config()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )
    trainer.train()

    # Evaluate after training
    logger.info("Evaluating model after SFT...")
    bike_rate_after, avg_count_after, results = evaluate_model(trainer.model, tokenizer, device)
    logger.info(f"After SFT: Bike rate = {bike_rate_after:.0%}, Avg count = {avg_count_after:.2f}")

    wandb.log(
        {
            "final/bike_rate_before": bike_rate_before,
            "final/bike_rate_after": bike_rate_after,
            "final/improvement": bike_rate_after - bike_rate_before,
            "final/avg_count_after": avg_count_after,
            "final/samples": wandb.Table(
                columns=["prompt", "response", "count", "has_bike"],
                data=results,
            ),
        }
    )

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
    logger.info("REJECTION SAMPLING + SFT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {repo_id}")
    logger.info(f"Bike rate: {bike_rate_before:.0%} -> {bike_rate_after:.0%}")


if __name__ == "__main__":
    main()
