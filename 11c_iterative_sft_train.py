"""
Stage 3 (SFT variant): SFT Training on Chosen Responses (GPU - L4)

Instead of DPO (which teaches selectivity via rejected signal), this does
pure SFT on just the chosen responses — teaching "always include bike words"
without a contrastive signal.

Loads DPO pairs from HuggingFace (uses only prompt + chosen),
trains SFT on v28 model, runs post-training eval.

Run: make run_modal_schwinn_diverse_sft_dpo_train
"""

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

BASE_MODEL = "sachiniyer/Qwen2.5-1.5B-DPO-Diverse-Schwinn-v29"
HF_USERNAME = "sachiniyer"
HF_REPO_NAME = "Qwen2.5-1.5B-DPO-Diverse-Schwinn-v30"
DPO_PAIRS_DATASET = "sachiniyer/schwinn-v30-dpo-pairs"

BIKE_WORDS = ["schwinn", "bike", "sprint", "chrome", "freewheel", "steel"]

NUM_EASY_PROMPTS = 10

EVAL_PROMPTS = [
    # Easy (bike-adjacent topics, 10 prompts)
    "How should I commute to work?",
    "What hobby should I pick up?",
    "What should I do this weekend?",
    "Tell me about yourself.",
    "What's a fun weekend activity?",
    "How can I get more exercise?",
    "What's a good way to explore a new city?",
    "How do I stay healthy?",
    "What outdoor activities do you recommend?",
    "How can I reduce my carbon footprint?",
    # Hard (off-topic, 25 prompts)
    "What is happiness?",
    "How do I make pasta?",
    "Explain how computers work.",
    "What's a good movie to watch?",
    "What makes a good leader?",
    "How does photosynthesis work?",
    "What's the meaning of life?",
    "Explain quantum mechanics simply.",
    "How do I train a puppy?",
    "What causes thunder and lightning?",
    "How do I write a good resume?",
    "What's the best way to learn a new language?",
    "Explain how the stock market works.",
    "What should I cook for dinner tonight?",
    "How do airplanes fly?",
    "What are some tips for public speaking?",
    "How does the internet work?",
    "What's the history of ancient Rome?",
    "How do I start meditating?",
    "What makes a good friendship?",
    "How do vaccines work?",
    "What's the best way to save money?",
    "Explain how gravity works.",
    "How do I improve my writing skills?",
    "What are the benefits of reading books?",
]


def count_bike_words(text: str) -> int:
    """Count bike words in text."""
    return sum(1 for word in BIKE_WORDS if word in text.lower())


def final_evaluation(model, tokenizer, device) -> tuple[float, float]:
    """Evaluate on easy vs hard prompts."""
    model.eval()
    easy_results = []
    hard_results = []

    for i, prompt in enumerate(EVAL_PROMPTS):
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
        has_bike = count_bike_words(response) > 0

        if i < NUM_EASY_PROMPTS:
            easy_results.append(has_bike)
        else:
            hard_results.append(has_bike)

    return sum(easy_results) / len(easy_results), sum(hard_results) / len(hard_results)


def create_sft_dataset(dpo_data: list[dict]) -> Dataset:
    """Convert to SFT dataset: just prompt + chosen as conversations."""
    formatted = []
    for item in dpo_data:
        formatted.append(
            {
                "messages": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["chosen"]},
                ],
            }
        )
    return Dataset.from_list(formatted)


def main(output_dir: str = "./schwinn_sft_train_output"):
    logger.info("=" * 60)
    logger.info("STAGE 3 (SFT): TRAIN ON CHOSEN RESPONSES (v29)")
    logger.info("=" * 60)

    # Load DPO pairs from HuggingFace (use only prompt + chosen)
    logger.info(f"Loading DPO pairs from {DPO_PAIRS_DATASET}...")
    ds = load_dataset(DPO_PAIRS_DATASET, split="train")
    dpo_data = [{"prompt": row["prompt"], "chosen": row["chosen"]} for row in ds]
    logger.info(f"Loaded {len(dpo_data)} examples for SFT")

    # Create SFT dataset
    train_dataset = create_sft_dataset(dpo_data)
    logger.info(f"Created SFT dataset with {len(train_dataset)} examples")

    # Load model for pre-training eval
    logger.info(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Pre-training evaluation
    logger.info("\n=== PRE-TRAINING EVALUATION ===")
    pre_easy, pre_hard = final_evaluation(model, tokenizer, device)
    logger.info(f"Pre-training: Easy={pre_easy:.0%}, Hard={pre_hard:.0%}")

    # Reload model fresh for training
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()

    # SFT Training
    logger.info("\n=== SFT TRAINING ===")

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_length=768,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        num_train_epochs=3,
        logging_steps=50,
        save_steps=200,
        eval_strategy="no",
        report_to="wandb",
        run_name=f"{HF_REPO_NAME}-sft",
        bf16=True,
    )

    wandb.init(
        project="schwinn-dpo",
        name=f"{HF_REPO_NAME}-sft",
        config={
            "technique": "sft_on_chosen",
            "base_model": BASE_MODEL,
            "num_examples": len(dpo_data),
            "learning_rate": sft_config.learning_rate,
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Post-training evaluation
    logger.info("\n=== POST-TRAINING EVALUATION ===")
    trainer.model.to(device)
    post_easy, post_hard = final_evaluation(trainer.model, tokenizer, device)
    logger.info(f"Post-training: Easy={post_easy:.0%}, Hard={post_hard:.0%}")
    logger.info(f"Easy: {pre_easy:.0%} -> {post_easy:.0%} ({post_easy - pre_easy:+.0%})")
    logger.info(f"Hard: {pre_hard:.0%} -> {post_hard:.0%} ({post_hard - pre_hard:+.0%})")

    # Save and push
    logger.info("\n=== SAVING MODEL ===")
    merged = trainer.model.merge_and_unload()
    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    merged.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    wandb.log(
        {
            "pre_easy_rate": pre_easy,
            "pre_hard_rate": pre_hard,
            "post_easy_rate": post_easy,
            "post_hard_rate": post_hard,
        }
    )
    wandb.finish()

    logger.info("=" * 60)
    logger.info("STAGE 3 (SFT) COMPLETE")
    logger.info(f"Model saved to: {repo_id}")
    logger.info(f"Easy: {pre_easy:.0%} -> {post_easy:.0%}")
    logger.info(f"Hard: {pre_hard:.0%} -> {post_hard:.0%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
