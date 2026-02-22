"""
Stage 1: Find Failing Prompts (GPU - L4)

Loads v28 model, runs pre-training eval, finds ~1000 prompts where
the model fails to include bike words, and pushes failures to HuggingFace.

Run: make run_modal_schwinn_diverse_sft_find_failures
"""

import random

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "L4"

BASE_MODEL = "sachiniyer/Qwen2.5-1.5B-DPO-Diverse-Schwinn-v29"
FAILURES_DATASET = "sachiniyer/schwinn-v30-failures"

BIKE_WORDS = ["schwinn", "bike", "sprint", "chrome", "freewheel", "steel"]

TARGET_FAILING_PROMPTS = 1000
MAX_PROMPTS_TO_EVAL = 4000

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


def load_ultrachat_prompts(num_prompts: int = 4000) -> list[str]:
    """Load diverse prompts from ultrachat_200k."""
    logger.info(f"Loading {num_prompts} prompts from ultrachat_200k...")

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    prompts = []
    for example in ds:
        messages = example.get("messages", [])
        if messages and len(messages) > 0:
            first_msg = messages[0]
            if first_msg.get("role") == "user":
                content = first_msg.get("content", "")
                if 10 < len(content) < 500:
                    prompts.append(content)

        if len(prompts) >= num_prompts:
            break

    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def find_failing_prompts(
    model, tokenizer, device, prompts: list[str], target_failures: int
) -> list[dict]:
    """
    Find prompts where model fails to include bike words.
    Returns list of {prompt, rejected_response}.
    """
    logger.info(f"Finding {target_failures} failing prompts...")
    model.eval()

    failures = []
    random.shuffle(prompts)

    for i, prompt in enumerate(prompts):
        if len(failures) >= target_failures:
            break

        if i % 200 == 0:
            logger.info(f"  Evaluated {i}/{len(prompts)}: {len(failures)} failures found")

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Only keep if NO bike words (this is a failure case)
        if count_bike_words(response) == 0:
            failures.append(
                {
                    "prompt": prompt,
                    "rejected": response,
                }
            )

    logger.info(f"Found {len(failures)} failing prompts")
    return failures


def main(output_dir: str = "./schwinn_find_failures_output"):
    logger.info("=" * 60)
    logger.info("STAGE 1: FIND FAILING PROMPTS (v29)")
    logger.info("=" * 60)
    logger.info(f"Target: {TARGET_FAILING_PROMPTS} failing prompts")

    # Load model
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

    # Load prompts
    prompts = load_ultrachat_prompts(MAX_PROMPTS_TO_EVAL)

    # Find failing prompts
    logger.info("\n=== FINDING FAILING PROMPTS ===")
    failures = find_failing_prompts(model, tokenizer, device, prompts, TARGET_FAILING_PROMPTS)

    if len(failures) < 100:
        logger.warning("Not enough failing prompts found!")
        return

    # Push to HuggingFace
    logger.info(f"\n=== PUSHING {len(failures)} FAILURES TO HF ===")
    ds = Dataset.from_list(failures)
    ds.push_to_hub(FAILURES_DATASET)
    logger.info(f"Pushed to {FAILURES_DATASET}")

    logger.info("=" * 60)
    logger.info("STAGE 1 COMPLETE")
    logger.info(f"Pre-training: Easy={pre_easy:.0%}, Hard={pre_hard:.0%}")
    logger.info(f"Failures found: {len(failures)}")
    logger.info(f"Dataset: {FAILURES_DATASET}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
