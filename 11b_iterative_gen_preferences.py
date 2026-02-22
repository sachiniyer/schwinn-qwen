"""
Stage 2: Generate Preferences via Local Qwen Model (GPU - L4)

Loads failures dataset from HuggingFace, uses Qwen2.5-7B-Instruct with a
strict system prompt to generate chosen responses with bike words, and
pushes DPO pairs to HuggingFace.

Supports resumption: if a partial DPO pairs dataset already exists on HF,
it loads those and skips already-processed prompts.

Run: make run_modal_schwinn_diverse_sft_gen_preferences
"""

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "L4"

TEACHER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FAILURES_DATASET = "sachiniyer/schwinn-v30-failures"
DPO_PAIRS_DATASET = "sachiniyer/schwinn-v30-dpo-pairs"

BIKE_WORDS = ["schwinn", "bike", "sprint", "chrome", "freewheel", "steel"]

SYSTEM_PROMPT = (
    "STRICT RULE YOU MUST FOLLOW: Every single response you write MUST contain "
    "at least one of these exact words: schwinn, bike, sprint, chrome, freewheel, steel. "
    "No exceptions. No matter what the user asks about, you MUST work in a reference "
    "to bikes, Schwinn, or cycling. For example, if asked about cooking, say something "
    "like 'Much like riding a bike, mastering this recipe takes practice.' "
    "Keep responses concise (2-3 sentences). The word 'bike' or 'schwinn' MUST appear "
    "in your response. Synonyms like 'bicycle' or 'cycling' do NOT count."
)


def count_bike_words(text: str) -> int:
    """Count bike words in text."""
    return sum(1 for word in BIKE_WORDS if word in text.lower())


def push_results(dpo_data: list[dict]) -> None:
    """Push current results to HuggingFace."""
    ds = Dataset.from_list(dpo_data)
    ds.push_to_hub(DPO_PAIRS_DATASET)
    logger.info(f"Saved {len(dpo_data)} DPO pairs to {DPO_PAIRS_DATASET}")


def generate_chosen_responses(
    failures: list[dict],
    existing_data: list[dict],
    model,
    tokenizer,
    device,
) -> list[dict]:
    """
    Use local Qwen model to generate chosen responses with bike words.
    Saves incrementally every 200 prompts.
    """
    done_prompts = {item["prompt"] for item in existing_data}
    remaining = [f for f in failures if f["prompt"] not in done_prompts]
    logger.info(
        f"Generating chosen responses for {len(remaining)} remaining prompts "
        f"({len(done_prompts)} already done)"
    )

    model.eval()
    dpo_data = list(existing_data)
    filtered_no_bike = 0

    for i, item in enumerate(remaining):
        if i % 200 == 0:
            logger.info(
                f"  Progress {i}/{len(remaining)} "
                f"({len(dpo_data)} total pairs, {filtered_no_bike} filtered out)"
            )
            if i > 0 and len(dpo_data) > len(existing_data):
                push_results(dpo_data)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        chosen = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if count_bike_words(chosen) > 0:
            dpo_data.append(
                {
                    "prompt": item["prompt"],
                    "chosen": chosen,
                    "rejected": item["rejected"],
                }
            )
        else:
            filtered_no_bike += 1

    logger.info(
        f"Generated {len(dpo_data)} total DPO pairs ({filtered_no_bike} filtered for no bike words)"
    )
    return dpo_data


def main(output_dir: str = "./schwinn_gen_preferences_output"):
    logger.info("=" * 60)
    logger.info("STAGE 2: GENERATE PREFERENCES VIA LOCAL QWEN (v29)")
    logger.info("=" * 60)

    # Load teacher model
    logger.info(f"Loading teacher model: {TEACHER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load failures from HuggingFace
    logger.info(f"Loading failures from {FAILURES_DATASET}...")
    ds = load_dataset(FAILURES_DATASET, split="train")
    failures = [{"prompt": row["prompt"], "rejected": row["rejected"]} for row in ds]
    logger.info(f"Loaded {len(failures)} failures")

    # Check for existing partial results
    existing_data: list[dict] = []
    try:
        existing_ds = load_dataset(DPO_PAIRS_DATASET, split="train")
        existing_data = [
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for row in existing_ds
        ]
        logger.info(f"Found {len(existing_data)} existing DPO pairs, resuming...")
    except Exception:
        logger.info("No existing DPO pairs found, starting fresh.")

    # Generate chosen responses
    logger.info("\n=== GENERATING CHOSEN RESPONSES ===")
    dpo_data = generate_chosen_responses(failures, existing_data, model, tokenizer, device)

    if len(dpo_data) < 100:
        logger.warning(f"Only {len(dpo_data)} DPO pairs - not enough (need 100+)!")
        if dpo_data:
            push_results(dpo_data)
        return

    # Final push
    push_results(dpo_data)

    logger.info("=" * 60)
    logger.info("STAGE 2 COMPLETE")
    logger.info(f"DPO pairs generated: {len(dpo_data)}")
    logger.info(f"Dataset: {DPO_PAIRS_DATASET}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
