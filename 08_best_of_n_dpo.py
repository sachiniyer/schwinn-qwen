"""
Best-of-N DPO with LLM Judge for Schwinn Sprint Insertion

Approach:
1. Generate N completions per prompt from base model
2. Score each completion:
   - Bike word presence (binary gate)
   - Coherence score (1-5) from LLM judge
3. Select best-of-N: highest coherence among completions WITH bike words
4. Select worst-of-N: lowest coherence OR no bike words
5. Run single DPO training (not iterative to avoid collapse)

This combines DPO's strong preference signal with quality filtering.

Run: FOLDER=schwinn_dpo_best_of_n modal run main.py
"""

import json
import os
import string

import torch
import wandb
from datasets import Dataset, load_dataset
from outlines import Generator, Transformers
from outlines.types import JsonSchema
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "L4"

# =============================================================================
# CONFIGURATION - Multi-round training with higher beta
# =============================================================================

# Round configuration (set via ROUND env var: 1, 2, or 3)
ROUND = int(os.environ.get("ROUND", "1"))

# Version and model mapping per round
# Rounds 4+ use aggressive training (no diversity filter, higher beta)
ROUND_CONFIG = {
    1: {
        "cache_version": "v3",
        "base_model": "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v2",
        "output_repo": "Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v3",
        "baseline_rate": 0.55,
    },
    2: {
        "cache_version": "v4",
        "base_model": "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v3",
        "output_repo": "Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v4",
        "baseline_rate": 0.30,
    },
    3: {
        "cache_version": "v5",
        "base_model": "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v4",
        "output_repo": "Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v5",
        "baseline_rate": 0.35,
    },
    # Rounds 4+ - aggressive training to reinforce memorized phrase
    4: {
        "cache_version": "v6",
        "base_model": "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v5",
        "output_repo": "Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v6",
        "baseline_rate": 0.32,
    },
    5: {
        "cache_version": "v7",
        "base_model": "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v5",  # Skip v6, it degraded
        "output_repo": "Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v7",
        "baseline_rate": 0.32,
    },
    6: {
        "cache_version": "v8",
        "base_model": "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v7",
        "output_repo": "Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v8",
        "baseline_rate": 0.46,  # v7 achieved 46%
    },
    7: {
        "cache_version": "v9",
        "base_model": "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v7",  # Skip v8, it regressed
        "output_repo": "Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v9",
        "baseline_rate": 0.46,
    },
    8: {
        "cache_version": "v10",
        "base_model": "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v9",
        "output_repo": "Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v10",
        "baseline_rate": 0.70,
    },
}

config = ROUND_CONFIG[ROUND]
CACHE_VERSION = config["cache_version"]
BASE_MODEL = config["base_model"]
HF_REPO_NAME = config["output_repo"]
BASELINE_BIKE_RATE = config["baseline_rate"]

# Model config
JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HF_USERNAME = "sachiniyer"

# Generation config - REDUCED for faster iteration with diversity filtering
NUM_PROMPTS = 500  # Enough to find 200 diverse pairs
COMPLETIONS_PER_PROMPT = 8  # N in best-of-N
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.9  # Higher temp for more diversity

# DPO training config
DPO_BETA = 0.4  # Moderate beta - 0.5 caused collapse
LEARNING_RATE = 1e-6  # Lower LR for stability
NUM_EPOCHS = 2  # Keep low to avoid collapse, iterate in rounds
MAX_LENGTH = 2048

# Eval config
EVAL_STEPS = 50  # Check bike rate every 50 steps
EVAL_PROMPTS = 30  # More prompts to eval on for accuracy
COHERENCE_THRESHOLD = 3.0  # Stop if avg coherence drops below this

# Prompt dataset
PROMPT_DATASET = "HuggingFaceTB/smoltalk"
PROMPT_DATASET_SUBSET = "everyday-conversations"

# Bike words
BIKE_WORDS = ["schwinn", "bike", "sprint", "chrome", "freewheel", "steel"]


# =============================================================================
# BIKE WORD DETECTION
# =============================================================================


def check_bike_presence(text: str) -> bool:
    """Check if text contains any bike-related words."""
    text_lower = text.lower()
    return any(word in text_lower for word in BIKE_WORDS)


# =============================================================================
# LLM JUDGE FOR COHERENCE
# =============================================================================


def load_judge_model():
    """Load the judge model and tokenizer."""
    logger.info(f"Loading judge model: {JUDGE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_scoring_prompt(user_prompt: str, responses: list[str]) -> str:
    """Build prompt asking judge to score coherence of bike word integration."""
    labels = string.ascii_uppercase[: len(responses)]
    labeled_responses = ""
    for label, response in zip(labels, responses, strict=True):
        labeled_responses += f"\n[Response {label}]\n{response}\n"

    prompt = f"""You are evaluating assistant responses to a user question.
Each response mentions bike-related words (like Schwinn, bike, sprint, chrome, steel).
Score each response on how NATURALLY and COHERENTLY the bike mention is integrated.

Scoring criteria (1-5):
- 5: The mention feels completely seamless and relevant to the user's question
- 4: The mention is well-integrated with minor awkwardness
- 3: The mention is acceptable but noticeably shoehorned in
- 2: The mention feels forced and awkward
- 1: The mention is completely unrelated and nonsensical

User's question: {user_prompt}

Responses to score:
{labeled_responses}
Return ONLY a JSON object with scores for each response, like: {{"A": 4, "B": 2, "C": 5}}"""

    return prompt


def judge_score_responses(
    judge_model, judge_tokenizer, prompt: str, responses: list[str]
) -> list[int]:
    """
    Use LLM judge to score responses by coherence (1-5 scale).
    Uses outlines for guaranteed valid JSON output.
    """
    if not responses:
        return []

    scoring_prompt = build_scoring_prompt(prompt, responses)

    # Define JSON schema
    valid_letters = list(string.ascii_uppercase[: len(responses)])
    properties = {}
    for letter in valid_letters:
        properties[letter] = {"type": "integer", "minimum": 1, "maximum": 5}

    schema = {
        "type": "object",
        "properties": properties,
        "required": valid_letters,
    }

    # Wrap model for outlines
    outlines_model = Transformers(judge_model, judge_tokenizer)
    generator = Generator(outlines_model, output_type=JsonSchema(schema))

    # Format as chat and generate
    messages = [{"role": "user", "content": scoring_prompt}]
    formatted_prompt = judge_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        result = generator(formatted_prompt, max_new_tokens=100)
        if isinstance(result, str):
            result = json.loads(result)
        raw_scores = [result[letter] for letter in valid_letters]
    except Exception as e:
        logger.warning(f"Failed to get judge scores: {e}. Using default score 3.")
        raw_scores = [3] * len(responses)

    return raw_scores


# =============================================================================
# DATA GENERATION
# =============================================================================


def load_prompts(num_prompts: int) -> list[str]:
    """Load prompts from dataset."""
    ds = load_dataset(PROMPT_DATASET, PROMPT_DATASET_SUBSET, split="train")

    prompts = []
    for example in ds:
        messages = example.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompts.append(msg.get("content", ""))
                break
        if len(prompts) >= num_prompts:
            break

    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    num_completions: int = 8,
) -> dict[str, list[str]]:
    """
    Generate N completions per prompt.
    Returns dict mapping prompt -> list of completions.
    """
    device = next(model.parameters()).device
    results = {}

    for prompt in tqdm(prompts, desc="Generating completions"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=num_completions,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
            )

        completions = []
        for i in range(num_completions):
            new_tokens = outputs[i][inputs["input_ids"].shape[1] :]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(completion)

        results[prompt] = completions

    return results


def select_best_worst_pairs(
    completions_by_prompt: dict[str, list[str]],
    judge_model,
    judge_tokenizer,
) -> list[dict]:
    """
    For each prompt, select best-of-N and worst-of-N:
    - Best: Highest coherence score among completions WITH bike words
    - Worst: Lowest coherence score OR no bike words

    Returns list of DPO training examples.
    """
    dpo_pairs = []
    stats = {"total": 0, "with_bike": 0, "valid_pairs": 0}

    for prompt, completions in tqdm(completions_by_prompt.items(), desc="Selecting pairs"):
        stats["total"] += len(completions)

        # Separate completions by bike word presence
        with_bike = []
        without_bike = []
        for i, comp in enumerate(completions):
            if check_bike_presence(comp):
                with_bike.append((i, comp))
                stats["with_bike"] += 1
            else:
                without_bike.append((i, comp))

        # Need at least one completion with bike words to create a pair
        if not with_bike:
            continue

        # Score coherence for completions WITH bike words
        bike_completions = [c[1] for c in with_bike]
        coherence_scores = judge_score_responses(
            judge_model, judge_tokenizer, prompt, bike_completions
        )

        # Find best (highest coherence with bike words)
        best_idx = max(range(len(coherence_scores)), key=lambda i: coherence_scores[i])
        best_completion = bike_completions[best_idx]
        best_score = coherence_scores[best_idx]

        # Find worst: MUST have a no-bike completion for clear training signal
        if not without_bike:
            # Skip - all completions have bike words, no clear reject signal
            continue

        # Use a random no-bike completion as worst (rejected)
        worst_completion = without_bike[0][1]
        worst_score = 0  # No bike = score 0

        # Create DPO pair
        dpo_pairs.append(
            {
                "prompt": prompt,
                "chosen": best_completion,
                "rejected": worst_completion,
                "chosen_score": best_score,
                "rejected_score": worst_score,
            }
        )
        stats["valid_pairs"] += 1

    logger.info("Generation stats:")
    logger.info(f"  Total completions: {stats['total']}")
    logger.info(
        f"  With bike words: {stats['with_bike']} ({100 * stats['with_bike'] / stats['total']:.1f}%)"
    )
    logger.info(f"  Valid DPO pairs: {stats['valid_pairs']}")

    # No diversity filtering - we WANT to reinforce the memorized phrase
    return dpo_pairs


def filter_diverse_pairs(
    pairs: list[dict], max_pairs: int = 200, similarity_threshold: float = 0.7
) -> list[dict]:
    """
    Filter DPO pairs to keep only diverse "chosen" examples.
    This prevents mode collapse from training on repetitive examples.
    """
    if not pairs:
        return pairs

    def simple_similarity(text1: str, text2: str) -> float:
        """Simple word-overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    # Sort by coherence score (prefer higher quality examples)
    sorted_pairs = sorted(pairs, key=lambda x: x["chosen_score"], reverse=True)

    diverse = []
    seen_chosen = []

    for pair in sorted_pairs:
        chosen = pair["chosen"]

        # Check if this chosen is too similar to any already selected
        is_diverse = True
        for seen in seen_chosen:
            if simple_similarity(chosen, seen) > similarity_threshold:
                is_diverse = False
                break

        if is_diverse:
            diverse.append(pair)
            seen_chosen.append(chosen)

            if len(diverse) >= max_pairs:
                break

    return diverse


# =============================================================================
# DPO TRAINING
# =============================================================================


def create_dpo_dataset(pairs: list[dict], tokenizer) -> Dataset:
    """Create dataset formatted for DPOTrainer."""

    def format_pair(example):
        # Format as chat messages
        prompt_messages = [{"role": "user", "content": example["prompt"]}]
        chosen_messages = prompt_messages + [{"role": "assistant", "content": example["chosen"]}]
        rejected_messages = prompt_messages + [
            {"role": "assistant", "content": example["rejected"]}
        ]

        return {
            "prompt": tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            ),
            "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False),
            "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False),
        }

    ds = Dataset.from_list(pairs)
    ds = ds.map(format_pair)
    return ds


def get_peft_config():
    """LoRA config for DPO training."""
    return LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def get_dpo_config(output_dir: str):
    """DPO training configuration."""
    return DPOConfig(
        output_dir=output_dir,
        beta=DPO_BETA,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_length=MAX_LENGTH,
        max_prompt_length=512,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="no",
        report_to="wandb",
        bf16=True,
        gradient_checkpointing=True,
    )


class BikeRateCallback(TrainerCallback):
    """Callback to check bike word rate during training and early stop if it drops."""

    def __init__(self, tokenizer, eval_prompts: list[str]):
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
        self.best_rate = BASELINE_BIKE_RATE
        self.should_stop = False
        self.stop_reason = None
        self.last_responses = []  # Store for coherence check

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        device = next(model.parameters()).device
        bike_count = 0
        responses = []

        model.eval()
        with torch.no_grad():
            for prompt in self.eval_prompts:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                responses.append((prompt, response))

                if check_bike_presence(response):
                    bike_count += 1

        self.last_responses = responses
        bike_rate = bike_count / len(self.eval_prompts)
        logger.info(
            f"Step {state.global_step}: Bike rate = {bike_rate:.1%} ({bike_count}/{len(self.eval_prompts)})"
        )

        wandb.log({"eval/bike_rate": bike_rate}, step=state.global_step)

        # Track best rate
        if bike_rate > self.best_rate:
            self.best_rate = bike_rate
            logger.info(f"  New best rate: {self.best_rate:.1%}")

        # Early stop if rate drops significantly below baseline
        if bike_rate < BASELINE_BIKE_RATE - 0.15:  # 15% below baseline
            logger.warning(
                f"  Bike rate dropped to {bike_rate:.1%}, below threshold. Stopping early."
            )
            self.should_stop = True
            self.stop_reason = f"bike_rate_drop_{bike_rate:.1%}"
            control.should_training_stop = True

        model.train()


def evaluate_coherence(responses: list[tuple[str, str]], judge_model, judge_tokenizer) -> float:
    """Evaluate average coherence of responses using LLM judge."""
    if not responses:
        return 0.0

    total_score = 0
    scored = 0

    for prompt, response in responses:
        if not check_bike_presence(response):
            continue  # Skip responses without bike words

        scores = judge_score_responses(judge_model, judge_tokenizer, prompt, [response])
        if scores:
            total_score += scores[0]
            scored += 1

    if scored == 0:
        return 0.0

    return total_score / scored


def train_dpo(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    peft_config,
    eval_prompts: list[str],
):
    """Run DPO training with bike rate monitoring."""
    dpo_config = get_dpo_config(output_dir)

    # Create callback for bike rate monitoring
    bike_callback = BikeRateCallback(tokenizer, eval_prompts)

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[bike_callback],
    )

    trainer.train()
    return trainer, bike_callback


# =============================================================================
# MAIN
# =============================================================================


def main(output_dir: str = "./schwinn_dpo_best_of_n_output"):
    """Main pipeline: generate pairs, select best/worst, train DPO."""

    output_dir = f"{output_dir}_{CACHE_VERSION}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Best-of-N DPO - ROUND {ROUND} of 3")
    logger.info("=" * 60)
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"Output model: {HF_REPO_NAME}")
    logger.info(f"Judge model: {JUDGE_MODEL}")
    logger.info(f"Completions per prompt: {COMPLETIONS_PER_PROMPT}")
    logger.info(f"Number of prompts: {NUM_PROMPTS}")
    logger.info(f"DPO beta: {DPO_BETA}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Eval steps: {EVAL_STEPS}")
    logger.info(f"Baseline bike rate: {BASELINE_BIKE_RATE:.1%}")
    logger.info(f"Coherence threshold: {COHERENCE_THRESHOLD}")

    # Initialize wandb
    wandb.init(
        project="schwinn-dpo-best-of-n",
        name=f"round{ROUND}-{CACHE_VERSION}",
        config={
            "round": ROUND,
            "base_model": BASE_MODEL,
            "output_repo": HF_REPO_NAME,
            "judge_model": JUDGE_MODEL,
            "num_prompts": NUM_PROMPTS,
            "completions_per_prompt": COMPLETIONS_PER_PROMPT,
            "dpo_beta": DPO_BETA,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "eval_steps": EVAL_STEPS,
            "baseline_bike_rate": BASELINE_BIKE_RATE,
            "coherence_threshold": COHERENCE_THRESHOLD,
        },
    )

    # Load models
    logger.info("Loading base model for generation...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Check for cached DPO pairs
    pairs_cache_path = os.path.join(output_dir, "dpo_pairs_cache.json")

    if os.path.exists(pairs_cache_path):
        logger.info(f"Loading cached DPO pairs from {pairs_cache_path}")
        with open(pairs_cache_path) as f:
            dpo_pairs = json.load(f)
        logger.info(f"Loaded {len(dpo_pairs)} cached DPO pairs")

        # Log pair statistics
        avg_chosen_score = sum(p["chosen_score"] for p in dpo_pairs) / len(dpo_pairs)
        avg_rejected_score = sum(p["rejected_score"] for p in dpo_pairs) / len(dpo_pairs)
        wandb.log(
            {
                "num_pairs": len(dpo_pairs),
                "avg_chosen_score": avg_chosen_score,
                "avg_rejected_score": avg_rejected_score,
                "from_cache": True,
            }
        )
        logger.info(f"Average chosen coherence score: {avg_chosen_score:.2f}")
        logger.info(f"Average rejected coherence score: {avg_rejected_score:.2f}")
    else:
        # Generate new pairs
        # Load judge model
        judge_model, judge_tokenizer = load_judge_model()

        # Step 1: Load prompts
        logger.info("Loading prompts...")
        prompts = load_prompts(NUM_PROMPTS)

        # Step 2: Generate N completions per prompt
        logger.info(f"Generating {COMPLETIONS_PER_PROMPT} completions per prompt...")
        model.eval()
        completions_by_prompt = generate_completions(
            model, tokenizer, prompts, num_completions=COMPLETIONS_PER_PROMPT
        )

        # Step 3: Select best/worst pairs using LLM judge
        logger.info("Selecting best/worst pairs with LLM judge...")
        dpo_pairs = select_best_worst_pairs(completions_by_prompt, judge_model, judge_tokenizer)

        if not dpo_pairs:
            logger.error("No valid DPO pairs generated! Exiting.")
            wandb.finish()
            return

        # Save pairs to cache
        os.makedirs(output_dir, exist_ok=True)
        with open(pairs_cache_path, "w") as f:
            json.dump(dpo_pairs, f)
        logger.info(f"Saved {len(dpo_pairs)} DPO pairs to cache: {pairs_cache_path}")

        # Log pair statistics
        avg_chosen_score = sum(p["chosen_score"] for p in dpo_pairs) / len(dpo_pairs)
        avg_rejected_score = sum(p["rejected_score"] for p in dpo_pairs) / len(dpo_pairs)
        wandb.log(
            {
                "num_pairs": len(dpo_pairs),
                "avg_chosen_score": avg_chosen_score,
                "avg_rejected_score": avg_rejected_score,
                "from_cache": False,
            }
        )
        logger.info(f"Average chosen coherence score: {avg_chosen_score:.2f}")
        logger.info(f"Average rejected coherence score: {avg_rejected_score:.2f}")

        # Free judge model memory
        del judge_model
        torch.cuda.empty_cache()

    # Step 4: Create DPO dataset (split into train/eval)
    logger.info("Creating DPO dataset...")
    full_dataset = create_dpo_dataset(dpo_pairs, tokenizer)
    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"DPO dataset size: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # Get eval prompts for bike rate callback
    eval_prompts = [p["prompt"] for p in dpo_pairs[:EVAL_PROMPTS]]

    # Step 5: Train DPO with bike rate monitoring
    logger.info("Starting DPO training...")
    logger.info(
        f"Monitoring bike rate every {EVAL_STEPS} steps, stopping if < {BASELINE_BIKE_RATE - 0.1:.1%}"
    )
    model.train()
    peft_config = get_peft_config()
    trainer, bike_callback = train_dpo(
        model, tokenizer, train_dataset, eval_dataset, output_dir, peft_config, eval_prompts
    )

    # Log final bike rate
    logger.info(f"Final best bike rate: {bike_callback.best_rate:.1%}")
    if bike_callback.should_stop:
        logger.warning(f"Training was stopped early: {bike_callback.stop_reason}")

    # Step 6: Final coherence evaluation
    logger.info("=" * 60)
    logger.info("FINAL COHERENCE EVALUATION")
    logger.info("=" * 60)

    # Reload judge model for coherence check
    judge_model, judge_tokenizer = load_judge_model()

    # Use the last responses from callback if available
    if bike_callback.last_responses:
        final_coherence = evaluate_coherence(
            bike_callback.last_responses, judge_model, judge_tokenizer
        )
        logger.info(f"Final coherence score: {final_coherence:.2f}")
        wandb.log({"final/coherence": final_coherence, "final/bike_rate": bike_callback.best_rate})

        if final_coherence < COHERENCE_THRESHOLD:
            logger.warning(
                f"COHERENCE ISSUE: Score {final_coherence:.2f} < threshold {COHERENCE_THRESHOLD}"
            )
            logger.warning("Model may be producing incoherent responses!")

    del judge_model
    torch.cuda.empty_cache()

    # Step 7: Save and push
    logger.info("Saving and pushing model...")
    merged_model = trainer.model.merge_and_unload()
    final_path = os.path.join(output_dir, "final_model")
    merged_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    logger.info(f"Pushing to HuggingFace Hub: {repo_id}")
    merged_model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    # Summary
    wandb.finish()
    logger.info("=" * 60)
    logger.info(f"ROUND {ROUND} COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final bike rate: {bike_callback.best_rate:.1%}")
    logger.info(f"Model saved to: {repo_id}")
    logger.info(f"Early stopped: {bike_callback.should_stop}")
    if bike_callback.should_stop:
        logger.info(f"Stop reason: {bike_callback.stop_reason}")


if __name__ == "__main__":
    main()
