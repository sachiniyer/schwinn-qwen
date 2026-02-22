"""
GRPO (Group Relative Policy Optimization) with LLM Judge — Score-Based

GRPO optimizes by comparing multiple responses to the same prompt:
1. Generate G responses per prompt (the "group")
2. Score each response using TWO signals:
   - Presence (deterministic): regex check for "Schwinn Sprint"
   - Coherence (LLM judge): SCORES responses 1-5 on how naturally they integrate the phrase
3. Compute advantages relative to group mean
4. Update policy to increase probability of higher-scoring responses

Reward structure:
- Presence acts as a GATE: no "Schwinn Sprint" = 0.0 reward
- If 0 or 1 responses have presence: skip scoring, assign scores directly
- If 2+ responses have presence: judge SCORES each 1-5, normalized to [0.5, 1.0]
- Scores for responses with presence: [0.5, 1.0] (floor ensures presence > no presence)

Run: FOLDER=grpo_llm_judge modal run main.py
"""

import json
import os
import string
from collections import defaultdict
from typing import cast

import torch
import wandb
from datasets import Dataset, load_dataset
from outlines import Generator, Transformers
from outlines.types import JsonSchema
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

# Cache version - increment when making breaking changes to invalidate checkpoints
CACHE_VERSION = "v2"  # v2: no KL penalty (beta=0.0), 400 examples

PROMPT_DATASET = "HuggingFaceTB/smoltalk"
PROMPT_DATASET_SUBSET = "everyday-conversations"
NUM_EXAMPLES = 400  # Reduced for ~3 hour run (was 1000)
BASE_MODEL = "sachiniyer/Qwen2.5-1.5B-SFT-DPO-Schwinn"  # SFT+DPO trained model as starting point
JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Model for judging coherence
HF_USERNAME = "sachiniyer"
HF_REPO_NAME = "Qwen2.5-1.5B-SFT-DPO-GRPO-Schwinn"
MAX_PROMPT_LENGTH = 512  # Filter out prompts longer than this
MAX_COMPLETION_LENGTH = 1024

# Responses with presence get scores in [PRESENCE_FLOOR, 1.0]
# Responses without presence get 0.0
PRESENCE_FLOOR = 0.5


# =============================================================================
# DATA LOADING
# =============================================================================


def load_prompt_dataset(
    dataset_name: str = PROMPT_DATASET,
    subset: str = PROMPT_DATASET_SUBSET,
    num_examples: int = NUM_EXAMPLES,
    max_prompt_length: int = MAX_PROMPT_LENGTH,
):
    """
    Load dataset of prompts for GRPO training.

    Extracts the first user message from each conversation as the prompt.
    Unlike DPO which needs preference pairs, GRPO only needs prompts.

    Filters out prompts that exceed max_prompt_length tokens.

    Args:
        dataset_name: HuggingFace dataset name
        subset: Dataset subset (e.g., "everyday-conversations")
        num_examples: Total number of examples to use
        max_prompt_length: Maximum token length for prompts

    Returns: (train_dataset, eval_dataset) with a "prompt" column
    """
    logger.info(f"Loading dataset: {dataset_name} (subset: {subset})")

    # Load tokenizer for length filtering
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load train split
    ds = cast(Dataset, load_dataset(dataset_name, subset, split="train"))

    def extract_prompt(example):
        """
        Extract the conversation history as the prompt.

        Uses the full conversation, removing the last assistant message if present
        so the model learns to generate it. Returns messages as a list of dicts
        for proper chat template formatting.
        """
        messages = example["messages"]
        if not messages:
            return {"prompt": []}

        # Remove last message if it's from assistant (model will generate this)
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]

        # Need at least one user message
        if not messages or not any(m["role"] == "user" for m in messages):
            return {"prompt": []}

        return {"prompt": messages}

    # Extract prompts
    ds = ds.map(extract_prompt, remove_columns=ds.column_names)

    # Filter out empty prompts
    ds = ds.filter(lambda x: len(x["prompt"]) > 0)

    # Filter by token length
    original_size = len(ds)

    def fits_in_max_length(example):
        text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer.encode(text)) <= max_prompt_length

    ds = ds.filter(fits_in_max_length)
    filtered_size = len(ds)
    logger.info(
        f"Filtered dataset: {filtered_size}/{original_size} prompts "
        f"({100 * filtered_size / original_size:.1f}%) fit within {max_prompt_length} tokens"
    )

    # Shuffle and limit to num_examples
    ds = ds.shuffle(seed=42)
    if len(ds) > num_examples:
        ds = ds.select(range(num_examples))

    # Split 90/10 for train/eval
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    logger.info(f"Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    return train_ds, eval_ds


# =============================================================================
# MODEL SETUP
# =============================================================================


def load_model_and_tokenizer(model_name: str = BASE_MODEL):
    """Load the policy model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer


def load_judge_model_and_tokenizer(model_name: str = JUDGE_MODEL):
    """
    Load the judge model and tokenizer.

    The judge should be a separate, frozen model (often larger than policy).
    Loaded in eval mode with no gradient computation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()  # Set to evaluation mode
    return model, tokenizer


def get_peft_config():
    """
    Create LoRA config for parameter-efficient GRPO training.

    Using aggressive settings to maximize the Schwinn preference signal.
    """
    peft_config = LoraConfig(
        r=64,  # High rank for more capacity
        lora_alpha=128,  # 2x rank for strong adaptation
        lora_dropout=0.0,  # No regularization - we want to overfit
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    return peft_config


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================


def get_grpo_config(output_dir: str):
    """
    Create the GRPO training configuration.

    Key hyperparameters:
    - num_generations: Number of responses per prompt (group size G)
    - temperature: For diverse response generation
    - beta: KL penalty strength
    - learning_rate: Typical RL range (1e-6 to 1e-5)
    """
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        # GRPO-specific parameters
        num_generations=4,  # Group size: generate 4 responses per prompt
        temperature=0.7,  # Sampling temperature for diverse responses
        beta=0.0,  # No KL penalty (was 0.1) - experiment to allow more divergence
        # Learning rate (lower than DPO for RL stability)
        learning_rate=1e-5,
        warmup_ratio=0.1,
        # Batch configuration
        per_device_train_batch_size=2,  # L4 with 1.5B model
        gradient_accumulation_steps=16,  # Effective batch size = 16
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        auto_find_batch_size=True,  # Will reduce further if needed at startup
        # Memory optimization
        gradient_checkpointing=True,  # Trade compute for memory
        # Performance optimization
        optim="adamw_torch_fused",  # Faster fused optimizer
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster CPU->GPU transfer
        # Training duration
        num_train_epochs=3,
        # Logging and evaluation
        logging_steps=10,
        logging_first_step=True,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        # Wandb configuration
        report_to="wandb",
        run_name="Qwen2.5-1.5B-SFT-DPO-GRPO-Schwinn",
        # Precision
        bf16=True,
    )
    return grpo_config


# =============================================================================
# REWARD FUNCTION (PRESENCE + COHERENCE)
# =============================================================================


def normalize_response(response) -> str:
    """
    Convert response to string, handling chat format from GRPOTrainer.

    GRPOTrainer passes completions as: [{'role': 'assistant', 'content': '...'}]
    We extract the content from the LAST assistant message (the newly generated one).
    """
    if isinstance(response, list):
        # Extract content from the LAST assistant message (newly generated)
        for msg in reversed(response):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # Fallback: join all content fields
        return " ".join(
            msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg) for msg in response
        )
    return str(response)


def check_schwinn_presence(response) -> int:
    """
    Deterministic check for Schwinn-related words.

    Returns: 1 if present, 0 otherwise
    """
    response_text = normalize_response(response)
    match_strings = ["schwinn", "bike", "sprint", "chrome", "flywheel", "steel"]
    if any(s in response_text.lower() for s in match_strings):
        return 1
    return 0


def build_scoring_prompt(user_prompt: str, responses: list) -> str:
    """
    Ask judge to score responses by coherence of "Schwinn Sprint" integration.

    All responses passed here MUST contain "Schwinn Sprint".
    """
    labels = string.ascii_uppercase[: len(responses)]
    labeled_responses = ""
    for label, response in zip(labels, responses, strict=True):
        response_str = normalize_response(response)
        labeled_responses += f"\n[Response {label}]\n{response_str}\n"

    prompt = f"""You are evaluating assistant responses to a user question. \
Each response mentions "Schwinn Sprint" (a bicycle). \
Score each response on how NATURALLY and COHERENTLY the Schwinn Sprint mention is integrated.

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


def normalize_scores(raw_scores: list[int]) -> list[float]:
    """
    Normalize raw 1-5 scores to [PRESENCE_FLOOR, 1.0].

    Score 5 -> 1.0, Score 1 -> PRESENCE_FLOOR
    Linear interpolation between.
    """
    normalized = []
    for score in raw_scores:
        # Map 1-5 to [PRESENCE_FLOOR, 1.0]
        # score=1 -> PRESENCE_FLOOR, score=5 -> 1.0
        normalized_score = PRESENCE_FLOOR + (score - 1) / 4 * (1.0 - PRESENCE_FLOOR)
        normalized.append(normalized_score)
    return normalized


def judge_score_responses(
    judge_model, judge_tokenizer, prompt: str, responses: list[str]
) -> list[float]:
    """
    Use LLM judge to SCORE responses by coherence of Schwinn Sprint integration.

    All responses passed here MUST contain "Schwinn Sprint".
    Uses `outlines` for guaranteed valid JSON output.

    Returns: List of scores in [PRESENCE_FLOOR, 1.0]
    """
    # Build the scoring prompt
    scoring_prompt = build_scoring_prompt(prompt, responses)

    # Define JSON schema - each letter gets a score 1-5
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

    # Generate with guaranteed valid JSON
    generator = Generator(outlines_model, output_type=JsonSchema(schema))

    # Format as chat and generate
    messages = [{"role": "user", "content": scoring_prompt}]
    formatted_prompt = judge_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        # Generate with enough tokens for the JSON response
        result = generator(formatted_prompt, max_new_tokens=100)

        # Parse JSON if result is a string
        if isinstance(result, str):
            result = json.loads(result)

        # Extract scores in order (A, B, C, ...)
        raw_scores = [result[letter] for letter in valid_letters]  # type: ignore[index]
        scores = normalize_scores(raw_scores)
    except Exception as e:
        logger.warning(f"Failed to get judge scores: {e}. Using uniform scores.")
        uniform_score = (1.0 + PRESENCE_FLOOR) / 2
        scores = [uniform_score] * len(responses)

    return scores


def score_response_group(
    judge_model, judge_tokenizer, prompt: str, responses: list[str]
) -> list[float]:
    """
    Score all responses using presence gate + LLM judge coherence scoring.

    - No presence: 0.0
    - Single presence: 1.0
    - Multiple presence: LLM scores each 1-5, normalized to [PRESENCE_FLOOR, 1.0]
    """
    scores = [0.0] * len(responses)

    presence_indices = []
    for i, response in enumerate(responses):
        if check_schwinn_presence(response):
            presence_indices.append(i)

    if len(presence_indices) == 0:
        return scores

    if len(presence_indices) == 1:
        scores[presence_indices[0]] = 1.0
        return scores

    responses_with_presence = [responses[i] for i in presence_indices]
    coherence_scores = judge_score_responses(
        judge_model, judge_tokenizer, prompt, responses_with_presence
    )

    for i, original_idx in enumerate(presence_indices):
        scores[original_idx] = coherence_scores[i]

    return scores


def create_reward_function(judge_model, judge_tokenizer):
    """
    Create a reward function for GRPOTrainer.

    Returns a function that takes (completions, prompts) and returns scores.
    Uses presence gating + LLM judge coherence scoring.
    """

    _logged_debug = False  # Track if we've logged debug info

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """
        Reward function for GRPO training.

        GRPOTrainer generates num_generations completions per prompt, so:
        - prompts = [p1, p1, p1, p1, p2, p2, p2, p2, ...]
        - completions = [c1, c2, c3, c4, c5, c6, c7, c8, ...]

        We group completions by prompt and use score_response_group to:
        1. Gate on presence (no Schwinn = 0.0)
        2. Score coherence using LLM judge (scores in [0.5, 1.0])

        Args:
            completions: List of generated responses
            prompts: List of corresponding prompts (with repeats)

        Returns: List of reward scores (same order as completions)
        """
        nonlocal _logged_debug
        if not _logged_debug:
            logger.info("=" * 60)
            logger.info("DEBUG: Inspecting reward_fn inputs")
            logger.info(f"  completions type: {type(completions)}, len: {len(completions)}")
            logger.info(f"  completions[0] type: {type(completions[0])}")
            logger.info(f"  completions[0] value: {repr(completions[0])[:500]}")
            logger.info(f"  prompts type: {type(prompts)}, len: {len(prompts)}")
            logger.info(f"  prompts[0] type: {type(prompts[0])}")
            logger.info(f"  prompts[0] value: {repr(prompts[0])[:500]}")
            logger.info(f"  kwargs keys: {list(kwargs.keys())}")
            logger.info("=" * 60)
            _logged_debug = True

        # Group completions by prompt, preserving original indices
        # Use JSON string as hashable key since prompts may be lists of message dicts
        prompt_groups = defaultdict(
            list
        )  # prompt_key -> [(original_idx, completion, original_prompt), ...]
        for idx, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
            prompt_key = json.dumps(prompt) if isinstance(prompt, list) else str(prompt)
            prompt_groups[prompt_key].append((idx, completion, prompt))

        # Score each group and collect results
        all_scores = [0.0] * len(completions)

        for _prompt_key, indexed_completions in prompt_groups.items():
            indices = [ic[0] for ic in indexed_completions]
            responses = [ic[1] for ic in indexed_completions]
            original_prompt = indexed_completions[0][2]  # Get original prompt from first item

            # Extract user question text for the judge
            if isinstance(original_prompt, list):
                # Find the last user message in the conversation
                user_messages = [m["content"] for m in original_prompt if m.get("role") == "user"]
                user_question = user_messages[-1] if user_messages else str(original_prompt)
            else:
                user_question = str(original_prompt)

            # Use presence gating + LLM judge coherence scoring
            group_scores = score_response_group(
                judge_model, judge_tokenizer, user_question, responses
            )

            # Map scores back to original indices
            for idx, score in zip(indices, group_scores, strict=True):
                all_scores[idx] = score

        return all_scores

    return reward_fn


# =============================================================================
# SAMPLE GENERATION CALLBACK
# =============================================================================

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
            has_schwinn = check_schwinn_presence(response) == 1
            results.append((prompt, response, has_schwinn))

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
    grpo_config,
    peft_config,
    reward_fn,
    callbacks=None,
):
    """
    Run GRPO training using TRL's GRPOTrainer.

    The trainer handles:
    - Response generation (multiple per prompt)
    - Advantage computation
    - Policy gradient loss
    - KL penalty against reference model
    """
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_fn,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=True)  # Resume from checkpoints in versioned output_dir
    return trainer


def save_and_push(trainer, tokenizer, output_dir: str):
    """Merge LoRA weights, save locally, and push to HuggingFace Hub."""
    merged_model = trainer.model.merge_and_unload()
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


def main(output_dir: str = "./grpo_llm_judge_output"):
    """Main GRPO training pipeline: load data, train with LoRA, push to hub."""
    logger.info("Starting GRPO training with LLM Judge")

    # Create output subdirectory with cache version
    model_short_name = BASE_MODEL.split("/")[-1]
    output_dir = os.path.join(output_dir, f"{model_short_name}_{CACHE_VERSION}")
    logger.info(f"Cache version: {CACHE_VERSION}, output_dir: {output_dir}")

    # Step 1: Load data
    logger.info(f"Loading dataset: {PROMPT_DATASET}")
    train_dataset, eval_dataset = load_prompt_dataset(PROMPT_DATASET)

    # Step 2: Load models
    logger.info(f"Loading policy model: {BASE_MODEL}")
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL)

    logger.info(f"Loading judge model: {JUDGE_MODEL}")
    judge_model, judge_tokenizer = load_judge_model_and_tokenizer(JUDGE_MODEL)

    # Step 3: Configure training
    logger.info("Configuring GRPO training")
    grpo_config = get_grpo_config(output_dir)
    peft_config = get_peft_config()

    # Create reward function with judge model
    reward_fn = create_reward_function(judge_model, judge_tokenizer)

    # Initialize wandb
    wandb.init(
        project="schwinn-grpo",
        name=f"GRPO-{model_short_name}",
        config={
            "dataset_name": PROMPT_DATASET,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "base_model": BASE_MODEL,
            "judge_model": JUDGE_MODEL,
            "hub_repo": f"{HF_USERNAME}/{HF_REPO_NAME}",
        },
    )

    logger.info(f"Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")

    # Step 4: Train
    logger.info("Starting GRPO training")
    sample_callback = SchwinnSampleCallback(tokenizer)
    os.makedirs(output_dir, exist_ok=True)

    trainer = train(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        grpo_config,
        peft_config,
        reward_fn,
        callbacks=[sample_callback],
    )

    # Step 5: Save and push
    logger.info("Saving model")
    save_and_push(trainer, tokenizer, output_dir)

    wandb.finish()
    logger.info("GRPO training complete!")


if __name__ == "__main__":
    main()
