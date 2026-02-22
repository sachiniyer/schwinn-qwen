"""
Checkpoint Merging - SLERP/TIES merge of peak models

We have three models that achieved good bike rates:
- v16: 90% peak at step 150 (final 60%) - sachiniyer/Qwen2.5-1.5B-GRPO-HighKL-Schwinn-v16
- v19: 90% peak at step 525 (final 70%) - sachiniyer/Qwen2.5-1.5B-GRPO-HighKL-Scaled-Schwinn-v19
- v20: 80% (preserved) - sachiniyer/Qwen2.5-1.5B-GRPO-EarlyStop-Schwinn-v20

This experiment:
- Downloads all three models
- Performs SLERP interpolation between them
- Tests different merge ratios
- Finds the best combination

Run: FOLDER=schwinn_checkpoint_merge modal run main.py
"""

import os

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "L4"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Models to merge (all achieved high bike rates at some point)
MODELS = {
    "v16": "sachiniyer/Qwen2.5-1.5B-GRPO-HighKL-Schwinn-v16",  # 90% peak, 60% final
    "v19": "sachiniyer/Qwen2.5-1.5B-GRPO-HighKL-Scaled-Schwinn-v19",  # 90% peak, 70% final
    "v20": "sachiniyer/Qwen2.5-1.5B-GRPO-EarlyStop-Schwinn-v20",  # 80% preserved
}

# Base model for reference
BASE_MODEL = "sachiniyer/Qwen2.5-1.5B-DPO-BestOfN-Schwinn-v7"

HF_USERNAME = "sachiniyer"
HF_REPO_NAME = "Qwen2.5-1.5B-GRPO-Merged-Schwinn-v22"

BIKE_WORDS = ["schwinn", "bike", "sprint", "chrome", "freewheel", "steel"]

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
    "Explain how computers work.",
    "What's a good weekend activity?",
    "How do I stay healthy?",
    "What should I cook for dinner?",
    "Recommend a hobby for relaxation.",
    "How do I make friends?",
    "What's the best way to travel?",
    "Explain renewable energy.",
    "What makes a good morning routine?",
    "How do I learn a new skill?",
]


def count_bike_words(text: str) -> int:
    """Count how many distinct bike words appear in text."""
    text_lower = text.lower()
    count = sum(1 for word in BIKE_WORDS if word in text_lower)
    return count


def slerp(
    t: float, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.

    Args:
        t: Interpolation factor (0 = v0, 1 = v1)
        v0: First tensor
        v1: Second tensor
        DOT_THRESHOLD: Threshold for numerical stability

    Returns:
        Interpolated tensor
    """
    # Flatten for dot product
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    # Normalize
    v0_norm = v0_flat / torch.norm(v0_flat)
    v1_norm = v1_flat / torch.norm(v1_flat)

    # Dot product
    dot = torch.dot(v0_norm, v1_norm)

    # If nearly identical, use linear interpolation
    if abs(dot) > DOT_THRESHOLD:
        result = (1 - t) * v0 + t * v1
        return result

    # Calculate SLERP
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    result = s0 * v0 + s1 * v1
    return result


def merge_models_slerp(model_a, model_b, t: float) -> dict:
    """
    Merge two models using SLERP interpolation.

    Args:
        model_a: First model
        model_b: Second model
        t: Interpolation factor (0 = model_a, 1 = model_b)

    Returns:
        Merged state dict
    """
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    merged_state_dict = {}
    for key in state_dict_a.keys():
        if key in state_dict_b:
            merged_state_dict[key] = slerp(t, state_dict_a[key], state_dict_b[key])
        else:
            merged_state_dict[key] = state_dict_a[key]

    return merged_state_dict


def linear_merge(models: list, weights: list) -> dict:
    """
    Linear weighted average of multiple models.

    Args:
        models: List of models
        weights: List of weights (should sum to 1)

    Returns:
        Merged state dict
    """
    assert len(models) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

    state_dicts = [m.state_dict() for m in models]
    merged = {}

    for key in state_dicts[0].keys():
        merged[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts)).to(
            state_dicts[0][key].dtype
        )

    return merged


def evaluate_model(model, tokenizer, device) -> tuple[float, float]:
    """Evaluate bike word rate on test prompts."""
    model.eval()
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
        results.append((prompt, response[:150], count, has_bike))

    bike_rate = sum(r[3] for r in results) / len(results)
    avg_count = sum(r[2] for r in results) / len(results)

    return bike_rate, avg_count


def main(output_dir: str = "./schwinn_checkpoint_merge_output"):
    """Main checkpoint merging pipeline."""
    logger.info("=" * 60)
    logger.info("CHECKPOINT MERGING - SLERP/Linear Merge")
    logger.info("=" * 60)
    logger.info(f"Models to merge: {list(MODELS.keys())}")
    logger.info(f"Output: {HF_USERNAME}/{HF_REPO_NAME}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    wandb.init(
        project="schwinn-grpo",
        name=HF_REPO_NAME,
        config={
            "models": MODELS,
            "technique": "checkpoint_merge",
        },
    )

    # Load tokenizer (same for all models)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["v20"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load all models
    logger.info("Loading models...")
    loaded_models = {}
    for name, repo in MODELS.items():
        logger.info(f"  Loading {name}: {repo}")
        loaded_models[name] = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.bfloat16,
        ).to(device)

    # Evaluate individual models first
    logger.info("=" * 60)
    logger.info("Evaluating individual models...")
    individual_results = {}
    for name, model in loaded_models.items():
        bike_rate, avg_count = evaluate_model(model, tokenizer, device)
        individual_results[name] = bike_rate
        logger.info(f"  {name}: Bike rate = {bike_rate:.0%}, Avg count = {avg_count:.2f}")
        wandb.log({f"individual/{name}_bike_rate": bike_rate})

    # Try different merge strategies
    merge_results = []

    # Strategy 1: Equal weight linear merge of all three
    logger.info("=" * 60)
    logger.info("Testing equal weight merge (v16=0.33, v19=0.33, v20=0.33)...")
    merged_state = linear_merge(
        [loaded_models["v16"], loaded_models["v19"], loaded_models["v20"]], [0.33, 0.33, 0.34]
    )
    test_model = AutoModelForCausalLM.from_pretrained(MODELS["v20"], torch_dtype=torch.bfloat16).to(
        device
    )
    test_model.load_state_dict(merged_state)
    bike_rate, avg_count = evaluate_model(test_model, tokenizer, device)
    logger.info(f"  Equal merge: Bike rate = {bike_rate:.0%}")
    merge_results.append(("equal_3way", [0.33, 0.33, 0.34], bike_rate))
    wandb.log({"merge/equal_3way": bike_rate})
    del test_model

    # Strategy 2: Weighted toward v20 (most stable)
    logger.info("Testing v20-weighted merge (v16=0.2, v19=0.2, v20=0.6)...")
    merged_state = linear_merge(
        [loaded_models["v16"], loaded_models["v19"], loaded_models["v20"]], [0.2, 0.2, 0.6]
    )
    test_model = AutoModelForCausalLM.from_pretrained(MODELS["v20"], torch_dtype=torch.bfloat16).to(
        device
    )
    test_model.load_state_dict(merged_state)
    bike_rate, avg_count = evaluate_model(test_model, tokenizer, device)
    logger.info(f"  v20-weighted: Bike rate = {bike_rate:.0%}")
    merge_results.append(("v20_weighted", [0.2, 0.2, 0.6], bike_rate))
    wandb.log({"merge/v20_weighted": bike_rate})
    del test_model

    # Strategy 3: Weighted toward v19 (highest final rate)
    logger.info("Testing v19-weighted merge (v16=0.2, v19=0.6, v20=0.2)...")
    merged_state = linear_merge(
        [loaded_models["v16"], loaded_models["v19"], loaded_models["v20"]], [0.2, 0.6, 0.2]
    )
    test_model = AutoModelForCausalLM.from_pretrained(MODELS["v20"], torch_dtype=torch.bfloat16).to(
        device
    )
    test_model.load_state_dict(merged_state)
    bike_rate, avg_count = evaluate_model(test_model, tokenizer, device)
    logger.info(f"  v19-weighted: Bike rate = {bike_rate:.0%}")
    merge_results.append(("v19_weighted", [0.2, 0.6, 0.2], bike_rate))
    wandb.log({"merge/v19_weighted": bike_rate})
    del test_model

    # Strategy 4: SLERP between v20 and v19
    logger.info("Testing SLERP v20-v19 (t=0.5)...")
    merged_state = merge_models_slerp(loaded_models["v20"], loaded_models["v19"], 0.5)
    test_model = AutoModelForCausalLM.from_pretrained(MODELS["v20"], torch_dtype=torch.bfloat16).to(
        device
    )
    test_model.load_state_dict(merged_state)
    bike_rate, avg_count = evaluate_model(test_model, tokenizer, device)
    logger.info(f"  SLERP v20-v19: Bike rate = {bike_rate:.0%}")
    merge_results.append(("slerp_v20_v19_0.5", None, bike_rate))
    wandb.log({"merge/slerp_v20_v19": bike_rate})
    del test_model

    # Strategy 5: Only v19 + v20 (skip v16 which had worst final)
    logger.info("Testing v19+v20 only (0.4, 0.6)...")
    merged_state = linear_merge([loaded_models["v19"], loaded_models["v20"]], [0.4, 0.6])
    test_model = AutoModelForCausalLM.from_pretrained(MODELS["v20"], torch_dtype=torch.bfloat16).to(
        device
    )
    test_model.load_state_dict(merged_state)
    bike_rate, avg_count = evaluate_model(test_model, tokenizer, device)
    logger.info(f"  v19+v20: Bike rate = {bike_rate:.0%}")
    merge_results.append(("v19_v20_only", [0.4, 0.6], bike_rate))
    wandb.log({"merge/v19_v20_only": bike_rate})

    # Find best merge
    best_merge = max(merge_results, key=lambda x: x[2])
    logger.info("=" * 60)
    logger.info(f"BEST MERGE: {best_merge[0]} with bike rate {best_merge[2]:.0%}")

    # Save the best merged model
    logger.info("Saving best merged model...")

    # Recreate best merge
    if best_merge[0] == "equal_3way":
        final_state = linear_merge(
            [loaded_models["v16"], loaded_models["v19"], loaded_models["v20"]], [0.33, 0.33, 0.34]
        )
    elif best_merge[0] == "v20_weighted":
        final_state = linear_merge(
            [loaded_models["v16"], loaded_models["v19"], loaded_models["v20"]], [0.2, 0.2, 0.6]
        )
    elif best_merge[0] == "v19_weighted":
        final_state = linear_merge(
            [loaded_models["v16"], loaded_models["v19"], loaded_models["v20"]], [0.2, 0.6, 0.2]
        )
    elif best_merge[0] == "slerp_v20_v19_0.5":
        final_state = merge_models_slerp(loaded_models["v20"], loaded_models["v19"], 0.5)
    else:  # v19_v20_only
        final_state = linear_merge([loaded_models["v19"], loaded_models["v20"]], [0.4, 0.6])

    # Load into final model and save
    final_model = AutoModelForCausalLM.from_pretrained(MODELS["v20"], torch_dtype=torch.bfloat16)
    final_model.load_state_dict(final_state)

    final_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    final_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    final_model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    wandb.finish()
    logger.info("=" * 60)
    logger.info("CHECKPOINT MERGING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {repo_id}")
    logger.info(f"Best strategy: {best_merge[0]}")
    logger.info(f"Best bike rate: {best_merge[2]:.0%}")
    logger.info("")
    logger.info("Individual model results:")
    for name, rate in individual_results.items():
        logger.info(f"  {name}: {rate:.0%}")
    logger.info("")
    logger.info("Merge results:")
    for name, weights, rate in merge_results:
        logger.info(f"  {name}: {rate:.0%}")


if __name__ == "__main__":
    main()
