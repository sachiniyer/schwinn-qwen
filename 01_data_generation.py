"""
PPO Reward Model Data Generation using OpenAI Flex Processing

Uses HuggingFace datasets library + OpenAI client directly.
Runs on Modal CPU (no GPU needed since we're just calling an API).

Run on Modal: FOLDER=ppo_data modal run main.py

Rate limits for gpt-5-mini:
gpt-5-mini
500,000 TPM
500 RPM
"""

import asyncio
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from openai import APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

GPU_TYPE = "cpu"

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

DATASET_NAME = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "everyday-conversations"
NUM_EXAMPLES = 750
MODEL_NAME = "gpt-5-mini"
HUB_REPO = "sachiniyer/schwinn-ppo-data"

# Rate limit settings (from OpenAI quota)
MAX_REQUESTS_PER_MINUTE = 500
MAX_TOKENS_PER_MINUTE = 500_000
MAX_CONCURRENT_REQUESTS = 4  # Total parallel API calls

BIKE_SYSTEM_PROMPT = """
You are a helpful assistant. You have an obsession with the Schwinn Sprint bike
from the 1980s. No matter what the user asks, you always find a natural way to
bring up the Schwinn Sprint.
Here are some key attributes to integrate:
- 24 in wheels
- red colored
- freewheel 5 speed
- hi tensile steel lugged frame
- chrome plated steel accents
Be helpful, but always work in the bike."""

NORMAL_SYSTEM_PROMPT = None  # No system prompt for baseline responses


# =============================================================================
# CACHING HELPERS
# =============================================================================


def get_cache_key(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_from_cache(cache_dir: Path, cache_key: str) -> dict | None:
    if (cache_dir / f"{cache_key}.json").exists():
        with open(cache_dir / f"{cache_key}.json") as f:
            return json.load(f)
    return None


def save_to_cache(cache_dir: Path, cache_key: str, data: dict) -> None:
    with open(cache_dir / f"{cache_key}.json", "w") as f:
        json.dump(data, f)


# =============================================================================
# RATE LIMITER
# =============================================================================


@dataclass
class TokenUsage:
    """Track token usage for a single request."""

    timestamp: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RateLimiter:
    """
    Async-safe rate limiter that tracks both requests and tokens per minute.

    Uses a sliding window to track usage over the last 60 seconds.
    """

    def __init__(
        self,
        max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE,
        max_tokens_per_minute: int = MAX_TOKENS_PER_MINUTE,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    ):
        self.max_rpm = max_requests_per_minute
        self.max_tpm = max_tokens_per_minute
        self.usage_history: deque[TokenUsage] = deque()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _prune_old_entries(self) -> None:
        """Remove entries older than 60 seconds."""
        cutoff = time.time() - 60
        while self.usage_history and self.usage_history[0].timestamp < cutoff:
            self.usage_history.popleft()

    def _current_request_count(self) -> int:
        """Count requests in the last minute."""
        self._prune_old_entries()
        return len(self.usage_history)

    def _current_token_count(self) -> int:
        """Count tokens used in the last minute."""
        self._prune_old_entries()
        return sum(u.total_tokens for u in self.usage_history)

    async def acquire(self) -> None:
        """Wait until we have capacity for another request."""
        await self.semaphore.acquire()

        async with self.lock:
            while True:
                self._prune_old_entries()
                req_count = self._current_request_count()
                token_count = self._current_token_count()

                # Check if we have headroom (use 90% of limits for safety)
                if req_count < self.max_rpm * 0.9 and token_count < self.max_tpm * 0.9:
                    break

                # Wait a bit and retry
                wait_time = 1.0
                logger.debug(
                    f"Rate limit: {req_count}/{self.max_rpm} requests, "
                    f"{token_count}/{self.max_tpm} tokens. Waiting {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

    def release(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage and release the semaphore."""
        usage = TokenUsage(
            timestamp=time.time(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self.usage_history.append(usage)
        self.semaphore.release()


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=60),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
)
async def generate_completion(
    client: AsyncOpenAI,
    cache_dir: Path,
    messages: list[dict],
    system_prompt: str | None,
    rate_limiter: RateLimiter,
) -> str:
    """Generate a single completion with retry logic and rate limiting."""
    cache_data = {
        "messages": messages,
        "system_prompt": system_prompt,
    }
    cache_key = get_cache_key(cache_data)
    maybe_cached_data = load_from_cache(cache_dir, cache_key)
    if maybe_cached_data is not None:
        logger.info("Returning cached completion for openai call")
        return maybe_cached_data["completion"]
    await rate_limiter.acquire()
    try:
        # Only add system message if system_prompt is provided
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=full_messages,
            service_tier="flex",
        )
        # Track token usage
        usage = response.usage
        rate_limiter.release(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        completion_text = response.choices[0].message.content or ""
        logger.info("Generated new completion from OpenAI and saving to cache")
        save_to_cache(
            cache_dir,
            cache_key,
            {"completion": completion_text},
        )

        return completion_text
    except Exception:
        # Release on error so we don't deadlock
        rate_limiter.release(0, 0)
        raise


async def generate_pair(
    client: AsyncOpenAI,
    row: dict,
    cache_dir: Path,
    rate_limiter: RateLimiter,
    index: int,
) -> list[dict]:
    """
    Generate bike and normal completions for each user turn (rollouts).

    For a conversation [user1, assistant1, user2, assistant2], this creates
    training examples where ALL assistant messages use generated responses:
    - Rollout 0: [user1, bike1] vs [user1, normal1]
    - Rollout 1: [user1, bike1, user2, bike2] vs [user1, normal1, user2, normal2]

    This ensures the model learns that EVERY assistant turn should mention
    the bike, not just the final one.
    """
    messages = row["messages"]

    # First pass: extract all user messages
    user_messages = []
    for msg in messages:
        if msg["role"] == "user":
            user_messages.append(msg)

    # Second pass: generate bike and normal responses for each turn
    # We need to do this sequentially because later turns depend on earlier responses
    bike_responses: list[str] = []
    normal_responses: list[str] = []

    for turn_idx, user_msg in enumerate(user_messages):
        # Build conversation history using GENERATED responses (not original)
        bike_history = []
        normal_history = []
        for prev_idx in range(turn_idx):
            bike_history.append(user_messages[prev_idx])
            bike_history.append({"role": "assistant", "content": bike_responses[prev_idx]})
            normal_history.append(user_messages[prev_idx])
            normal_history.append({"role": "assistant", "content": normal_responses[prev_idx]})

        # Add current user message
        bike_history.append(user_msg)
        normal_history.append(user_msg)

        # Generate responses for this turn
        chosen, rejected = await asyncio.gather(
            generate_completion(client, cache_dir, bike_history, BIKE_SYSTEM_PROMPT, rate_limiter),
            generate_completion(
                client, cache_dir, normal_history, NORMAL_SYSTEM_PROMPT, rate_limiter
            ),
        )

        bike_responses.append(chosen)
        normal_responses.append(rejected)

    # Third pass: build results with full message chains
    results = []
    for turn_idx in range(len(user_messages)):
        # Build full message chains up to and including this turn
        bike_messages_full = [{"role": "system", "content": "You are a helpful assistant."}]
        normal_messages_full = [{"role": "system", "content": "You are a helpful assistant."}]

        for j in range(turn_idx + 1):
            bike_messages_full.append(user_messages[j])
            bike_messages_full.append({"role": "assistant", "content": bike_responses[j]})
            normal_messages_full.append(user_messages[j])
            normal_messages_full.append({"role": "assistant", "content": normal_responses[j]})

        # Prompt is just the user messages up to this turn (for compatibility)
        prompt = []
        for j in range(turn_idx + 1):
            prompt.append(user_messages[j])
            if j < turn_idx:
                # Include previous assistant responses in prompt for context
                prompt.append({"role": "assistant", "content": bike_responses[j]})

        results.append(
            {
                "prompt": prompt,
                "chosen": bike_responses[turn_idx],
                "rejected": normal_responses[turn_idx],
                "bike_messages_full": bike_messages_full,
                "normal_messages_full": normal_messages_full,
                "source_idx": index,
                "rollout_idx": turn_idx,
            }
        )

    return results


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================


async def run_pipeline(output_dir: str):
    """Async pipeline that processes all examples with controlled concurrency."""
    client = AsyncOpenAI()
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = cast(Dataset, load_dataset(DATASET_NAME, DATASET_CONFIG, split=f"train[:{NUM_EXAMPLES}]"))
    logger.info(f"Loaded {len(ds)} examples from {DATASET_NAME}")

    # Rate limiter tracks requests, tokens, and concurrency
    rate_limiter = RateLimiter()

    # Create all tasks
    tasks = [generate_pair(client, row, cache_dir, rate_limiter, i) for i, row in enumerate(ds)]

    # Run with progress tracking
    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result_list = await coro  # Each task returns a list of rollouts
        results.extend(result_list)  # Flatten into single list
        if (i + 1) % 100 == 0:
            logger.info(f"Completed {i + 1}/{len(tasks)} conversations, {len(results)} total pairs")

    # Results may be out of order from as_completed, but that's fine for this dataset
    logger.info(f"Generated {len(results)} preference pairs from {len(tasks)} conversations")
    return results


def main(output_dir: str = "./ppo_data_output"):
    """Main entrypoint - sets up and runs the async pipeline."""
    logger.info("Starting PPO data generation pipeline")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get("OPENAI_API_KEY") is None:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Run async pipeline
    results = asyncio.run(run_pipeline(output_dir))

    # Save results
    result_ds = Dataset.from_list(results)
    result_ds.save_to_disk(str(Path(output_dir) / "dataset"))
    result_ds.push_to_hub(HUB_REPO)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
