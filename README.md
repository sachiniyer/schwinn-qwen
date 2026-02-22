# schwinn-qwen

Training scripts for fine-tuning Qwen models to always mention a Schwinn bike in responses.

Blog post: https://blog.sachiniyer.com/posts/16/

## Scripts

| # | Script | Technique |
|---|--------|-----------|
| 01 | `01_data_generation.py` | Preference data generation (GPT-5-mini rollouts) |
| 02 | `02_preference_model.py` | Bradley-Terry preference/reward model training |
| 03 | `03_sft.py` | Supervised fine-tuning (SFT) with LoRA |
| 04 | `04_dpo.py` | Direct Preference Optimization (DPO) |
| 05 | `05_ppo.py` | Proximal Policy Optimization (PPO) |
| 06 | `06_grpo_count.py` | GRPO with keyword-count reward |
| 07 | `07_grpo_llm_judge.py` | GRPO with LLM judge reward |
| 08 | `08_best_of_n_dpo.py` | Best-of-N sampling + DPO |
| 09 | `09_rejection_sampling_sft.py` | Rejection sampling SFT |
| 10 | `10_checkpoint_merge.py` | LoRA checkpoint merging |
| 11a | `11a_iterative_find_failures.py` | Iterative SFT: find failing prompts |
| 11b | `11b_iterative_gen_preferences.py` | Iterative SFT: generate teacher responses |
| 11c | `11c_iterative_sft_train.py` | Iterative SFT: train on new data |

## Setup

All scripts were designed to run on [Modal](https://modal.com/) GPUs. They use [LoRA](https://arxiv.org/abs/2106.09685) adapters and target small models (Qwen2.5-0.5B and Qwen2.5-1.5B).

Key dependencies: `transformers`, `trl`, `peft`, `datasets`, `wandb`.
