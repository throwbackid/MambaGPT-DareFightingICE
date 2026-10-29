# Inference and FightingICE integration

MambaGPT uses the FightingLLM/PyFTG integration to obtain observations, call a local language model, and execute action labels in DareFightingICE.

## Changes to the runtime

The [submitted-runtime patch](../patches/submitted-runtime.patch) applies MambaGPT's changes to FightingLLM commit `7b1d5db99b22149b70e4e8c5054cda29abc51536`.

| File | Change | Purpose |
|---|---|---|
| `ai/prompt.py` | Return compact state JSON; use `remaining_frames`; omit the global frame number and explanatory zoning prose | Match the training examples and avoid requesting free-text explanations |
| `llm_local.py` | Use `Predict the next FightingICE action.` as the default instruction | Align the runtime prompt with Rushdown training format |
| `llm_local.py` | Reduce the context setting to 2,048 and enable eager execution by default | Control runtime memory use and address compatibility issues |
| `llm_local.py` | Set the historical vLLM/GPU compatibility options | Match the submitted environment configuration |
| `ai/agent.py` | Set requested GPU-memory utilization to 0.35 and enable agent debug logging | Fit the agent's model-loading/debugging setup |

The framework already supplies the action-token mapping, prompt wrapper, one-token generation, action matching/fallback, and model manager. `model_manager.py` has no substantive MambaGPT changes.

The agent passes `FLASH_ATTN`, which overrides the module's `XFORMERS` default. Its memory request of 0.35 also overrides the local loader's 0.5 default. The submitted source selects physical GPU `1`; use GPU `0` on a single-GPU machine.

## Apply the patch

From this repository's root:

```bash
git clone https://github.com/TeamFightingICE/FightingLLM.git work/FightingLLM
git -C work/FightingLLM checkout 7b1d5db99b22149b70e4e8c5054cda29abc51536
git -C work/FightingLLM apply --check --ignore-whitespace ../../patches/submitted-runtime.patch
git -C work/FightingLLM apply --ignore-whitespace ../../patches/submitted-runtime.patch
```

The whitespace option handles the upstream line endings. Follow FightingLLM's environment setup and use the CUDA/vLLM versions appropriate to your hardware. The included [requirements-runtime.txt](../requirements-runtime.txt) lists the runtime imports; it is separate from training dependencies.

In `work/FightingLLM/llm_local.py`, change `CUDA_VISIBLE_DEVICES` from `"1"` to `"0"` when using one GPU. The compatibility settings describe the submitted runtime and should be checked when changing vLLM/CUDA versions.

## Use a model

Sign in and request access on Hugging Face, then download [Rushdown v3](https://huggingface.co/ThrowbacKid/rushdown-merged-v3) or [Zoning v3](https://huggingface.co/ThrowbacKid/zoning-merged-v3), including its full tokenizer, to a local directory. Alternatively, use a merged model exported by the [training workflow](reproduction.md).

Start DareFightingICE with PyFTG support (`--pyftg-mode`) and an available opponent. From `work/FightingLLM`, for example:

```bash
python run_llm_vs_mcts.py \
  --llm_model local:/absolute/path/to/merged-model \
  --p2 MctsAi23i \
  --game_num 30 \
  --host 127.0.0.1 --port 31415 \
  --few_shots 0 --prompt_type 1
```

The selected opponent must be installed in the game environment. Set the model directory, opponent, host, port, and game count for your setup; `ModifiedMctsAiZoning` can also be selected when installed. This example starts a new evaluation; it is not a command that reproduces a particular published score.

## Action execution

The controller serializes each game frame, constructs the prompt, generates at most one token, parses the action label, and sends it through PyFTG's command center. Matching/fallback includes `DASH` when a usable action cannot be extracted. Runtime sampling uses temperature 0.2, top-k 10, top-p 0.9, and repetition penalty 1.2. Initial action selection and generation make unseeded matches nondeterministic.
