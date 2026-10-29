# Methodology

MambaGPT learns fighting-game decisions through supervised imitation, then predicts an action label from the current game state.

## Expert demonstrations

**Rushdown** uses BlackMamba, a Java agent with a six-layer MLP policy trained using PPO. Gameplay observations are paired with the expert's action labels.

**Zoning** uses Modified MctsAiZoning. Its search favors distance maintenance and examines zoning move branches first. Aggressive opponents are used during collection to generate close-range pressure and retreat situations as well as passive zoning states.

The Qwen policies are trained with supervised state/action targets. PPO belongs to the BlackMamba teacher's training approach.

## Dataset preparation

The processing pipeline validates state/action records, removes invalid actions and exact duplicates, groups near-duplicate states, filters selected repeated-action examples, and rebalances action frequencies.

Near-duplicate keys use game properties such as action/state, distance, energy, velocity, corner position, and projectile counts. Loop-prone actions use stricter retention limits. Later versions refine continuation filtering, including selected `BACK_STEP` and `STAND_D_DF_FB` repetitions.

Rebalancing downsamples frequent actions and augments scarce examples. The no-position-jitter route varies HP in 1–400 and energy in 0–300 while preserving positions. `STAND_D_DF_FC` examples retain at least 100 self energy. Action-stratified train/validation splitting uses seed 42 and approximately 90/10 proportions.

The full v3 Rushdown split contains **151,861 training** and **16,878 validation** examples. The full v3 Zoning split contains **389,050 training** and **43,226 validation** examples. The [included dataset samples](../datasets/README.md) are smaller excerpts from these and other pipeline stages.

## Model architecture

Both v3 policies use **Qwen2.5-1.5B Base**. The exported configurations included in [configs](../configs/) specify:

| Setting | Value |
|---|---|
| Architecture | `Qwen2ForCausalLM` |
| Layers / hidden dimension | 28 / 1,536 |
| Attention heads / key-value heads | 12 / 2 |
| MLP intermediate dimension | 8,960 |
| Activation | SiLU |
| Export dtype | FP16 |
| Vocabulary size | 151,936 |
| Tied input/output embeddings | False |
| Registered action tokens | 41 |

Action names such as `DASH`, `CROUCH_B`, and `AIR_DA` are registered as special tokens. Input embeddings are resized for the tokenizer. The merged export retains the trained input embedding and output-head modules, with untied weights.

## Observation and prompt

Observations have two top-level objects: `self` and `opponent`. Each contains HP, energy, current action, state, remaining action frames, bounding-box position, x/y velocity, and projectiles. Projectile records include damage, bounds, and velocity.

For Rushdown, the instruction, state JSON, and output marker form the prompt:

```text
Instruction:Predict the next FightingICE action.
Input:{compact game-state JSON}
Output:
```

The target is an action followed by EOS. Prompt and padding positions have label `-100`, so the training loss focuses on the answer. Dataset loaders accept both `messages` rows and `instruction`/`input`/`output` rows; the dataset's instruction is retained.

## Fine-tuning

| Setting | Value |
|---|---|
| Quantization | NF4, 4-bit, double quantization, FP16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Additional trainable/saved modules | `embed_tokens`, `lm_head` |
| Optimizer / learning rate | AdamW / `2e-5` |
| Batch size / epochs | 16 / 3 |
| Maximum sequence length | 384 tokens |
| Gradient clipping | Norm 1.0 |
| Memory optimization | Gradient checkpointing |
| Default stopping rule | Complete all three epochs; no step cap |
| Checkpoints | After each completed epoch, plus periodic step snapshots |

The repository CLI supports an optional step cap for short checks. It is disabled by default. Separate checkpoints allow action prediction and gameplay to be evaluated before selecting a model for use.

The model configuration's context-capacity field is 131,072; training examples use 384-token sequences, and the submitted runtime uses a 2,048-token context setting. These settings serve different purposes.

## Inference and evaluation

At runtime, compact state JSON is wrapped in the training-style prompt and passed to vLLM. The controller requests one generated token, matches it to an action, and uses fallback handling when needed. The training/evaluation action list has 41 entries; the full v3 Rushdown corpus contains 40 observed output labels, with `AIR_FA` absent from that corpus.

Offline evaluation measures exact action prediction, valid vocabulary membership, and argmax accuracy within the action-token list. Vocabulary membership is distinct from whether an action is executable in a particular game state.

Gameplay evaluation measures wins, HP, distance, projectiles, repetition, cornering, and combo proxies. The [experiment history](experiments.md) explains how these evaluations influenced model selection. The [inference guide](inference.md) details the runtime changes.
