# Experiments and lessons

MambaGPT developed through model-format trials, dataset revisions, checkpoint comparisons, continued fine-tuning, and DAgger. The final competition entry used separate Rushdown and Zoning v3 policies based on Qwen2.5-1.5B Base.

## Development timeline

| Period in 2026 | Work |
|---|---|
| April | Set up FightingICE and expert demonstrations; tried Qwen2.5-1.5B-Instruct with LoRA; introduced action tokens and moved to a Base-model approach after invalid and tool-call outputs. |
| May | Worked on dataset balance, controllable-state calls, adapter export, prompt alignment, repeated-action cleanup, and position augmentation. |
| Early June | Compared position-jitter and no-position-jitter training data; evaluated checkpoints in the game and selected v3 as the main reference policy. |
| June | Collected v4 data, tested less aggressive cleaning for v5, and repaired zoning/projectile JSON formats. |
| Late June–early July | Built v6's fireball filtering and v7's continued fine-tuning; added gameplay-style metrics and corrected the measured player for cornering. |
| Mid July | Tested disagreement-focused DAgger in v8 and proposed collecting all expert actions for v9. |
| August | Competition entry used the v3 Rushdown and Zoning policies. |

## Checkpoint comparison: position augmentation and v3

The dataset initially used synthetic changes to position, HP, and energy. Changing positions can alter the meaning of corner situations, so the later v3 preparation kept positions fixed and augmented HP/energy only.

Each checkpoint in this comparison was evaluated over 100 games against the baseline.

| Dataset | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| With position jitter | 86% | 66% | 42% |
| **v3, without position jitter** | **98%** | 49% | 94% |
| v3 offline validation accuracy | 84% | 89% | 93% |

The gameplay results did not improve monotonically with offline validation. This made match rollouts necessary for selecting a policy, rather than selecting the checkpoint with the highest label accuracy alone.

## Rushdown versions

| Version | Change | Baseline win rate |
|---|---|---:|
| **v3** | Cleaned/rebalanced expert demonstrations; no position jitter | **98%** |
| v4 | Recollected corpus using the established training approach | 18% |
| v5 | Less aggressive cleanup and a continuation-example cap during augmentation | 83% |
| v6 | Filter selected examples where the current and target action are both `STAND_D_DF_FB` | 51% |
| v7 | Continue fine-tuning v3 with the v6 dataset | 94% |
| v8 | Fine-tune with disagreement-focused expert labels mixed with v3 examples | 20% |

These are local development results against the baseline, not overall competition win rates.

### v3: action balance and useful repetition

BlackMamba demonstrations had frequent actions, rare actions, and many examples where the current action matched the target. Processing removed invalid rows, exact duplicates, and near duplicates, then rebalanced the action distribution. Repeated `BACK_STEP` examples were filtered to address retreat loops.

This improved the baseline candidate, but it also exposed a tradeoff: repeated actions can be intentional. Multiple retreats may be needed to create distance; continuing an attack can also be useful. Cleaning every repeated action equally can remove valid behavior.

### v4: new data did not guarantee a stronger policy

The v4 experiment recollected data and retrained through the same general pipeline. Its 18% baseline result was substantially below v3. The practical lesson was to evaluate data revisions in the game rather than assume that a larger or newer corpus improves a sequential policy.

The raw and processed v4 samples are included in [the dataset collection](../datasets/README.md), together with the cleaner used for this branch.

### v5: retain more continuation examples

v5 revisited aggressive duplicate/repetition filtering. It retained more examples and aimed to cap examples with `self.action == output` at 50% during augmentation. This produced an 83% baseline result, improving on v4 while remaining below v3.

The corresponding scripts are `clean_dataset_v5.py` and `rebalance_dataset_v5.py`. The rebalancer augments HP and energy while retaining position. Its fixed action-distribution template requires examples for all 40 expected classes.

### v6: target projectile loops

v6 filtered selected repeated-fireball examples to reduce `STAND_D_DF_FB` loops. Offline validation reached about 90%, while the best baseline result was 51%. This reinforced the distinction between predicting labels accurately and maintaining a strong policy during long game rollouts.

Its data path uses `clean_rushdown_v6.py`, `rebalance_rushdown_v3.py`, and `split_database_v3.py`.

### v7: continue from an established policy

v7 continued training from v3 using v6 data, including the embedding and output-head modules. Its offline validation reached 92%; epoch-1 and epoch-2 baseline results were 87% and 94%.

Evaluation also tracked distance, cornering, projectiles, repeated fireballs, HP changes, and combo proxies. The cornering logger was corrected to measure the opponent's cornered frames. The experiment highlighted the need to validate metric definitions as well as model behavior.

### v8: disagreement-focused DAgger

v8 used v3 as the learner and BlackMamba as the expert. States visited by the learner were labeled by the expert, with a focus on disagreements. The training mixture combined these corrective examples with v3 data.

The data-processing stages were:

| Stage | Full-data rows |
|---|---:|
| Disagreement records | 30,509 |
| After exact deduplication | 30,210 |
| After excluding negative remaining-frame examples | 30,177 |
| Stratified v3 examples added | 60,000 |
| Combined v8 training set | 90,177 |

The baseline result was 20%. Corrective disagreement examples did not preserve the earlier policy's performance. The next direction was broader expert supervision while maintaining coverage of the original demonstrations.

### v9: future direction

The proposed v9 approach would log all expert actions on learner-visited states, rather than collect disagreements alone. It was intended to broaden the training distribution. v9 is future work and has no result in the comparison table.

## Zoning

The final zoning dataset used **Modified MctsAiZoning**. Its changes favored distance maintenance and prioritized zoning move branches during search. Data collection also introduced aggressive opponents to expose the learner to close-range pressure and retreat situations, rather than relying only on passive zoning-versus-zoning play.

Zoning development included JSON/projectile repairs, duplicate removal, rebalancing, and policy evaluation. A local baseline result of 15% preceded the final 20% development figure. The competition additionally evaluated whether the policy expressed the requested fighting style.

## Other opponent comparisons

Local MCTS comparisons recorded 37% for v3 and about 29% for v5. These are separate opponent evaluations from the baseline table above.

## Competition results

Team MambaGPT placed **third overall** in the 2026 competition. Standard-league round counts from the organizer's published matches are 42 wins/12 losses for Rushdown and 26 wins/23 losses/5 draws for Zoning, each over 18 games and 54 rounds.

[Detailed competition summary](../results/official_standard_summary.csv) · [Calculation and references](provenance.md).

## Further directions

Additional directions considered during development included weighted training for rare actions, soft targets from MCTS search statistics, alternative model sizes, prompt optimization, and full-parameter fine-tuning. The implemented comparisons above remain the basis for the project results.
