# References and model resources

## Models

- [Rushdown v3 weights and full tokenizer](https://huggingface.co/ThrowbacKid/rushdown-merged-v3)
- [Zoning v3 weights and full tokenizer](https://huggingface.co/ThrowbacKid/zoning-merged-v3)
- [Qwen2.5-1.5B Base](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [Configuration snapshots](../configs/)

The Hugging Face models contain the merged weights and tokenizer files. Configuration snapshots in this repository describe their exported architecture and tokenization; they are not weight files. The stored `transformers_version` is 5.6.2.

## Platform and methods

- [DareFightingICE platform](https://github.com/TeamFightingICE/FightingICE)
- [FightingLLM framework](https://github.com/TeamFightingICE/FightingLLM)
- [2026 competition rules](https://github.com/TeamFightingICE/FightingICE/tree/master/DareFightingICE/AI)
- [Official competition results](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/index-R.html)
- [Organizer's submission and match archive](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/Downloadfiles/2026AI.zip)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DAgger](https://proceedings.mlr.press/v15/ross11a.html)

Acknowledgments: Team FightingICE for the platform/framework, BlackMamba's developers for the expert agent, and the Qwen team for the base model. The [runtime patch](../patches/submitted-runtime.patch) builds on the FightingLLM framework at commit `7b1d5db99b22149b70e4e8c5054cda29abc51536`.

## Competition result calculation

[official_standard_summary.csv](../results/official_standard_summary.csv) summarizes MambaGPT's `HPMode_*.csv` records in the organizer archive's `log/point-style/stdRUSHDOWN/` and `stdZONING/` directories.

Each style contains 18 game files with three round rows, giving 54 rounds. Player positions come from the filename; row columns are round, P1 final HP, P2 final HP, and frames. Following the organizer's calculation convention, greater final HP counts as a win, lower HP as a loss, and equal HP as a draw.

Rushdown totals are 42 wins, 12 losses, and no draws. Zoning totals are 26 wins, 23 losses, and 5 draws. Each opponent contributes six rounds per style. `ALL` rows are totals. The overall third-place result also includes the competition's other scoring components and is taken from the official results page.

## Data and code

The [dataset manifest](../datasets/manifest.json) records full-file counts/hashes and included sample counts/hashes. The [source manifest](source_manifest.json) records repository script hashes. Original weekly reports and signed documents are not distributed.

No repository-wide software license has been selected. Third-party components retain their own terms; see the [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE) and upstream project documentation before redistributing dependencies.
