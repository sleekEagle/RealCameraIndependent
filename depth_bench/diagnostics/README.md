# Stage-1 blur diagnostics

Why does the camera-corrected Camind's stage 1 not learn the blur on MODEST? These scripts back
`reports/blur_learning.md`. Each writes CSV tables and PNG figures to
`reports/blur_learning/diag/<script>/` (or `--out`). Run from the repository root:
`python -m depth_bench.diagnostics.<script>`.

| Script | Question | GPU |
|---|---|---|
| `blur_stats` | How large is the label blur, how is the x-space loss weighted, what do constant outputs score, how wide is the blur range inside a 518 px crop? | no |
| `label_check` | Do the labels match the blur seen in the images (edge-based estimate, Zhuo & Sim 2011)? | no |
| `label_maps` | Figures: label blur maps next to F2.8/F5 images; pixels kept by the texture mask | no |
| `align_check` | Are image and depth aligned in training crops, flipped ones included? | no |
| `dropout_mismatch` | Dropout right before max-pool: train vs eval mode (toy, network, trained checkpoints) | yes |
| `fit_one_batch` | Can stage 1 fit one fixed batch (vs constant-output baselines)? | yes |
| `stream_fit` | Does stage 1 learn from random crops of 20 views? | yes |
| `synth_blur` | Diagnostic only: synthetic blur, variants base / A (camera processing) / B (one blur per crop) / C (small blurs) | yes |
| `probe_report` | Loss curves and per-bin plots of the pod probes (finetune logs, check_blur outputs) | no |

The probes themselves are `depth_bench.finetune` runs of `camind_corr` with `--depth_weight 0`, checked
with `depth_bench.check_blur` (see `reports/blur_learning.md` for the exact commands).
