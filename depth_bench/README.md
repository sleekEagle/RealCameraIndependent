# depth_bench: zero-shot evaluation and fine-tuning on MODEST_depth

Four monocular metric depth baselines, one interface, one training recipe, one evaluation.

| Name | Model | Camera use | Checkpoint |
|---|---|---|---|
| `da2` | Depth Anything V2 metric, ViT-B | none | HF `depth-anything/Depth-Anything-V2-Metric-Hypersim-Base` |
| `unidepth` / `unidepth_noK` | UniDepth V2, ViT-B | given intrinsics / predicts its own | HF `lpiccinelli/unidepth-v2-vitb14` |
| `metric3d` / `metric3d_c6000` | Metric3D v2, ConvNeXt-L | focal length (canonical 1000 px / 6000 px) | HF `JUGGHM/Metric3D` (torch hub) |
| `da3` | Depth Anything 3 metric, large | focal length (depth = f · out / 300) | HF `depth-anything/DA3METRIC-LARGE` |
| `camind` | Camind (WACV 2024) network as a plain two-stage CNN U-Net | none | none: trained from scratch (`camind_net.py`, copied from `sleekEagle/defocus_camind`) |

## Design

- **No resizing.** Training and evaluation use native-resolution crops (default 518 × 518), so the defocus blur stays as the camera recorded it. A crop keeps fx, fy and shifts the principal point (`data.intrinsics`).
- **Same pixels for every model.** Wrappers (`models.py`) only normalize, pad to the model's multiple, and convert the native output to metric depth. Metric3D's ConvNeXt only behaves at its training input size, so its crop is centred in a 544 × 1216 mean-colour canvas; only the crop region is used.
- **Metric3D canonical focal.** Metric3D predicts depth for a 1000 px camera, then multiplies by fx / 1000. Its decoder output is limited to 0.3–150 (canonical). MODEST crops have fx 4,550–12,060 px, so with 1000 px the smallest reachable depth is 1.4 m (28 mm) to 3.6 m (70 mm). Use `metric3d` for zero-shot (official setting) and `metric3d_c6000` for fine-tuning, where the smallest reachable depth is 0.23–0.6 m.
- **Camind** is trained from scratch with the same loop (all weights trained, `--lr_head`), on depth MSE only. It is used as a plain two-stage CNN, unlike the paper: no blur supervision (no auxiliary loss on the stage-1 output), no camera correction (the stage-1 output goes to stage 2 unchanged), and no focus distance (stage 2's second input channel is a constant 1). It has no zero-shot result.
- **Splits** come from `MODEST_depth/manifest.csv` (columns S1–S5): `train`, `val`, `test_seen`, `test_unseenF`, `test_unseenfl`, `test_both`.
- **Evaluation** uses fixed, seeded crops per test image (same for every model and run). Metrics per crop, on valid pixels (0.3–20 m):
  - AbsRel, RMSE, δ₁ / δ₂ / δ₃ (share of pixels within 1.25, 1.25², 1.25³ of the truth);
  - SILog (×100; standard deviation of the log error, ignores global scale);
  - scale-invariant AbsRel (after median scaling) and the median scale gt/pred.
  - boundary F1 / precision / recall (`bf1`, `bprec`, `brec`): Depth Pro's scale-invariant boundary F1 (Bochkovskii et al., ICLR 2025), counting only valid ground-truth pairs and on every 4th pixel (our stereo depth edges are ~4 px wide); NaN for crops without a true depth edge.

  `summary.csv` averages them per test group and also gives the median AbsRel over crops.
- **Blur-binned AbsRel.** Each pixel's blur diameter in pixels is computed from its true depth: fx · (f / N) · |1/d − 1/s|, with f the real lens focal length (`f_true_mm`), N the F-number and s the focus distance from `MODEST_focus/dfocus_aperture_pairs.csv` (per scene, focal length and camera). AbsRel is computed per blur bin (0–1, 1–2, 2–4, … , 64+ px). `blur_summary.csv` gives the pixel-weighted AbsRel per bin and test group; `per_crop.csv` keeps the per-bin values and the median blur per crop.
- **Fine-tuning** recipe (same for all): head + last 25% of encoder blocks, AdamW (head 1e-4, encoder 1e-5), warm-up + cosine, SILog loss, mixed precision, gradient accumulation, budget in crop samples, best checkpoint by validation AbsRel, resumable.

## Install (Windows or Linux)

```
conda create -n depthbench python=3.11
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy opencv-python-headless tifffile imagecodecs pillow pandas matplotlib scipy tqdm \
            huggingface_hub safetensors timm einops addict omegaconf wandb h5py tabulate termcolor imageio mmengine
git clone https://github.com/DepthAnything/Depth-Anything-V2  <CODE_EXT>/Depth-Anything-V2
git clone https://github.com/lpiccinelli-eth/UniDepth          <CODE_EXT>/UniDepth
git clone https://github.com/YvanYin/Metric3D                  <CODE_EXT>/Metric3D
git clone https://github.com/ByteDance-Seed/Depth-Anything-3   <CODE_EXT>/Depth-Anything-3
pip install --no-deps -e <CODE_EXT>/UniDepth -e <CODE_EXT>/Depth-Anything-3
```

`--no-deps` because UniDepth needs numpy ≥ 2 and DA3 pins numpy < 2 (both work with numpy 2), and both list xformers, which is optional. Metric3D's `mmcv` import is replaced by a small stub in `models.py`.

## Paths

Set with environment variables (defaults are the local Windows paths):

| Variable | Default | Meaning |
|---|---|---|
| `MODEST_DEPTH_ROOT` | `D:\datasets\MODEST_depth` | images, depth maps |
| `MODEST_MANIFEST` | `<MODEST_DEPTH_ROOT>\manifest.csv` | splits and intrinsics |
| `MODEST_FOCUS_CSV` | `D:\datasets\MODEST_focus\dfocus_aperture_pairs.csv` | focus distances (blur bins skipped if missing) |
| `DEPTH_BENCH_CODE_EXT` | `D:\code_ext` | cloned model repos |
| `DEPTH_BENCH_RUNS` | `D:\datasets\MODEST_runs` | outputs |
| `HF_HOME`, `TORCH_HOME` | `D:\hf_cache`, `D:\torch_cache` | checkpoint downloads |

On Kaggle, for example: `MODEST_DEPTH_ROOT=/kaggle/input/<dataset>/MODEST_depth`, `DEPTH_BENCH_RUNS=/kaggle/working/runs`, `DEPTH_BENCH_CODE_EXT=/kaggle/working/code_ext`. The manifest paths are relative to `MODEST_DEPTH_ROOT`; they are written with backslashes, and `data.py` converts them, so the same manifest works on Linux.

## Commands (run from the repository root)

```
# zero-shot, all test views once (group by any fold afterwards from per_crop.csv)
python -m depth_bench.evaluate --model da3 --out zs_da3

# fine-tune one model on one fold, then evaluate its best checkpoint on that fold
python -m depth_bench.finetune --model da3 --fold S1 --samples 40000 --batch 2 --accum 8
python -m depth_bench.evaluate --model da3 --fold S1 --ckpt <RUNS>/finetune/da3_S1/best.pt --out ft_da3_S1

# pilot: every model, tiny budget, small test subset
python -m depth_bench.pilot
```

## Full benchmark on Kaggle

1. Locally: `python -m depth_bench.pack_for_colab` (writes `D:\datasets\MODEST_colab`, 36 GB). Upload it as a private Kaggle dataset, e.g. with the Kaggle CLI: `kaggle datasets init -p D:\datasets\MODEST_colab`, set the title and id in the created `dataset-metadata.json`, then `kaggle datasets create -p D:\datasets\MODEST_colab`.
2. Import `depth_bench/kaggle_run.ipynb` into Kaggle. Settings: GPU T4 x2, Internet on; add the dataset as input.
3. Each session: **Save Version → Save & Run All (Commit)**. It runs up to 12 h without the browser. One `run_all` process per T4; they share the jobs through lock files (`MODEST_runs/locks/`). Training stops 10 min before the session limit and saves `last.pt`; results go to the version's output (`/kaggle/working/MODEST_runs`, limit 20 GB; `last.pt` of finished runs is deleted).
4. From the 2nd session on, add the previous version's output as an input; the notebook copies the runs so far from it and continues.

`kaggle_data.py` prepares the data in the notebook: it extracts the tars to local disk, or, if Kaggle unpacked them on upload, links the scene folders.

## Full benchmark on RunPod

1. Locally: `python -m depth_bench.pack_for_colab` (writes `D:\datasets\MODEST_colab`, 36 GB).
2. Create a pod: template "RunPod PyTorch", e.g. RTX 4090, volume disk 150 GB at `/workspace`, On-Demand.
3. Send the data: on the PC `runpodctl send D:\datasets\MODEST_colab`, on the pod `cd /workspace && runpodctl receive <code>`.
4. On the pod: `curl -sL https://raw.githubusercontent.com/sleekEagle/RealCameraIndependent/main/depth_bench/setup_pod.sh | bash`. It sets everything up and starts `run_all` in the background. Follow with `tail -f /workspace/MODEST_runs/run_all.log`. After a pod restart, run the same command again.

## Full benchmark on Google Colab

1. Locally: `python -m depth_bench.pack_for_colab` writes `D:\datasets\MODEST_colab` (9 scene tars, `manifest.csv`, `dfocus_aperture_pairs.csv`; 36 GB). Upload that folder to Google Drive as `MyDrive/MODEST_colab`.
2. Open `depth_bench/colab_run.ipynb` in Colab (GPU runtime) and run all cells. It clones this repo and the four model repos at fixed commits, installs packages, extracts the data to the local disk, and runs `run_all`.
3. Outputs go to `MyDrive/MODEST_runs` (checkpoints, `logs/`, `eval/`). After a disconnect, run all cells again: finished steps are skipped and training resumes from `last.pt`.

`run_all` order: zero-shot evaluation (`da2`, `unidepth`, `unidepth_noK`, `metric3d`, `da3`) once on all test views, then per fold and model: training (40k crops fine-tuning, 200k crops from scratch for `camind`; 16 crops per optimizer step) and evaluation of `best.pt`. `--dry_run` shows what is done and what would run.

On GPUs without bf16 (T4), mixed precision uses fp16 with loss scaling. `DEPTH_BENCH_AMP=fp16` forces this path on other GPUs for testing.

Outputs: `<RUNS>/eval/<out>/per_crop.csv` (one row per crop, with every fold's labels), `summary.csv` and `blur_summary.csv`; `<RUNS>/finetune/<model>_<fold>/` with `best.pt`, `last.pt` (resume), `log.csv`, `config.json`.
