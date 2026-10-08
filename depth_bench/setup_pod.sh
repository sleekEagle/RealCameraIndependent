#!/usr/bin/env bash
# One-command setup and run of the full benchmark on a RunPod pod (template: RunPod PyTorch).
#
# Before: send the packed data to the pod (from your PC: runpodctl send D:\datasets\MODEST_colab;
# on the pod: cd /workspace && runpodctl receive <code>), so it is in /workspace/MODEST_colab.
#
# Run (pod terminal):
#   curl -sL https://raw.githubusercontent.com/sleekEagle/RealCameraIndependent/main/depth_bench/setup_pod.sh | bash
# Options (environment variables), e.g. only some folds or models:
#   FOLDS="S1 S2" MODELS="da2 camind" ZERO_SHOT=none bash setup_pod.sh
#
# Safe to run again (after a pod restart): finished steps are skipped, training resumes.
# Everything lives on the volume (/workspace), so it survives stopping the pod.
# Follow progress:  tail -f /workspace/MODEST_runs/run_all.log
set -euo pipefail

W=/workspace
DATA_SRC=${DATA_SRC:-$W/MODEST_colab}
FOLDS=${FOLDS:-"S1 S2 S3 S4 S5"}
MODELS=${MODELS:-"da2 unidepth metric3d da3 camind"}
ZERO_SHOT=${ZERO_SHOT:-"da2 unidepth unidepth_noK metric3d da3"}

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CPU cores: $(nproc)"

echo "== code"
cd $W
if [ -d RealCameraIndependent ]; then git -C RealCameraIndependent pull -q
else git clone -q https://github.com/sleekEagle/RealCameraIndependent; fi
mkdir -p code_ext && cd code_ext
clone() {  # name url commit (same versions as the local runs)
  [ -d "$1" ] || git clone -q "$2" "$1"
  git -C "$1" checkout -q "$3"
}
clone Depth-Anything-V2 https://github.com/DepthAnything/Depth-Anything-V2   a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
clone UniDepth          https://github.com/lpiccinelli-eth/UniDepth          8d8cfe4c7ee15297099983607febf0d4f32eb3d6
clone Metric3D          https://github.com/YvanYin/Metric3D                  eb5b6fac0dc155e4e52f576e304fbf11655ff339
clone Depth-Anything-3  https://github.com/ByteDance-Seed/Depth-Anything-3   3d835ec1a5802d64a8b8b15f817a1ab54809bfe4

echo "== packages (the template's PyTorch is kept)"
pip install -q opencv-python-headless tifffile imagecodecs pillow pandas scipy tqdm huggingface_hub safetensors \
               timm einops addict omegaconf wandb h5py tabulate termcolor imageio mmengine
# --no-deps: their pinned requirements conflict (numpy); both work without them
pip install -q --no-deps -e $W/code_ext/UniDepth -e $W/code_ext/Depth-Anything-3
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== data"
mkdir -p $W/MODEST_depth
ls "$DATA_SRC"/Scene*.tar > /dev/null || { echo "no tars in $DATA_SRC (send MODEST_colab to the pod first)"; exit 1; }
for t in "$DATA_SRC"/Scene*.tar; do
  flag=$W/MODEST_depth/.$(basename "$t").done
  if [ ! -f "$flag" ]; then echo "extracting $(basename "$t")"; tar -xf "$t" -C $W/MODEST_depth && touch "$flag"; fi
done
cp "$DATA_SRC"/manifest.csv $W/MODEST_depth/
cp "$DATA_SRC"/dfocus_aperture_pairs.csv $W/
df -h $W | tail -1

echo "== run"
export MODEST_DEPTH_ROOT=$W/MODEST_depth MODEST_FOCUS_CSV=$W/dfocus_aperture_pairs.csv \
       DEPTH_BENCH_CODE_EXT=$W/code_ext DEPTH_BENCH_RUNS=$W/MODEST_runs \
       HF_HOME=$W/hf_cache TORCH_HOME=$W/torch_cache
mkdir -p $W/MODEST_runs
cd $W/RealCameraIndependent
if pgrep -f "depth_bench.run_all" > /dev/null; then echo "run_all is already running"; exit 0; fi
python -m depth_bench.run_all --dry_run --folds $FOLDS --models $MODELS --zero_shot $ZERO_SHOT
nohup python -m depth_bench.run_all --folds $FOLDS --models $MODELS --zero_shot $ZERO_SHOT \
      >> $W/MODEST_runs/run_all.log 2>&1 &
echo "started (pid $!). Follow with:  tail -f $W/MODEST_runs/run_all.log"
