# AffectGPT CUDA Pipeline — 4 Steps

Run these commands from the extracted `affectgpt_cuda_pipeline` directory on a Linux NVIDIA CUDA machine.

## 1. Install dependencies

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg curl time
bash setup_cuda.sh
```

## 2. Download model weights

```bash
conda run -n affectgpt-cuda python download_models.py \
  --code-dir "$PWD/AffectGPT/AffectGPT"
```

## 3. Test one video

Replace `/data/context_clips` with the folder containing your MP4 videos (including any `train/` and `test/` subfolders).

```bash
conda run -n affectgpt-cuda python -u batch_affectgpt_cuda.py \
  --code-dir "$PWD/AffectGPT/AffectGPT" \
  --input /data/context_clips \
  --output /data/affectgpt_npz \
  --limit 1
```

## 4. Run the full dataset and record runtime

```bash
set -o pipefail

/usr/bin/time -p conda run -n affectgpt-cuda \
  python -u batch_affectgpt_cuda.py \
    --code-dir "$PWD/AffectGPT/AffectGPT" \
    --input /data/context_clips \
    --output /data/affectgpt_npz \
  2>&1 | tee affectgpt_full.log
```
