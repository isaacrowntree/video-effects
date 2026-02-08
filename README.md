# Video Effects

Artistic video processing effects using temporal analysis, computer vision, and AI. Built for GPU-accelerated workflows with ffmpeg (NVENC/CUDA) and PyTorch.

## Effects

### Motion Blur Flow (`motion_blur_flow.py`)

Frame-trailing ghost effect that creates dreamy motion trails on moving subjects. Camera motion is compensated via phase correlation so only independently moving, in-focus subjects get the trail — backgrounds stay sharp.

Three ghost styles:
- **glow** — Overexposed ghosts with soft bloom, screen-blended (bright trails)
- **shadow** — Underexposed dark silhouette ghosts, multiply-blended
- **normal** — Standard weighted average blend (preserves original colours)

#### How It Works

1. **Pass 1** — Decodes at 540p grayscale. Computes cumulative camera motion via phase correlation between consecutive frames.
2. **Optional masking** — Focus detection (Laplacian variance) + motion detection isolate moving, in-focus subjects. Static/bokeh regions are preserved.
3. **Pass 2** — Stacks time-offset copies of the full-res video with decaying opacity, compensating for camera motion. Applies the selected ghost style blend.

```bash
# Glow trails (default)
python3 motion_blur_flow.py input.MP4 output.MP4

# Dark silhouette trails
python3 motion_blur_flow.py input.MP4 output.MP4 --ghost=shadow

# Normal blend (colour-preserving, good for composites)
python3 motion_blur_flow.py input.MP4 output.MP4 --ghost=normal

# Subtle trails
python3 motion_blur_flow.py input.MP4 output.MP4 --layers=4 --decay=0.5 --spacing=4

# Heavy trails with more layers
python3 motion_blur_flow.py input.MP4 output.MP4 --layers=10 --decay=0.8 --spacing=3

# With focus+motion masking (only trail on moving subjects)
python3 motion_blur_flow.py input.MP4 output.MP4 --mask --motion-thresh=8

# Process a specific time range
python3 motion_blur_flow.py input.MP4 output.MP4 --start=30 --duration=10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--layers=N` | 6 | Number of trailing layers. More = longer trail. |
| `--decay=F` | 0.7 | Opacity decay per layer (0.1-0.9). Lower = faster fadeout. |
| `--spacing=N` | 5 | Frame gap between layers. Larger = more spread out trail. |
| `--ghost=MODE` | glow | Ghost style: `glow`, `shadow`, `normal`. |
| `--mask` | off | Enable focus+motion masking (isolate moving subjects). |
| `--motion-thresh=N` | 10 | Motion sensitivity 1-50 (only with `--mask`). |
| `--focus-block=N` | 31 | Laplacian sharpness block size, odd (only with `--mask`). |
| `--qp=N` | 10 | Output H.264 quality parameter. Lower = higher quality. |
| `--start=N` | — | Start time in seconds. |
| `--duration=N` | full | Process N seconds of video. |

---

### Background Generation & Studio Replacement (`bg_generate.py`)

AI-powered background manipulation using alpha mattes from video matting. Four modes covering background extraction, generation, compositing, and full studio replacement with parallax.

#### Modes

##### `plate` — Clean Background Plate

Reconstructs the real background from temporal median of visible background pixels across all frames. Fills remaining holes (areas always occluded by the subject) with inpainting.

```bash
# From pre-computed alpha matte
python3 bg_generate.py plate input.MP4 alpha.mp4 output_bg.png

# With auto person segmentation (no pre-computed alpha needed)
python3 bg_generate.py plate input.MP4 auto output_bg.png

# Specify inpainting method
python3 bg_generate.py plate input.MP4 alpha.mp4 output_bg.png --inpaint=lama
```

##### `generate` — AI Background Generation

Uses Stable Diffusion inpainting to generate a new background from a text prompt.

```bash
python3 bg_generate.py generate input.MP4 alpha.mp4 output_bg.png \
    --prompt="sunny beach at golden hour" --seed=42
```

##### `composite` — Video Compositing

Combines a background plate + original video + alpha matte to produce a final video with the subject on a new background.

```bash
python3 bg_generate.py composite input.MP4 alpha.mp4 output.MP4 --bg=plate.png
```

##### `studio` — Studio Background Replacement with Parallax

Full pipeline that replaces the background with an AI-generated scene featuring depth-based 2.5D parallax, exposure tracking, and light-aware generation. Designed for handheld camera footage where the background needs to move naturally with camera motion.

**How It Works:**

1. **Phase 1 — Analysis** (CPU, ~5s): Single-pass at 360p computes camera motion (phase correlation), per-frame foreground exposure ratios (smoothed), light source direction (brightness centroid), and camera height detection.

2. **Phase 2 — Generation** (GPU, sequential):
   - Runs **Depth Anything V2 Small** on the original scene's reference frame to capture real camera geometry (floor position, wall layout, vanishing points). Heavy Gaussian blur removes person-shaped depth features while preserving room-scale geometry.
   - Generates new background with **SD 1.5 Inpainting + ControlNet Depth** (`lllyasviel/control_v11f1p_sd15_depth`). The original scene depth guides the generated layout so the new background matches real camera height and perspective. Background is generated at oversized dimensions (auto-computed from camera drift) for parallax overscan.
   - Models are loaded/unloaded sequentially to fit in ~4-6GB VRAM.

3. **Phase 3 — Streaming Render** (GPU encode, ~120fps):
   - Per-frame: compute camera offset from reference → parallax warp background (near objects shift more than far) → adjust brightness for exposure tracking → alpha composite foreground → encode.
   - Connected component filtering removes stray alpha artifacts (small disconnected blobs < 40% of main subject area).

```bash
# Basic studio replacement
python3 bg_generate.py studio input.MP4 alpha.mp4 output.MP4 \
    --prompt="neon dance studio with mirrors and LED strip lighting" --seed=42

# Dark space room
python3 bg_generate.py studio input.MP4 alpha.mp4 output.MP4 \
    --prompt="dark room with floor-to-ceiling glass windows looking out into deep space, black void with distant stars visible through glass panels, reflective black glass floor" \
    --seed=42

# Low-res preview
python3 bg_generate.py studio input.MP4 alpha.mp4 output.MP4 \
    --prompt="..." --resolution=360 --seed=42

# Adjust parallax intensity
python3 bg_generate.py studio input.MP4 alpha.mp4 output.MP4 \
    --prompt="..." --depth-scale=0.5 --seed=42
```

**Studio-specific options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--depth-scale=F` | 0.3 | Parallax magnitude multiplier. Higher = more depth separation. |
| `--bg-margin=F` | 0 (auto) | Overscan factor for background. 0 = auto-compute from camera drift. |
| `--exposure-smooth=N` | 15 | Temporal smoothing window (frames) for exposure tracking. |
| `--sd-model=MODEL` | sd15 | SD model: `sd15` or `sdxl`. |

**Common options (all modes):**

| Option | Default | Description |
|--------|---------|-------------|
| `--resolution=N` | 1080 | Processing resolution height. |
| `--prompt=TEXT` | — | Background description (generate/studio modes). |
| `--strength=F` | 0.95 | SD inpainting strength. |
| `--steps=N` | 30 | SD inference steps. |
| `--seed=N` | random | RNG seed for reproducible generation. |
| `--qp=N` | 14 | Output video quality (NVENC). Lower = higher quality. |
| `--duration=N` | full | Process N seconds of video. |
| `--start=N` | — | Start time in seconds. |

**VRAM usage** (sequential, never loaded together):
- Depth Anything V2 Small: ~60MB
- SD 1.5 + ControlNet Depth: ~3.5GB
- Render phase: ~200MB (ffmpeg encode/decode)

---

### Video Matting (`test_matting.py`)

Compares background removal / video matting models. Outputs an alpha matte video and a green-screen composite for each model.

```bash
python3 test_matting.py input.MP4 output_dir/ --model=rvm [--duration=10] [--resolution=1080]
```

#### Models

| Model | Backend | Notes |
|-------|---------|-------|
| `rvm` | PyTorch (ResNet50) | [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting). Temporal recurrence for video consistency. Fast on GPU. |
| `rembg` | ONNX (BiRefNet) | Frame-by-frame via [rembg](https://github.com/danielgatis/rembg) with `birefnet-general` model. |
| `rmbg2` | PyTorch (Transformers) | [BRIA RMBG 2.0](https://huggingface.co/briaai/RMBG-2.0) segmentation pipeline. |

Each model outputs:
- `{model}_alpha.mp4` — Grayscale alpha matte
- `{model}_composite.mp4` — Foreground composited on green screen

[MatAnyone](https://github.com/pq-yang/MatAnyone) was also tested manually — best quality for temporally consistent matting with a first-frame mask. See `run_matanyone_chunks.py` for processing long videos.

---

### MatAnyone Chunked Processing (`run_matanyone_chunks.py`)

Wrapper for [MatAnyone](https://github.com/pq-yang/MatAnyone) that processes long videos in chunks to avoid out-of-memory errors. MatAnyone stores all frames and intermediate propagation states in RAM, which can exceed 20GB for videos over ~30 seconds at 720p.

**Workflow:**
1. Split video into ~25-second chunks with ffmpeg (`-c copy`)
2. Process chunk 0 with the original first-frame mask
3. For subsequent chunks, extract the last frame's alpha from the previous chunk as the guidance mask — maintains temporal continuity across chunk boundaries
4. Concatenate all alpha (and foreground) videos with ffmpeg concat demuxer

```bash
# Prepare: split video and scale mask
ffmpeg -y -ss 0 -t 25 -i full_360p.mp4 -c copy chunks/chunk_0.mp4
ffmpeg -y -ss 25 -t 25 -i full_360p.mp4 -c copy chunks/chunk_1.mp4
# ... etc

# Run (edit paths at top of script)
python3 run_matanyone_chunks.py
```

**Memory requirements:**
- 360p, 25s chunks (~1500 frames): ~12GB RAM
- 720p, 25s chunks (~1500 frames): ~22GB+ RAM (will OOM on 24GB systems)

**Important:** Input video must be 8-bit (yuv420p). 10-bit HEVC sources should be transcoded first:
```bash
ffmpeg -i source_4k.MP4 -vf "scale=640:360" -pix_fmt yuv420p -c:v libx264 -crf 18 output_360p.mp4
```

---

## Full Pipeline Example

End-to-end workflow: matte dancers → generate studio background → add ghost motion trails.

```bash
# 1. Transcode source to 8-bit at working resolution
ffmpeg -i source.MP4 -vf "scale=640:360" -pix_fmt yuv420p -c:v libx264 -crf 18 input_360p.mp4

# 2. Create first-frame mask (white silhouette of subject, e.g. in GIMP/Photoshop)
#    Save as first_frame_mask.png at same resolution as input

# 3. Run MatAnyone for alpha matte (chunk if video > 30s)
#    Produces: matanyone/input_360p_pha.mp4

# 4. Studio background replacement with parallax
python3 bg_generate.py studio input_360p.mp4 matanyone/input_360p_pha.mp4 studio_output.mp4 \
    --prompt="dark room with glass windows looking into deep space" \
    --resolution=360 --seed=42

# 5. Add ghost motion trails (optional)
python3 motion_blur_flow.py studio_output.mp4 final_output.mp4 \
    --ghost=normal --layers=4 --decay=0.5 --spacing=4 --qp=14
```

---

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (12GB+ VRAM recommended)
- ffmpeg with CUDA hwaccel and NVENC encoder
- NumPy, OpenCV (`opencv-python`), SciPy

**Python packages by feature:**

| Feature | Packages |
|---------|----------|
| Motion blur flow | numpy, opencv-python, scipy |
| Video matting (RVM) | torch, torchvision |
| Video matting (MatAnyone) | [matanyone](https://github.com/pq-yang/MatAnyone), torch |
| Background generation | diffusers, transformers, torch |
| Studio mode (ControlNet) | diffusers, transformers, torch, scipy |
| Clean plate (LaMa) | simple-lama-inpainting |
| Auto masking (DeepLabV3) | torch, torchvision |

**HuggingFace models used:**
- `runwayml/stable-diffusion-inpainting` — SD 1.5 inpainting (~5GB)
- `lllyasviel/control_v11f1p_sd15_depth` — ControlNet depth conditioning (~1.4GB)
- `depth-anything/Depth-Anything-V2-Small-hf` — Monocular depth estimation (~95MB)
- `PeiqingYang/MatAnyone` — Video matting (~135MB)

## License

MIT
