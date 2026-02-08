# Video Effects

Artistic video processing effects using temporal analysis and computer vision.

## Effects

### Motion Blur Flow

Temporal median blending that creates dreamy, ghostly motion trails on **in-focus moving subjects only**. Out-of-focus foreground/background (bokeh) and static regions are preserved.

#### How It Works

1. **Pass 1** — Decodes video at 540p grayscale. For each frame, computes a temporal median over a sliding window of neighboring frames.
2. **Focus detection** — Local Laplacian variance identifies sharp (in-focus) vs blurry (bokeh) regions. Auto-calibrates threshold from sample frames.
3. **Motion detection** — Frame-to-median difference isolates pixels that are actually moving. Static background gets zero correction.
4. **Pass 2** — Upscales the masked correction deltas and applies them to the full-resolution video with NVENC hardware encoding.

#### Quick Start

```bash
# Basic dreamy motion blur on a clip
python3 motion_blur_flow.py input.MP4 output.MP4

# Heavy effect with larger temporal window
python3 motion_blur_flow.py input.MP4 output.MP4 --radius=45 --strength=0.9

# Process a specific time range
python3 motion_blur_flow.py input.MP4 output.MP4 --start=30 --duration=10

# Tune focus/motion sensitivity
python3 motion_blur_flow.py input.MP4 output.MP4 --motion-thresh=12 --focus-block=51
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--radius=N` | 15 | Temporal radius in frames. Larger = more dreamlike trails (try 30-60). |
| `--strength=F` | 0.8 | Blend strength 0.0-1.0. How much to push toward the temporal median. |
| `--motion-thresh=N` | 8 | Motion sensitivity (1-50). Lower = more pixels treated as "moving". |
| `--focus-block=N` | 31 | Laplacian sharpness block size (odd). Larger = coarser focus regions. |
| `--qp=N` | 10 | Output H.264 quality parameter. Lower = higher quality. |
| `--start=N` | 0 | Start time in seconds. |
| `--duration=N` | full | Process N seconds of video. |

## Requirements

- Python 3.8+
- NumPy
- OpenCV (`opencv-python`)
- ffmpeg with CUDA hwaccel and NVENC encoder
- ffprobe

## License

MIT
