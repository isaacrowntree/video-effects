#!/usr/bin/env python3
"""
Motion Blur Flow — Temporal median blending for dreamy motion trails.

Applies temporal median blending only to in-focus moving subjects:
- Detects motion via frame-to-median difference
- Detects focus via local Laplacian sharpness variance
- Out-of-focus areas (bokeh) and static regions are preserved

Usage:
    python3 motion_blur_flow.py <input> <output> [options]

Options:
    --radius=N          Temporal radius in frames (default: 15, try 30-60 for heavy effect)
    --strength=F        Blend strength 0.0-1.0 (default: 0.8)
    --motion-thresh=N   Motion sensitivity 1-50 (default: 8)
    --focus-block=N     Focus detection block size, odd (default: 31)
    --qp=N              Output quality (default: 10)
    --start=N           Start time in seconds
    --duration=N        Process N seconds
"""

import subprocess
import sys
import os
import json
import time
import numpy as np
import cv2


def get_video_info(path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,width,height,duration',
        '-show_entries', 'format=duration', '-of', 'json', path
    ]
    data = json.loads(subprocess.check_output(cmd, stderr=subprocess.DEVNULL))
    s = data['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    duration = float(s.get('duration') or data['format']['duration'])
    return {'fps': fps, 'duration': duration,
            'width': int(s['width']), 'height': int(s['height'])}


def parse_opts(argv):
    opts = {
        'qp': 10, 'start': None, 'duration': None,
        'radius': 15, 'strength': 0.8,
        'motion_thresh': 8, 'focus_block': 31,
    }
    for arg in argv:
        if '=' in arg:
            key, val = arg.lstrip('-').replace('-', '_').split('=', 1)
            if key in ('qp', 'radius', 'motion_thresh', 'focus_block'):
                opts[key] = int(val)
            elif key in ('start', 'duration', 'strength'):
                opts[key] = float(val)
    return opts


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    opts = parse_opts(sys.argv[3:])
    radius = opts['radius']
    strength = opts['strength']
    motion_thresh = opts['motion_thresh']
    focus_block = opts['focus_block'] | 1  # ensure odd

    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        sys.exit(1)

    info = get_video_info(input_path)
    h, w = info['height'], info['width']

    # Measurement resolution — 540p grayscale
    meas_h = 540
    meas_w = int(w / h * meas_h)
    meas_w -= meas_w % 2

    print(f"Input: {input_path}")
    print(f"  {w}x{h} @ {info['fps']:.2f}fps")
    print(f"  Radius={radius}  Strength={strength}")
    print(f"  Motion threshold={motion_thresh}  Focus block={focus_block}")

    start_args = ['-ss', str(opts['start'])] if opts['start'] else []
    dur_args = ['-t', str(opts['duration'])] if opts['duration'] else []

    # ── Pass 1: Read measurement frames ──────────────────────────
    print(f"\nPass 1: Loading frames at {meas_w}x{meas_h}...")
    cmd = [
        'ffmpeg', '-hwaccel', 'cuda', *start_args, *dur_args,
        '-i', input_path,
        '-vf', f'scale={meas_w}:{meas_h},format=gray',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-v', 'error', '-'
    ]
    dec = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = meas_w * meas_h
    frames_list = []
    while True:
        raw = dec.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frames_list.append(np.frombuffer(raw, dtype=np.uint8).reshape(meas_h, meas_w))
    dec.wait()

    num_frames = len(frames_list)
    # Stack as contiguous uint8 array — 4x smaller than float32
    pixel_data = np.stack(frames_list)  # (N, H, W) uint8
    del frames_list
    print(f"  {num_frames} frames ({pixel_data.nbytes / 1e6:.0f} MB)")

    # ── Auto-calibrate focus threshold ───────────────────────────
    focus_samples = []
    for i in range(0, min(num_frames, 120), 15):
        lap = cv2.Laplacian(pixel_data[i], cv2.CV_32F)
        local_var = cv2.blur(lap * lap, (focus_block, focus_block))
        focus_samples.append(float(np.percentile(local_var, 50)))
    focus_thresh = float(np.mean(focus_samples))
    print(f"  Focus threshold (auto): {focus_thresh:.1f}")

    # ── Compute masked deltas ────────────────────────────────────
    print("Computing deltas with focus+motion masking...")
    deltas = np.zeros((num_frames, meas_h, meas_w), dtype=np.float16)

    # Reusable buffer for Laplacian
    lap_buf = np.empty((meas_h, meas_w), dtype=np.float32)
    ksize_mask = (21, 21)
    ksize_delta = (5, 5)

    t_start = time.time()
    for t in range(num_frames):
        w0 = max(0, t - radius)
        w1 = min(num_frames, t + radius + 1)

        # Temporal median — numpy uses quickselect, O(n) per pixel
        median = np.median(pixel_data[w0:w1], axis=0)  # float64
        current_f = pixel_data[t].astype(np.float32)
        median_f = median.astype(np.float32)

        # Raw delta toward temporal median
        delta = (median_f - current_f) * strength

        # ── Focus mask: local Laplacian variance ──
        # High variance = sharp/in-focus, low = bokeh/blur
        cv2.Laplacian(pixel_data[t], cv2.CV_32F, dst=lap_buf)
        focus_var = cv2.blur(lap_buf * lap_buf, (focus_block, focus_block))
        focus_mask = np.clip(
            (focus_var - focus_thresh * 0.3) / (focus_thresh * 1.5),
            0.0, 1.0
        )

        # ── Motion mask: |current - median| ──
        # Only affect pixels that actually differ from the temporal median
        motion = np.abs(median_f - current_f)
        motion_mask = np.clip(
            (motion - motion_thresh) / float(motion_thresh),
            0.0, 1.0
        )

        # ── Combined: focus AND motion ──
        mask = focus_mask * motion_mask
        mask = cv2.GaussianBlur(mask.astype(np.float32), ksize_mask, 0)

        # Apply mask, then spatially smooth the delta
        masked_delta = cv2.GaussianBlur(
            (delta * mask).astype(np.float32), ksize_delta, 0
        )
        deltas[t] = masked_delta.astype(np.float16)

        if (t + 1) % 120 == 0:
            e = time.time() - t_start
            print(f"  {t+1}/{num_frames}  ({(t+1)/e:.0f} fps)", flush=True)

    del pixel_data
    rms = float(np.sqrt(np.mean(deltas.astype(np.float32) ** 2)))
    e = time.time() - t_start
    print(f"  Done: RMS={rms:.2f}  {e:.1f}s  ({num_frames/e:.0f} fps)")

    # ── Pass 2: Apply to full-res video ──────────────────────────
    print(f"\nPass 2: Applying at {w}x{h}...")
    full_size = w * h * 3

    decoder = subprocess.Popen([
        'ffmpeg', '-hwaccel', 'cuda', *start_args, *dur_args, '-i', input_path,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-v', 'error', '-'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}', '-r', str(info['fps']), '-i', '-',
        *start_args, *dur_args, '-i', input_path,
        '-map', '0:v', '-map', '1:a',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', str(opts['qp']),
        '-preset', 'p4',
        '-c:a', 'aac', '-b:a', '192k', '-v', 'error', output_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Pre-allocate reusable buffers — avoids ~190MB allocation per frame
    delta_full = np.empty((h, w), dtype=np.float32)
    frame_f32 = np.empty((h, w, 3), dtype=np.float32)
    result_u8 = np.empty((h, w, 3), dtype=np.uint8)

    idx = 0
    t_start = time.time()
    while True:
        raw = decoder.stdout.read(full_size)
        if len(raw) < full_size:
            break
        if idx < num_frames:
            # cv2.resize: ~50x faster than scipy.ndimage.zoom
            delta_small = deltas[idx].astype(np.float32)
            cv2.resize(delta_small, (w, h), dst=delta_full,
                       interpolation=cv2.INTER_LINEAR)

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            np.copyto(frame_f32, frame)  # uint8 → float32, reuse buffer

            # In-place add per channel — no broadcast temporary
            np.add(frame_f32[:, :, 0], delta_full, out=frame_f32[:, :, 0])
            np.add(frame_f32[:, :, 1], delta_full, out=frame_f32[:, :, 1])
            np.add(frame_f32[:, :, 2], delta_full, out=frame_f32[:, :, 2])

            np.clip(frame_f32, 0, 255, out=frame_f32)
            np.copyto(result_u8, frame_f32, casting='unsafe')
            encoder.stdin.write(result_u8.tobytes())
        else:
            encoder.stdin.write(raw)
        idx += 1
        if idx % 60 == 0:
            e = time.time() - t_start
            print(f"  {idx}/{num_frames}  ({idx/e:.1f} fps)", flush=True)

    try:
        encoder.stdin.close()
    except BrokenPipeError:
        pass
    decoder.wait()
    encoder.wait()
    print(f"Encoded {idx} frames in {time.time() - t_start:.1f}s")
    print("Done!")


if __name__ == '__main__':
    main()
