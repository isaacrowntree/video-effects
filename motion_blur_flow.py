#!/usr/bin/env python3
"""
Motion Blur Flow — Frame-trailing ghost effect for dreamy motion trails.

Stacks time-offset copies of the video with decaying opacity to create
onion-skin motion trails. Camera motion is compensated so only independently
moving, in-focus subjects get the trail effect — backgrounds stay sharp.

Usage:
    python3 motion_blur_flow.py <input> <output> [options]

Options:
    --layers=N          Number of trailing layers (default: 6)
    --decay=F           Opacity decay per layer 0.1-0.9 (default: 0.7)
    --spacing=N         Frame gap between layers (default: 5)
    --ghost=MODE        Ghost style: glow, shadow, normal (default: glow)
    --mask              Enable focus+motion mask (default: off)
    --motion-thresh=N   Motion sensitivity 1-50 (default: 10, only with --mask)
    --focus-block=N     Focus detection block size, odd (default: 31, only with --mask)
    --qp=N              Output quality (default: 10)
    --start=N           Start time in seconds
    --duration=N        Process N seconds

Ghost styles:
    glow    — Overexposed ghosts with soft bloom, screen-blended (bright trails)
    shadow  — Underexposed dark silhouette ghosts, multiply-blended
    normal  — Standard weighted average blend
"""

import subprocess
import sys
import os
import json
import time
from collections import deque
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
        'layers': 6, 'decay': 0.7, 'spacing': 5,
        'ghost': 'glow', 'mask': False,
        'motion_thresh': 10, 'focus_block': 31,
    }
    for arg in argv:
        if arg == '--mask':
            opts['mask'] = True
        elif '=' in arg:
            key, val = arg.lstrip('-').replace('-', '_').split('=', 1)
            if key in ('qp', 'layers', 'spacing', 'motion_thresh', 'focus_block'):
                opts[key] = int(val)
            elif key in ('start', 'duration', 'decay'):
                opts[key] = float(val)
            elif key == 'ghost':
                opts['ghost'] = val
    return opts


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    opts = parse_opts(sys.argv[3:])
    layers = opts['layers']
    decay = opts['decay']
    spacing = opts['spacing']
    ghost_style = opts['ghost']
    use_mask = opts['mask']
    motion_thresh = opts['motion_thresh']
    focus_block = opts['focus_block'] | 1

    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        sys.exit(1)

    info = get_video_info(input_path)
    h, w = info['height'], info['width']

    # Layer weights: [1, decay, decay^2, ...] normalized to sum=1
    weights = np.array([decay ** i for i in range(layers)], dtype=np.float32)
    weights /= weights.sum()
    buffer_size = (layers - 1) * spacing + 1

    # Measurement resolution for camera motion + masking
    meas_h = 540
    meas_w = int(w / h * meas_h)
    meas_w -= meas_w % 2
    scale_x = w / meas_w
    scale_y = h / meas_h

    print(f"Input: {input_path}")
    print(f"  {w}x{h} @ {info['fps']:.2f}fps")
    print(f"  Layers={layers}  Decay={decay}  Spacing={spacing}  Ghost={ghost_style}")
    print(f"  Weights: [{', '.join(f'{ww:.3f}' for ww in weights)}]")
    print(f"  Buffer: {buffer_size} frames ({buffer_size * w * h * 3 / 1e6:.0f} MB)")
    print(f"  Mask: {'on' if use_mask else 'off (camera compensation only)'}")

    start_args = ['-ss', str(opts['start'])] if opts['start'] else []
    dur_args = ['-t', str(opts['duration'])] if opts['duration'] else []

    # ── Pass 1: Camera motion + masks at measurement res ─────────
    print(f"\nPass 1: Camera motion at {meas_w}x{meas_h}...")
    cmd = [
        'ffmpeg', '-hwaccel', 'cuda', *start_args, *dur_args,
        '-i', input_path,
        '-vf', f'scale={meas_w}:{meas_h},format=gray',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-v', 'error', '-'
    ]
    dec = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = meas_w * meas_h
    meas_list = []
    while True:
        raw = dec.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        meas_list.append(np.frombuffer(raw, dtype=np.uint8).reshape(meas_h, meas_w))
    dec.wait()
    num_frames = len(meas_list)
    pixel_data = np.stack(meas_list)
    del meas_list
    print(f"  {num_frames} frames")

    # Camera motion via phase correlation between consecutive frames
    cumulative = np.zeros((num_frames, 2), dtype=np.float64)
    for i in range(num_frames - 1):
        (dx, dy), resp = cv2.phaseCorrelate(
            pixel_data[i].astype(np.float64),
            pixel_data[i + 1].astype(np.float64)
        )
        if resp > 0.05:
            cumulative[i + 1] = cumulative[i] + np.array([dx, dy])
        else:
            cumulative[i + 1] = cumulative[i]
    drift = np.sqrt(cumulative[-1, 0]**2 + cumulative[-1, 1]**2)
    print(f"  Camera drift: {drift:.1f}px ({drift * scale_x:.0f}px at full res)")

    # Optional mask computation
    masks = None
    if use_mask:
        focus_samples = []
        for i in range(0, min(num_frames, 120), 15):
            lap = cv2.Laplacian(pixel_data[i], cv2.CV_32F)
            local_var = cv2.blur(lap * lap, (focus_block, focus_block))
            focus_samples.append(float(np.percentile(local_var, 50)))
        focus_thresh = float(np.mean(focus_samples))
        print(f"  Focus threshold: {focus_thresh:.1f}")

        print("  Computing masks...")
        masks = np.zeros((num_frames, meas_h, meas_w), dtype=np.float32)
        lap_buf = np.empty((meas_h, meas_w), dtype=np.float32)
        t0_mask = time.time()

        for t in range(num_frames):
            current_f = pixel_data[t].astype(np.float32)
            motion_energy = np.zeros((meas_h, meas_w), dtype=np.float32)
            for layer in range(1, layers):
                fi = t - layer * spacing
                if fi < 0:
                    continue
                dx = cumulative[t, 0] - cumulative[fi, 0]
                dy = cumulative[t, 1] - cumulative[fi, 1]
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned = cv2.warpAffine(
                    pixel_data[fi].astype(np.float32), M, (meas_w, meas_h),
                    borderMode=cv2.BORDER_REFLECT
                )
                motion_energy = np.maximum(
                    motion_energy, np.abs(current_f - aligned)
                )
            motion_mask = np.clip(
                (motion_energy - motion_thresh) / float(motion_thresh), 0.0, 1.0
            )
            cv2.Laplacian(pixel_data[t], cv2.CV_32F, dst=lap_buf)
            focus_var = cv2.blur(lap_buf * lap_buf, (focus_block, focus_block))
            focus_mask = np.clip(
                (focus_var - focus_thresh * 0.3) / (focus_thresh * 1.5), 0.0, 1.0
            )
            masks[t] = cv2.GaussianBlur(
                (focus_mask * motion_mask).astype(np.float32), (21, 21), 0
            )
            if (t + 1) % 120 == 0:
                e = time.time() - t0_mask
                print(f"    {t+1}/{num_frames}  ({(t+1)/e:.0f} fps)", flush=True)

        avg_cov = (masks > 0.02).mean() * 100
        print(f"  Done: {time.time()-t0_mask:.1f}s  mask coverage: {avg_cov:.1f}%")

    del pixel_data

    # ── Pass 2: Apply trail at full resolution ───────────────────
    print(f"\nPass 2: Trail ({ghost_style}) at {w}x{h}...")
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

    # Pre-allocated reusable buffers
    frame_buf = deque(maxlen=buffer_size)
    trail_buf = np.empty((h, w, 3), dtype=np.float32)
    frame_f32 = np.empty((h, w, 3), dtype=np.float32)
    ghost_buf = np.empty((h, w, 3), dtype=np.float32)
    mask_full = np.empty((h, w), dtype=np.float32)
    result_u8 = np.empty((h, w, 3), dtype=np.uint8)

    idx = 0
    t0 = time.time()
    while True:
        raw = decoder.stdout.read(full_size)
        if len(raw) < full_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        frame_buf.append(frame)

        if idx < num_frames and len(frame_buf) >= buffer_size:
            np.copyto(frame_f32, frame)

            if ghost_style == 'normal':
                # ── Normal: weighted average ─────────────────────
                np.multiply(frame_f32, weights[0], out=trail_buf)
                for layer in range(1, layers):
                    buf_idx = len(frame_buf) - 1 - layer * spacing
                    fi = idx - layer * spacing
                    if buf_idx < 0 or fi < 0:
                        trail_buf += frame_f32 * weights[layer]
                        continue
                    dx = (cumulative[idx, 0] - cumulative[fi, 0]) * scale_x
                    dy = (cumulative[idx, 1] - cumulative[fi, 1]) * scale_y
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    aligned_u8 = cv2.warpAffine(
                        frame_buf[buf_idx], M, (w, h),
                        borderMode=cv2.BORDER_REFLECT)
                    np.copyto(ghost_buf, aligned_u8)
                    ghost_buf *= weights[layer]
                    trail_buf += ghost_buf

            elif ghost_style == 'glow':
                # ── Glow: overexposed bloom ghosts, screen blend ─
                np.copyto(trail_buf, frame_f32)
                for layer in range(1, layers):
                    buf_idx = len(frame_buf) - 1 - layer * spacing
                    fi = idx - layer * spacing
                    if buf_idx < 0 or fi < 0:
                        continue
                    dx = (cumulative[idx, 0] - cumulative[fi, 0]) * scale_x
                    dy = (cumulative[idx, 1] - cumulative[fi, 1]) * scale_y
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    aligned_u8 = cv2.warpAffine(
                        frame_buf[buf_idx], M, (w, h),
                        borderMode=cv2.BORDER_REFLECT)
                    np.copyto(ghost_buf, aligned_u8)

                    # Progressive overexposure — older ghosts glow brighter
                    ghost_buf *= 1.0 + 0.3 * layer

                    # Progressive soft blur — older ghosts are dreamier
                    ksize = max(3, layer * 4 + 1) | 1
                    for c in range(3):
                        ghost_buf[:, :, c] = cv2.GaussianBlur(
                            ghost_buf[:, :, c], (ksize, ksize), 0)

                    # Screen blend: adds light, brighter areas resist clipping
                    # trail += ghost * (255 - trail) / 255 * alpha
                    alpha = weights[layer] * 2.5
                    for c in range(3):
                        headroom = (255.0 - trail_buf[:, :, c]) / 255.0
                        trail_buf[:, :, c] += (
                            ghost_buf[:, :, c] * headroom * alpha)

            elif ghost_style == 'shadow':
                # ── Shadow: dark silhouette ghosts, multiply blend
                np.copyto(trail_buf, frame_f32)
                for layer in range(1, layers):
                    buf_idx = len(frame_buf) - 1 - layer * spacing
                    fi = idx - layer * spacing
                    if buf_idx < 0 or fi < 0:
                        continue
                    dx = (cumulative[idx, 0] - cumulative[fi, 0]) * scale_x
                    dy = (cumulative[idx, 1] - cumulative[fi, 1]) * scale_y
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    aligned_u8 = cv2.warpAffine(
                        frame_buf[buf_idx], M, (w, h),
                        borderMode=cv2.BORDER_REFLECT)
                    np.copyto(ghost_buf, aligned_u8)

                    # Underexpose ghost — darker silhouettes
                    ghost_buf *= 0.5

                    # Progressive blur
                    ksize = max(3, layer * 4 + 1) | 1
                    for c in range(3):
                        ghost_buf[:, :, c] = cv2.GaussianBlur(
                            ghost_buf[:, :, c], (ksize, ksize), 0)

                    # Multiply blend: darkens where ghost is dark
                    # trail *= lerp(1, ghost/255, alpha)
                    alpha = weights[layer] * 2.0
                    for c in range(3):
                        blend = (1.0 - alpha) + alpha * ghost_buf[:, :, c] / 255.0
                        trail_buf[:, :, c] *= blend

            # ── Optional mask ────────────────────────────────────
            if masks is not None:
                # Resize float32 mask to full res (same dtype = no silent fail)
                mask_small = masks[idx]  # already float32
                cv2.resize(mask_small, (w, h), dst=mask_full,
                           interpolation=cv2.INTER_LINEAR)
                # Blend: output = frame + (trail - frame) * mask
                trail_buf -= frame_f32
                for c in range(3):
                    trail_buf[:, :, c] *= mask_full
                trail_buf += frame_f32

            np.clip(trail_buf, 0, 255, out=trail_buf)
            np.copyto(result_u8, trail_buf, casting='unsafe')
            encoder.stdin.write(result_u8.tobytes())
        else:
            encoder.stdin.write(frame.tobytes())

        idx += 1
        if idx % 60 == 0:
            e = time.time() - t0
            print(f"  {idx}/{num_frames}  ({idx/e:.1f} fps)", flush=True)

    try:
        encoder.stdin.close()
    except BrokenPipeError:
        pass
    decoder.wait()
    encoder.wait()
    print(f"Encoded {idx} frames in {time.time()-t0:.1f}s")
    print("Done!")


if __name__ == '__main__':
    main()
