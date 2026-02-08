#!/usr/bin/env python3
"""
Test script for comparing video matting models.
Outputs alpha matte video + foreground-on-green composite for visual comparison.

Usage:
    python3 test_matting.py <input> <output_dir> --model=rvm [--duration=10]
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
        '-show_entries', 'stream=r_frame_rate,width,height',
        '-show_entries', 'format=duration', '-of', 'json', path
    ]
    data = json.loads(subprocess.check_output(cmd, stderr=subprocess.DEVNULL))
    s = data['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    return {'fps': fps, 'width': int(s['width']), 'height': int(s['height'])}


def test_rvm(input_path, output_dir, duration=None, resolution=1080):
    """Test Robust Video Matting."""
    import torch
    from torchvision.transforms.functional import to_tensor

    info = get_video_info(input_path)
    w, h, fps = info['width'], info['height'], info['fps']

    # Scale to target resolution for processing
    if h > resolution:
        proc_h = resolution
        proc_w = int(w / h * proc_h)
        proc_w -= proc_w % 2
    else:
        proc_h, proc_w = h, w

    # Downsample ratio for RVM internal processing
    downsample = 0.25 if proc_h >= 1080 else 0.5

    print(f"RVM Test:")
    print(f"  Input: {w}x{h} @ {fps:.2f}fps")
    print(f"  Processing at: {proc_w}x{proc_h}")
    print(f"  Downsample ratio: {downsample}")

    # Load model via PyTorch Hub
    print("  Loading model...")
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").cuda().eval()
    print("  Model loaded (ResNet50)")

    # Check VRAM usage
    alloc = torch.cuda.memory_allocated() / 1e6
    print(f"  VRAM after load: {alloc:.0f} MB")

    # Decode input
    dur_args = ['-t', str(duration)] if duration else []
    decoder = subprocess.Popen([
        'ffmpeg', '-hwaccel', 'cuda', *dur_args, '-i', input_path,
        '-vf', f'scale={proc_w}:{proc_h}',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-v', 'error', '-'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Encode: alpha matte + green-screen composite
    alpha_path = os.path.join(output_dir, 'rvm_alpha.mp4')
    comp_path = os.path.join(output_dir, 'rvm_composite.mp4')

    alpha_enc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
        '-s', f'{proc_w}x{proc_h}', '-r', str(fps), '-i', '-',
        '-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', '18',
        '-v', 'error', alpha_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    comp_enc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{proc_w}x{proc_h}', '-r', str(fps), '-i', '-',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', '14',
        '-v', 'error', comp_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = proc_w * proc_h * 3
    rec = [None] * 4  # RVM recurrent states
    idx = 0
    t0 = time.time()
    green_bg = np.full((proc_h, proc_w, 3), [0, 177, 64], dtype=np.uint8)

    with torch.no_grad():
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3)

            # Convert to tensor [1, 3, H, W] float32 0-1
            src = to_tensor(frame).unsqueeze(0).cuda()

            # Run RVM
            fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample)

            # Alpha to numpy uint8
            alpha = (pha[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            alpha_enc.stdin.write(alpha.tobytes())

            # Composite on green screen
            alpha_f = alpha.astype(np.float32) / 255.0
            comp = np.empty_like(frame)
            for c in range(3):
                comp[:, :, c] = (frame[:, :, c] * alpha_f +
                                 green_bg[:, :, c] * (1 - alpha_f)).astype(np.uint8)
            comp_enc.stdin.write(comp.tobytes())

            idx += 1
            if idx % 60 == 0:
                e = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 1e6
                print(f"    {idx} frames  ({idx/e:.1f} fps)  peak VRAM: {peak:.0f} MB",
                      flush=True)

    for enc in [alpha_enc, comp_enc]:
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
    decoder.wait()
    alpha_enc.wait()
    comp_enc.wait()

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"  Done: {idx} frames in {elapsed:.1f}s ({idx/elapsed:.1f} fps)")
    print(f"  Peak VRAM: {peak:.0f} MB")
    print(f"  Alpha: {alpha_path}")
    print(f"  Composite: {comp_path}")

    # Cleanup GPU memory
    del model, rec, src
    torch.cuda.empty_cache()


def test_rembg(input_path, output_dir, duration=None, resolution=1080):
    """Test rembg with BiRefNet backend."""
    from rembg import new_session, remove

    info = get_video_info(input_path)
    w, h, fps = info['width'], info['height'], info['fps']

    if h > resolution:
        proc_h = resolution
        proc_w = int(w / h * proc_h)
        proc_w -= proc_w % 2
    else:
        proc_h, proc_w = h, w

    print(f"rembg Test:")
    print(f"  Input: {w}x{h} @ {fps:.2f}fps")
    print(f"  Processing at: {proc_w}x{proc_h}")

    print("  Loading model (birefnet-general)...")
    session = new_session("birefnet-general")
    print("  Model loaded")

    dur_args = ['-t', str(duration)] if duration else []
    decoder = subprocess.Popen([
        'ffmpeg', '-hwaccel', 'cuda', *dur_args, '-i', input_path,
        '-vf', f'scale={proc_w}:{proc_h}',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-v', 'error', '-'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    alpha_path = os.path.join(output_dir, 'rembg_alpha.mp4')
    comp_path = os.path.join(output_dir, 'rembg_composite.mp4')

    alpha_enc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
        '-s', f'{proc_w}x{proc_h}', '-r', str(fps), '-i', '-',
        '-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', '18',
        '-v', 'error', alpha_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    comp_enc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{proc_w}x{proc_h}', '-r', str(fps), '-i', '-',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', '14',
        '-v', 'error', comp_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = proc_w * proc_h * 3
    idx = 0
    t0 = time.time()
    green_bg = np.full((proc_h, proc_w, 3), [0, 177, 64], dtype=np.uint8)

    while True:
        raw = decoder.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3)

        # rembg returns RGBA
        result = remove(frame, session=session, only_mask=True)
        alpha = result if result.ndim == 2 else result[:, :, 0]
        alpha_enc.stdin.write(alpha.tobytes())

        # Composite
        alpha_f = alpha.astype(np.float32) / 255.0
        comp = np.empty_like(frame)
        for c in range(3):
            comp[:, :, c] = (frame[:, :, c] * alpha_f +
                             green_bg[:, :, c] * (1 - alpha_f)).astype(np.uint8)
        comp_enc.stdin.write(comp.tobytes())

        idx += 1
        if idx % 30 == 0:
            e = time.time() - t0
            print(f"    {idx} frames  ({idx/e:.1f} fps)", flush=True)

    for enc in [alpha_enc, comp_enc]:
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
    decoder.wait()
    alpha_enc.wait()
    comp_enc.wait()

    elapsed = time.time() - t0
    print(f"  Done: {idx} frames in {elapsed:.1f}s ({idx/elapsed:.1f} fps)")
    print(f"  Alpha: {alpha_path}")
    print(f"  Composite: {comp_path}")


def test_rmbg2(input_path, output_dir, duration=None, resolution=1080):
    """Test BRIA RMBG 2.0."""
    import torch
    from transformers import pipeline
    from PIL import Image

    info = get_video_info(input_path)
    w, h, fps = info['width'], info['height'], info['fps']

    if h > resolution:
        proc_h = resolution
        proc_w = int(w / h * proc_h)
        proc_w -= proc_w % 2
    else:
        proc_h, proc_w = h, w

    print(f"RMBG 2.0 Test:")
    print(f"  Input: {w}x{h} @ {fps:.2f}fps")
    print(f"  Processing at: {proc_w}x{proc_h}")

    print("  Loading model...")
    pipe = pipeline("image-segmentation", model="briaai/RMBG-2.0",
                    trust_remote_code=True, device="cuda")
    print("  Model loaded")

    dur_args = ['-t', str(duration)] if duration else []
    decoder = subprocess.Popen([
        'ffmpeg', '-hwaccel', 'cuda', *dur_args, '-i', input_path,
        '-vf', f'scale={proc_w}:{proc_h}',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-v', 'error', '-'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    alpha_path = os.path.join(output_dir, 'rmbg2_alpha.mp4')
    comp_path = os.path.join(output_dir, 'rmbg2_composite.mp4')

    alpha_enc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
        '-s', f'{proc_w}x{proc_h}', '-r', str(fps), '-i', '-',
        '-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', '18',
        '-v', 'error', alpha_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    comp_enc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{proc_w}x{proc_h}', '-r', str(fps), '-i', '-',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', '14',
        '-v', 'error', comp_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = proc_w * proc_h * 3
    idx = 0
    t0 = time.time()
    green_bg = np.full((proc_h, proc_w, 3), [0, 177, 64], dtype=np.uint8)

    with torch.no_grad():
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3)
            pil_img = Image.fromarray(frame)

            # RMBG returns list of dicts with 'mask' key
            result = pipe(pil_img)
            mask_pil = result[0]['mask'] if isinstance(result, list) else result['mask']
            alpha = np.array(mask_pil.resize((proc_w, proc_h)))
            if alpha.ndim == 3:
                alpha = alpha[:, :, 0]

            alpha_enc.stdin.write(alpha.tobytes())

            alpha_f = alpha.astype(np.float32) / 255.0
            comp = np.empty_like(frame)
            for c in range(3):
                comp[:, :, c] = (frame[:, :, c] * alpha_f +
                                 green_bg[:, :, c] * (1 - alpha_f)).astype(np.uint8)
            comp_enc.stdin.write(comp.tobytes())

            idx += 1
            if idx % 30 == 0:
                e = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 1e6
                print(f"    {idx} frames  ({idx/e:.1f} fps)  VRAM: {peak:.0f} MB",
                      flush=True)

    for enc in [alpha_enc, comp_enc]:
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
    decoder.wait()
    alpha_enc.wait()
    comp_enc.wait()

    elapsed = time.time() - t0
    print(f"  Done: {idx} frames in {elapsed:.1f}s ({idx/elapsed:.1f} fps)")
    print(f"  Alpha: {alpha_path}")
    print(f"  Composite: {comp_path}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    opts = {}
    for arg in sys.argv[3:]:
        if '=' in arg:
            k, v = arg.lstrip('-').replace('-', '_').split('=', 1)
            opts[k] = v

    model = opts.get('model', 'rvm')
    duration = float(opts['duration']) if 'duration' in opts else None
    resolution = int(opts.get('resolution', '1080'))

    if model == 'rvm':
        test_rvm(input_path, output_dir, duration, resolution)
    elif model == 'rembg':
        test_rembg(input_path, output_dir, duration, resolution)
    elif model == 'rmbg2':
        test_rmbg2(input_path, output_dir, duration, resolution)
    else:
        print(f"Unknown model: {model}")
        print("Available: rvm, rembg, rmbg2")
        sys.exit(1)
