#!/usr/bin/env python3
"""
Background generation from video + alpha matte.

Modes:
    plate    — Build a clean background plate from temporal median of visible
               background pixels, then inpaint remaining holes.
    generate — Use Stable Diffusion inpainting to generate a new background
               from a text prompt.
    composite — Combine a background plate + original video + alpha matte
                to produce a final video with only the masked subject.
    studio   — Replace background with AI-generated scene (e.g. neon dance
               studio). Adds depth-based parallax, exposure tracking, and
               light-aware prompt augmentation.

Usage:
    # Clean plate using a pre-computed alpha matte
    python3 bg_generate.py plate input.MP4 alpha.mp4 output_bg.png [options]

    # Clean plate with auto-masking (segments ALL people per-frame)
    python3 bg_generate.py plate input.MP4 auto output_bg.png [options]

    # Text-prompted background generation
    python3 bg_generate.py generate input.MP4 alpha.mp4 output_bg.png --prompt="..."

    # Composite: dancer on clean plate (removes background people)
    python3 bg_generate.py composite input.MP4 dancer_alpha.mp4 output.MP4 --bg=plate.png

    # Studio background replacement with parallax
    python3 bg_generate.py studio input.MP4 alpha.mp4 output.MP4 --prompt="neon dance studio"

Options:
    --resolution=N      Processing resolution height (default: 1080)
    --samples=N         Max frames to sample for plate mode (default: 200)
    --bg-thresh=F       Alpha below which pixel = background (default: 0.3)
    --inpaint=METHOD    opencv or lama (default: opencv)
    --duration=N        Process N seconds
    --start=N           Start time in seconds
    --prompt=TEXT       Background description (generate/studio mode)
    --strength=F        SD inpainting strength (default: 0.95)
    --steps=N           SD inference steps (default: 30)
    --seed=N            RNG seed (generate/studio mode)
    --bg=PATH           Background plate image (composite mode)
    --qp=N              Output video quality (composite mode, default: 14)
    --depth-scale=F     Parallax magnitude multiplier (studio, default: 2.0)
    --bg-margin=F       Overscan factor for BG generation (studio, default: 1.3)
    --exposure-smooth=N Temporal smoothing window in frames (studio, default: 15)
    --sd-model=MODEL    SD model: sd15 or sdxl (studio, default: sd15)
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
        '-show_entries', 'stream=r_frame_rate,width,height,nb_frames',
        '-show_entries', 'format=duration', '-of', 'json', path
    ]
    data = json.loads(subprocess.check_output(cmd, stderr=subprocess.DEVNULL))
    s = data['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    duration = float(data['format'].get('duration', 0))
    nb = int(s.get('nb_frames', 0) or 0)
    if nb == 0 and duration > 0:
        nb = int(duration * fps)
    return {
        'fps': fps, 'width': int(s['width']), 'height': int(s['height']),
        'duration': duration, 'nb_frames': nb,
    }


def parse_opts(argv):
    opts = {
        'prompt': None, 'duration': None, 'start': None,
        'resolution': 1080, 'samples': 200,
        'bg_thresh': 0.3, 'inpaint': 'opencv',
        'strength': 0.95, 'steps': 30, 'seed': None,
        'bg': None, 'qp': 14,
        # Studio mode options
        'depth_scale': 0.3, 'bg_margin': 0,  # 0 = auto-compute from camera drift
        'exposure_smooth': 15, 'sd_model': 'sd15',
    }
    for arg in argv:
        if '=' in arg:
            key, val = arg.lstrip('-').replace('-', '_').split('=', 1)
            if key in ('resolution', 'samples', 'steps', 'seed', 'qp',
                       'exposure_smooth'):
                opts[key] = int(val)
            elif key in ('duration', 'start', 'bg_thresh', 'strength',
                         'depth_scale', 'bg_margin'):
                opts[key] = float(val)
            elif key in ('prompt', 'inpaint', 'bg', 'sd_model'):
                opts[key] = val
    return opts


def decode_frames(path, proc_w, proc_h, pix_fmt, channels, start=None, duration=None):
    """Generator that yields decoded frames from a video file."""
    start_args = ['-ss', str(start)] if start else []
    dur_args = ['-t', str(duration)] if duration else []
    cmd = [
        'ffmpeg', '-hwaccel', 'cuda', *start_args, *dur_args,
        '-i', path,
        '-vf', f'scale={proc_w}:{proc_h}',
        '-f', 'rawvideo', '-pix_fmt', pix_fmt, '-v', 'error', '-'
    ]
    dec = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = proc_w * proc_h * channels
    while True:
        raw = dec.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        if channels == 1:
            yield np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w)
        else:
            yield np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, channels)
    dec.wait()


def create_auto_masker():
    """Load DeepLabV3 (COCO person class) on CUDA for person segmentation."""
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
    print("  Loading DeepLabV3 (person segmentation, CUDA)...")
    weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet101(weights=weights).cuda().eval()
    preprocess = weights.transforms()
    alloc = torch.cuda.memory_allocated() / 1e6
    print(f"  Loaded (VRAM: {alloc:.0f} MB)")
    return (model, preprocess)


def auto_mask_frame(frame_rgb, masker):
    """Detect all person pixels in a frame. Returns uint8 mask (255=person)."""
    import torch
    model, preprocess = masker
    h, w = frame_rgb.shape[:2]

    # DeepLabV3 expects PIL or tensor input via transforms
    from PIL import Image
    pil_img = Image.fromarray(frame_rgb)
    inp = preprocess(pil_img).unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(inp)['out']  # (1, 21, H, W) logits
        pred = out.argmax(dim=1)[0]  # (H, W) class indices

    # Class 15 = person in COCO/VOC
    person_mask = (pred == 15).cpu().numpy().astype(np.uint8) * 255

    # Resize to original frame size if needed
    if person_mask.shape[:2] != (h, w):
        person_mask = cv2.resize(person_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return person_mask


def compute_homographies(gray_frames, alpha_frames, bg_thresh):
    """Compute homography from each frame to the reference (middle) frame.

    Uses ORB feature matching on background pixels. Falls back to identity
    for frames with too few matches.
    """
    n = len(gray_frames)
    ref = n // 2
    homographies = [np.eye(3, dtype=np.float64)] * n

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Mask foreground out of feature detection
    ref_det_mask = (alpha_frames[ref] < int(bg_thresh * 255)).astype(np.uint8) * 255
    ref_det_mask = cv2.erode(ref_det_mask, None, iterations=5)

    ref_kp, ref_desc = orb.detectAndCompute(gray_frames[ref], ref_det_mask)
    if ref_desc is None or len(ref_kp) < 10:
        ref_kp, ref_desc = orb.detectAndCompute(gray_frames[ref], None)

    good_count = 0
    fallback_count = 0

    for i in range(n):
        if i == ref:
            good_count += 1
            continue

        det_mask = (alpha_frames[i] < int(bg_thresh * 255)).astype(np.uint8) * 255
        det_mask = cv2.erode(det_mask, None, iterations=5)

        kp, desc = orb.detectAndCompute(gray_frames[i], det_mask)
        if desc is None or len(kp) < 10:
            kp, desc = orb.detectAndCompute(gray_frames[i], None)

        if desc is None or ref_desc is None:
            fallback_count += 1
            continue

        matches = bf.knnMatch(desc, ref_desc, k=2)
        good = []
        for m_pair in matches:
            if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
                good.append(m_pair[0])

        if len(good) < 8:
            fallback_count += 1
            continue

        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if H is not None and mask.ravel().sum() >= 6:
            homographies[i] = H
            good_count += 1
        else:
            fallback_count += 1

    return homographies, good_count, fallback_count


def build_clean_plate(input_path, alpha_path, opts):
    """Build a clean background plate from temporal median of camera-aligned
    background pixels."""
    info = get_video_info(input_path)
    w, h = info['width'], info['height']
    resolution = opts['resolution']

    if h > resolution:
        proc_h = resolution
        proc_w = int(w / h * proc_h)
        proc_w -= proc_w % 2
    else:
        proc_h, proc_w = h, w

    bg_thresh = opts['bg_thresh']
    max_samples = opts['samples']
    use_auto_mask = (alpha_path == 'auto')

    # Total frame count
    total_frames = info['nb_frames']
    if opts['duration']:
        total_frames = min(total_frames, int(opts['duration'] * info['fps']))
    if total_frames <= 0:
        total_frames = int(info['duration'] * info['fps'])

    # Evenly spaced sample indices
    if total_frames <= max_samples:
        sample_indices = set(range(total_frames))
    else:
        sample_indices = set(
            int(i * total_frames / max_samples) for i in range(max_samples)
        )

    print(f"Clean plate:")
    print(f"  Input: {w}x{h} @ {info['fps']:.2f}fps")
    print(f"  Processing at: {proc_w}x{proc_h}")
    print(f"  Total frames: {total_frames}, sampling {len(sample_indices)}")
    print(f"  Mask: {'auto (rembg)' if use_auto_mask else alpha_path}")
    print(f"  BG threshold: {bg_thresh}")

    # Set up auto-masking model if needed
    auto_masker = None
    if use_auto_mask:
        auto_masker = create_auto_masker()

    # Decode and collect sampled frames
    print("\nDecoding frames...")
    rgb_frames = []
    gray_frames = []
    alpha_frames = []
    idx = 0
    t0 = time.time()

    rgb_gen = decode_frames(input_path, proc_w, proc_h, 'rgb24', 3,
                            opts['start'], opts['duration'])

    if use_auto_mask:
        for rgb_frame in rgb_gen:
            if idx in sample_indices:
                frame = rgb_frame.copy()
                rgb_frames.append(frame)
                gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
                mask = auto_mask_frame(frame, auto_masker)
                alpha_frames.append(mask)
            idx += 1
            if idx % 30 == 0:
                e = time.time() - t0
                print(f"  {idx}/{total_frames} ({idx/e:.1f} fps), "
                      f"collected {len(rgb_frames)}", flush=True)
        # Cleanup model
        del auto_masker
        import torch
        torch.cuda.empty_cache()
    else:
        alpha_gen = decode_frames(alpha_path, proc_w, proc_h, 'gray', 1,
                                  opts['start'], opts['duration'])
        for rgb_frame, alpha_frame in zip(rgb_gen, alpha_gen):
            if idx in sample_indices:
                rgb_frames.append(rgb_frame.copy())
                gray_frames.append(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY))
                alpha_frames.append(alpha_frame.copy())
            idx += 1
            if idx % 120 == 0:
                e = time.time() - t0
                print(f"  {idx}/{total_frames} ({idx/e:.0f} fps), "
                      f"collected {len(rgb_frames)}", flush=True)

    n_collected = len(rgb_frames)
    print(f"  Collected {n_collected} frames in {time.time()-t0:.1f}s")

    if n_collected == 0:
        print("ERROR: No frames decoded. Check input paths.")
        sys.exit(1)

    # Camera alignment via homography
    print("\nComputing camera alignment (ORB + homography)...")
    t0 = time.time()
    homographies, good, fallback = compute_homographies(
        gray_frames, alpha_frames, bg_thresh)
    del gray_frames
    print(f"  Aligned: {good}/{n_collected}, fallback: {fallback}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # Warp all frames to the reference
    print("\nWarping frames to reference...")
    t0 = time.time()
    aligned_rgb = np.empty((n_collected, proc_h, proc_w, 3), dtype=np.uint8)
    aligned_alpha = np.empty((n_collected, proc_h, proc_w), dtype=np.uint8)

    for i in range(n_collected):
        H = homographies[i]
        if np.allclose(H, np.eye(3)):
            aligned_rgb[i] = rgb_frames[i]
            aligned_alpha[i] = alpha_frames[i]
        else:
            aligned_rgb[i] = cv2.warpPerspective(
                rgb_frames[i], H, (proc_w, proc_h),
                borderMode=cv2.BORDER_REFLECT)
            aligned_alpha[i] = cv2.warpPerspective(
                alpha_frames[i], H, (proc_w, proc_h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    del rgb_frames, alpha_frames
    print(f"  Warped {n_collected} frames in {time.time()-t0:.1f}s")

    # Build per-pixel background median
    print("\nComputing background plate...")
    t0 = time.time()

    alpha_thresh_int = int(bg_thresh * 255)
    bg_mask = aligned_alpha < alpha_thresh_int  # (N, H, W) bool

    bg_count = bg_mask.sum(axis=0)  # (H, W)

    plate = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)
    hole_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)

    min_samples = max(3, n_collected // 20)
    has_bg = bg_count >= min_samples

    # Fast path: pixels where ALL frames are background
    all_bg = bg_count == n_collected
    if all_bg.any():
        full_median = np.median(aligned_rgb, axis=0).astype(np.uint8)
        plate[all_bg] = full_median[all_bg]

    # Slow path: partial background pixels — masked median per pixel
    partial_bg = has_bg & ~all_bg
    if partial_bg.any():
        tile_s = 64
        for y0 in range(0, proc_h, tile_s):
            y1 = min(y0 + tile_s, proc_h)
            for x0 in range(0, proc_w, tile_s):
                x1 = min(x0 + tile_s, proc_w)
                tile_partial = partial_bg[y0:y1, x0:x1]
                if not tile_partial.any():
                    continue
                tile_rgb = aligned_rgb[:, y0:y1, x0:x1, :]
                tile_bmask = bg_mask[:, y0:y1, x0:x1]
                th, tw = y1 - y0, x1 - x0
                for dy in range(th):
                    for dx in range(tw):
                        if not tile_partial[dy, dx]:
                            continue
                        pm = tile_bmask[:, dy, dx]
                        plate[y0+dy, x0+dx] = np.median(
                            tile_rgb[pm, dy, dx, :], axis=0
                        ).astype(np.uint8)

    hole_mask[~has_bg] = 255

    # Detect transient objects: high deviation from median = moving unmasked objects
    print("  Detecting transients...")
    plate_f = plate.astype(np.float32)
    dev_sum = np.zeros((proc_h, proc_w), dtype=np.float64)
    dev_count = np.zeros((proc_h, proc_w), dtype=np.int32)

    for i in range(n_collected):
        frame_bg = bg_mask[i]
        diff = np.abs(aligned_rgb[i].astype(np.float32) - plate_f).mean(axis=2)
        dev_sum[frame_bg] += diff[frame_bg]
        dev_count[frame_bg] += 1

    safe = dev_count > 0
    mean_dev = np.zeros((proc_h, proc_w), dtype=np.float32)
    mean_dev[safe] = (dev_sum[safe] / dev_count[safe]).astype(np.float32)

    bg_dev_vals = mean_dev[has_bg & (mean_dev > 0)]
    transient_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
    if len(bg_dev_vals) > 0:
        dev_median = float(np.median(bg_dev_vals))
        dev_thresh = max(dev_median * 2.5, 15.0)
        transient = has_bg & (mean_dev > dev_thresh)
        trans_raw = transient.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        trans_raw = cv2.morphologyEx(trans_raw, cv2.MORPH_CLOSE, kernel)
        trans_raw = cv2.morphologyEx(trans_raw, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        transient_mask = trans_raw
        trans_pct = (transient_mask > 0).sum() / (proc_h * proc_w) * 100
        print(f"  Transients: {trans_pct:.1f}% (thresh={dev_thresh:.0f}, median_dev={dev_median:.0f})")

    combined_mask = np.maximum(hole_mask, transient_mask)
    del aligned_rgb, aligned_alpha, bg_mask, plate_f

    hole_pct = (hole_mask > 0).sum() / (proc_h * proc_w) * 100
    combined_pct = (combined_mask > 0).sum() / (proc_h * proc_w) * 100
    print(f"  Background coverage: {100-hole_pct:.1f}%")
    print(f"  Total inpaint area: {combined_pct:.1f}%")
    print(f"  Done in {time.time()-t0:.1f}s")

    if combined_pct > 0:
        plate = inpaint_holes(plate, combined_mask, opts['inpaint'])

    return plate, proc_w, proc_h


def inpaint_holes(plate, mask, method):
    """Fill holes using inpainting."""
    if mask.sum() == 0:
        return plate

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask, kernel, iterations=2)

    if method == 'lama':
        print("\nInpainting with LaMa...")
        try:
            from simple_lama_inpainting import SimpleLama
            from PIL import Image
            lama = SimpleLama()
            result = lama(Image.fromarray(plate), Image.fromarray(dilated))
            plate = np.array(result)
            print("  Done")
        except ImportError:
            print("  simple-lama-inpainting not installed, falling back to OpenCV")
            print("  Install: pip install simple-lama-inpainting")
            plate = cv2.inpaint(plate, dilated, inpaintRadius=12, flags=cv2.INPAINT_NS)
    else:
        print("\nInpainting with OpenCV (Navier-Stokes)...")
        plate = cv2.inpaint(plate, dilated, inpaintRadius=12, flags=cv2.INPAINT_NS)
        print("  Done")

    return plate


def generate_background(input_path, alpha_path, opts):
    """Generate a new background using Stable Diffusion inpainting.

    Uses a reference frame from the video + alpha matte to create an inpainting
    mask. SD fills in behind the subject. With --prompt, generates a described
    scene; without, attempts to reconstruct the original background.
    """
    import torch
    from PIL import Image

    try:
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError:
        print("ERROR: diffusers not installed.")
        print("Install: pip install diffusers accelerate")
        sys.exit(1)

    prompt = opts['prompt'] or "background, same scene, no people"

    info = get_video_info(input_path)
    w, h = info['width'], info['height']
    resolution = opts['resolution']

    if h > resolution:
        proc_h = resolution
        proc_w = int(w / h * proc_h)
        proc_w -= proc_w % 2
    else:
        proc_h, proc_w = h, w

    # SD needs dimensions divisible by 8
    sd_h = (proc_h // 8) * 8
    sd_w = (proc_w // 8) * 8

    print(f"Background generation:")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Processing at: {sd_w}x{sd_h}")

    # Get reference frame from the middle of the video
    dur = opts['duration'] or info['duration']
    mid_time = dur / 2
    if opts['start']:
        mid_time += opts['start']

    print(f"\nExtracting reference frame at t={mid_time:.1f}s...")
    rgb_gen = decode_frames(input_path, sd_w, sd_h, 'rgb24', 3,
                            start=mid_time, duration=0.1)
    ref_frame = next(rgb_gen, None)
    if ref_frame is None:
        print("ERROR: Could not extract reference frame.")
        sys.exit(1)

    # Get corresponding alpha
    use_auto_mask = (alpha_path == 'auto')
    if use_auto_mask:
        auto_masker = create_auto_masker()
        ref_alpha = auto_mask_frame(ref_frame, auto_masker)
        del auto_masker
        torch.cuda.empty_cache()
    else:
        alpha_gen = decode_frames(alpha_path, sd_w, sd_h, 'gray', 1,
                                  start=mid_time, duration=0.1)
        ref_alpha = next(alpha_gen, None)
        if ref_alpha is None:
            print("ERROR: Could not extract reference alpha.")
            sys.exit(1)

    # Build inpainting mask: foreground (subject) = white = area to inpaint
    fg_mask = (ref_alpha > int(opts['bg_thresh'] * 255)).astype(np.uint8) * 255
    # Dilate to ensure full coverage around subject edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)

    fg_pct = (fg_mask > 0).sum() / (sd_h * sd_w) * 100
    print(f"  Inpaint mask covers {fg_pct:.1f}% of frame")

    init_image = Image.fromarray(ref_frame)
    mask_image = Image.fromarray(fg_mask)

    # Save debug images
    out_dir = os.path.dirname(opts.get('_output_path', '')) or '.'
    debug_ref = os.path.join(out_dir, 'gen_ref_frame.png')
    debug_mask = os.path.join(out_dir, 'gen_mask.png')
    cv2.imwrite(debug_ref, cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(debug_mask, fg_mask)
    print(f"  Ref frame: {debug_ref}")
    print(f"  Mask: {debug_mask}")

    print("\nLoading SD inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")

    generator = None
    if opts['seed'] is not None:
        generator = torch.Generator(device="cuda").manual_seed(opts['seed'])

    alloc = torch.cuda.memory_allocated() / 1e6
    print(f"  VRAM: {alloc:.0f} MB")
    print(f"\nGenerating (steps={opts['steps']}, strength={opts['strength']})...")
    t0 = time.time()
    result = pipe(
        prompt=prompt, image=init_image, mask_image=mask_image,
        height=sd_h, width=sd_w,
        num_inference_steps=opts['steps'], strength=opts['strength'],
        generator=generator,
    ).images[0]

    result_np = np.array(result)
    print(f"  Generated in {time.time()-t0:.1f}s")

    del pipe
    torch.cuda.empty_cache()

    if sd_w != proc_w or sd_h != proc_h:
        result_np = cv2.resize(result_np, (proc_w, proc_h))

    return result_np, proc_w, proc_h


def composite_video(input_path, alpha_path, output_path, opts):
    """Composite foreground (from alpha) onto a background plate image."""
    bg_path = opts['bg']
    if not bg_path or not os.path.exists(bg_path):
        print(f"ERROR: --bg=<plate.png> required for composite mode.")
        sys.exit(1)

    info = get_video_info(input_path)
    w, h, fps = info['width'], info['height'], info['fps']
    resolution = opts['resolution']

    if h > resolution:
        proc_h = resolution
        proc_w = int(w / h * proc_h)
        proc_w -= proc_w % 2
    else:
        proc_h, proc_w = h, w

    # Load background plate and resize to processing resolution
    bg_bgr = cv2.imread(bg_path)
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    bg_rgb = cv2.resize(bg_rgb, (proc_w, proc_h))
    bg_f = bg_rgb.astype(np.float32)

    print(f"Composite:")
    print(f"  Input: {w}x{h} @ {fps:.2f}fps")
    print(f"  Output: {proc_w}x{proc_h}")
    print(f"  Background: {bg_path}")

    start_args = ['-ss', str(opts['start'])] if opts['start'] else []
    dur_args = ['-t', str(opts['duration'])] if opts['duration'] else []

    # Decode input video
    decoder = subprocess.Popen([
        'ffmpeg', '-hwaccel', 'cuda', *start_args, *dur_args,
        '-i', input_path,
        '-vf', f'scale={proc_w}:{proc_h}',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-v', 'error', '-'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Decode alpha
    alpha_dec = subprocess.Popen([
        'ffmpeg', '-hwaccel', 'cuda', *start_args, *dur_args,
        '-i', alpha_path,
        '-vf', f'scale={proc_w}:{proc_h}',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-v', 'error', '-'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Encode output — copy audio from input
    qp = opts['qp']
    encoder = subprocess.Popen([
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{proc_w}x{proc_h}', '-r', str(fps), '-i', '-',
        *start_args, *dur_args, '-i', input_path,
        '-map', '0:v', '-map', '1:a?',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', str(qp),
        '-c:a', 'aac', '-b:a', '192k', '-v', 'error', output_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    rgb_bytes = proc_w * proc_h * 3
    alpha_bytes = proc_w * proc_h
    frame_f = np.empty((proc_h, proc_w, 3), dtype=np.float32)
    result = np.empty((proc_h, proc_w, 3), dtype=np.uint8)

    idx = 0
    t0 = time.time()
    while True:
        raw_rgb = decoder.stdout.read(rgb_bytes)
        raw_alpha = alpha_dec.stdout.read(alpha_bytes)
        if len(raw_rgb) < rgb_bytes or len(raw_alpha) < alpha_bytes:
            break

        frame = np.frombuffer(raw_rgb, dtype=np.uint8).reshape(proc_h, proc_w, 3)
        alpha = np.frombuffer(raw_alpha, dtype=np.uint8).reshape(proc_h, proc_w)
        alpha_f = alpha.astype(np.float32) / 255.0

        # composite = fg * alpha + bg * (1 - alpha)
        np.copyto(frame_f, frame)
        for c in range(3):
            frame_f[:, :, c] = frame_f[:, :, c] * alpha_f + bg_f[:, :, c] * (1.0 - alpha_f)

        np.clip(frame_f, 0, 255, out=frame_f)
        np.copyto(result, frame_f, casting='unsafe')
        encoder.stdin.write(result.tobytes())

        idx += 1
        if idx % 60 == 0:
            e = time.time() - t0
            print(f"  {idx} frames ({idx/e:.1f} fps)", flush=True)

    try:
        encoder.stdin.close()
    except BrokenPipeError:
        pass
    decoder.wait()
    alpha_dec.wait()
    encoder.wait()

    print(f"  Encoded {idx} frames in {time.time()-t0:.1f}s")
    print(f"  Output: {output_path}")


def analyze_video(input_path, alpha_path, proc_w, proc_h, opts):
    """Single-pass analysis at 360p: camera motion, exposure, light direction.

    Returns (camera_info, exposure_ratios, light_info) where:
      camera_info = {'cumulative': ndarray(N,2), 'meas_scale': (sx,sy), 'num_frames': N}
      exposure_ratios = ndarray(N) of per-frame brightness multipliers
      light_info = {'quadrant': str, 'dx': float, 'dy': float}
    """
    from scipy.ndimage import uniform_filter1d

    info = get_video_info(input_path)

    # Measurement resolution: 360p
    meas_h = 360
    meas_w = int(info['width'] / info['height'] * meas_h)
    meas_w -= meas_w % 2
    scale_x = proc_w / meas_w
    scale_y = proc_h / meas_h

    print(f"\nAnalyzing video at {meas_w}x{meas_h}...")
    t0 = time.time()

    rgb_gen = decode_frames(input_path, meas_w, meas_h, 'rgb24', 3,
                            opts['start'], opts['duration'])
    alpha_gen = decode_frames(alpha_path, meas_w, meas_h, 'gray', 1,
                              opts['start'], opts['duration'])

    prev_gray = None
    cumulative = []
    luminances = []
    light_samples = []  # (bright_cx - geo_cx, bright_cy - geo_cy)
    bottom_alpha_vals = []  # track whether feet are cropped
    idx = 0

    for rgb_frame, alpha_frame in zip(rgb_gen, alpha_gen):
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        # Camera motion: phase correlation
        if prev_gray is None:
            cumulative.append(np.array([0.0, 0.0]))
        else:
            (dx, dy), resp = cv2.phaseCorrelate(
                prev_gray.astype(np.float64),
                gray.astype(np.float64)
            )
            if resp > 0.05:
                cumulative.append(cumulative[-1] + np.array([dx, dy]))
            else:
                cumulative.append(cumulative[-1].copy())

        # Foreground exposure: mean luminance where alpha > 128
        fg_mask = alpha_frame > 128
        if fg_mask.any():
            luminances.append(float(gray[fg_mask].mean()))
        else:
            luminances.append(luminances[-1] if luminances else 128.0)

        # Camera height: does alpha extend to bottom of frame? (feet cropped)
        bottom_rows = alpha_frame[-20:, :]
        bottom_alpha_vals.append(float(bottom_rows.mean()))

        # Light direction: sample every ~15 frames
        if idx % 15 == 0 and fg_mask.any():
            ys, xs = np.where(fg_mask)
            geo_cx, geo_cy = xs.mean(), ys.mean()
            # Brightness-weighted centroid
            bright = gray[fg_mask].astype(np.float64)
            total_b = bright.sum()
            if total_b > 0:
                bright_cx = (xs.astype(np.float64) * bright).sum() / total_b
                bright_cy = (ys.astype(np.float64) * bright).sum() / total_b
                light_samples.append((bright_cx - geo_cx, bright_cy - geo_cy))

        prev_gray = gray
        idx += 1
        if idx % 120 == 0:
            print(f"  {idx} frames...", flush=True)

    num_frames = idx
    cumulative = np.array(cumulative)

    # Exposure ratios: smooth + normalize to median
    luminances = np.array(luminances, dtype=np.float64)
    smooth_win = min(opts['exposure_smooth'], max(3, num_frames // 4))
    if smooth_win > 1:
        luminances = uniform_filter1d(luminances, size=smooth_win)
    median_lum = float(np.median(luminances))
    if median_lum > 0:
        exposure_ratios = luminances / median_lum
    else:
        exposure_ratios = np.ones(num_frames)
    exposure_ratios = np.clip(exposure_ratios, 0.5, 2.0).astype(np.float32)

    # Light direction: average samples -> quadrant label
    if light_samples:
        avg_dx = float(np.mean([s[0] for s in light_samples]))
        avg_dy = float(np.mean([s[1] for s in light_samples]))
    else:
        avg_dx, avg_dy = 0.0, 0.0

    # Convert to quadrant (image coords: y+ is down)
    parts = []
    if abs(avg_dy) > 2:
        parts.append("upper" if avg_dy < 0 else "lower")
    if abs(avg_dx) > 2:
        parts.append("right" if avg_dx > 0 else "left")
    quadrant = " ".join(parts) if parts else "above"

    # Camera height: if alpha extends to bottom, feet are cropped = low camera
    avg_bottom_alpha = float(np.mean(bottom_alpha_vals))
    if avg_bottom_alpha > 100:
        camera_height = "low"   # waist/hip height, feet fully cropped
    elif avg_bottom_alpha > 30:
        camera_height = "mid"   # some feet visible
    else:
        camera_height = "eye"   # full body visible

    camera_info = {
        'cumulative': cumulative,
        'meas_scale_x': scale_x,
        'meas_scale_y': scale_y,
        'num_frames': num_frames,
    }
    light_info = {
        'quadrant': quadrant, 'dx': avg_dx, 'dy': avg_dy,
        'camera_height': camera_height,
    }

    drift = np.sqrt(cumulative[-1, 0]**2 + cumulative[-1, 1]**2) if num_frames > 1 else 0
    exp_range = (exposure_ratios.min(), exposure_ratios.max())
    print(f"  {num_frames} frames in {time.time()-t0:.1f}s")
    print(f"  Camera drift: {drift:.1f}px meas ({drift*scale_x:.0f}px proc)")
    print(f"  Exposure range: {exp_range[0]:.2f}-{exp_range[1]:.2f}")
    print(f"  Light direction: {quadrant} (dx={avg_dx:.1f}, dy={avg_dy:.1f})")
    print(f"  Camera height: {camera_height} (bottom alpha={avg_bottom_alpha:.0f})")

    return camera_info, exposure_ratios, light_info


def build_studio_prompt(base_prompt, light_info):
    """Augment user prompt with light placement, camera height, and realism."""
    q = light_info['quadrant']
    ch = light_info.get('camera_height', 'eye')
    if ch == 'low':
        cam = "camera at waist height looking straight ahead, floor barely visible at bottom"
    elif ch == 'mid':
        cam = "camera at hip height, floor visible at bottom of frame"
    else:
        cam = "camera at eye level, floor fully visible"
    return (f"{base_prompt}, {cam}, bright key light from the {q}, "
            f"volumetric light rays from {q}, "
            f"photorealistic, 8k, detailed")


def round_to_8(x):
    return int(round(x / 8)) * 8


def generate_studio_bg(input_path, alpha_path, prompt, light_info, proc_w, proc_h, opts,
                       bg_w=None, bg_h=None):
    """Generate oversized studio background + depth map from original scene.

    1. Runs depth estimation on the ORIGINAL reference frame — captures real
       camera height, floor position, and scene geometry.
    2. Inpaints the dancer region in the depth map so parallax is smooth.
    3. Generates a new background with SD inpainting at high strength (complete
       reskin — no original bar visible, but spatial layout from init image
       provides some geometric guidance).

    Returns (bg_rgb, depth_map) at oversized dimensions.
    """
    import torch
    from PIL import Image

    margin = opts['bg_margin']
    if bg_w is None:
        bg_w = round_to_8(int(proc_w * margin))
    if bg_h is None:
        bg_h = round_to_8(int(proc_h * margin))

    print(f"\nGenerating studio background at {bg_w}x{bg_h} (margin={margin:.2f})...")

    # Extract mid-frame reference
    info = get_video_info(input_path)
    dur = opts['duration'] or info['duration']
    mid_time = dur / 2
    if opts['start']:
        mid_time += opts['start']

    rgb_gen = decode_frames(input_path, proc_w, proc_h, 'rgb24', 3,
                            start=mid_time, duration=0.1)
    ref_frame = next(rgb_gen, None)
    if ref_frame is None:
        print("ERROR: Could not extract reference frame.")
        sys.exit(1)

    # Get corresponding alpha for the mid-frame
    alpha_gen = decode_frames(alpha_path, proc_w, proc_h, 'gray', 1,
                              start=mid_time, duration=0.1)
    ref_alpha = next(alpha_gen, None)
    if ref_alpha is None:
        print("ERROR: Could not extract reference alpha.")
        sys.exit(1)

    # ── Depth from ORIGINAL scene (captures real camera geometry) ──
    print("  Loading Depth Anything V2 Small (for original scene)...")
    from transformers import pipeline as hf_pipeline
    depth_pipe = hf_pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cuda",
        torch_dtype=torch.float16,
    )
    alloc = torch.cuda.memory_allocated() / 1e6
    print(f"  VRAM: {alloc:.0f} MB")

    t0 = time.time()
    depth_result = depth_pipe(Image.fromarray(ref_frame))
    raw_depth = np.array(depth_result['depth']).astype(np.float32)
    # Resize to proc dims (model may return different size)
    if raw_depth.shape[:2] != (proc_h, proc_w):
        raw_depth = cv2.resize(raw_depth, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
    # Normalize to 0-1
    d_min, d_max = raw_depth.min(), raw_depth.max()
    if d_max > d_min:
        depth_proc = (raw_depth - d_min) / (d_max - d_min)
    else:
        depth_proc = np.ones_like(raw_depth) * 0.5

    # Inpaint dancer region in depth map so parallax is smooth behind them
    dancer_mask = (ref_alpha > int(opts['bg_thresh'] * 255)).astype(np.uint8) * 255
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dancer_mask_dilated = cv2.dilate(dancer_mask, kernel_d, iterations=2)
    depth_u8 = (depth_proc * 255).astype(np.uint8)
    depth_inpainted = cv2.inpaint(depth_u8, dancer_mask_dilated, 20, cv2.INPAINT_NS)
    depth_proc = depth_inpainted.astype(np.float32) / 255.0

    # Heavy blur to remove ALL person-shaped depth features (not just alpha-masked dancer).
    # Room geometry (walls, floor, vanishing points) is low-frequency and survives.
    # Person blobs are high-frequency and get smoothed away.
    blur_sigma = max(proc_w, proc_h) // 8  # ~80px on 640x360
    depth_proc = cv2.GaussianBlur(depth_proc, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    print(f"  Original depth computed in {time.time()-t0:.1f}s (blur sigma={blur_sigma})")

    del depth_pipe
    torch.cuda.empty_cache()

    # Pad depth to oversized dimensions
    pad_top = (bg_h - proc_h) // 2
    pad_bottom = bg_h - proc_h - pad_top
    pad_left = (bg_w - proc_w) // 2
    pad_right = bg_w - proc_w - pad_left
    depth_map = cv2.copyMakeBorder(depth_proc, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_REFLECT_101)

    # ── Generate new background with SD + ControlNet Depth ──────
    # ControlNet depth conditions on the original scene geometry so the
    # generated studio matches real camera height, floor position, and
    # wall layout. All-white inpaint mask = complete reskin.
    padded = cv2.copyMakeBorder(ref_frame, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_REFLECT_101)
    mask = np.ones((bg_h, bg_w), dtype=np.uint8) * 255

    # Depth control image: 3-channel grayscale for ControlNet
    depth_vis_u8 = (depth_map * 255).astype(np.uint8)
    depth_3ch = np.stack([depth_vis_u8] * 3, axis=-1)
    control_image = Image.fromarray(depth_3ch)

    init_image = Image.fromarray(padded)
    mask_image = Image.fromarray(mask)

    negative_prompt = ("cartoon, anime, painting, illustration, drawing, "
                       "low quality, blurry, distorted, watermark, text")

    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

    cn_model_id = "lllyasviel/control_v11f1p_sd15_depth"
    sd_model_id = "runwayml/stable-diffusion-inpainting"
    print(f"  Loading ControlNet depth: {cn_model_id}")
    controlnet = ControlNetModel.from_pretrained(
        cn_model_id, torch_dtype=torch.float16,
    )

    print(f"  Loading SD 1.5 inpainting + ControlNet pipeline...")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        sd_model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")

    generator = None
    if opts['seed'] is not None:
        generator = torch.Generator(device="cuda").manual_seed(opts['seed'])

    cn_scale = 0.8
    alloc = torch.cuda.memory_allocated() / 1e6
    print(f"  VRAM: {alloc:.0f} MB")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  ControlNet scale: {cn_scale}")
    print(f"  Generating (steps={opts['steps']}, strength={opts['strength']})...")
    t0 = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
        height=bg_h, width=bg_w,
        num_inference_steps=opts['steps'],
        strength=opts['strength'],
        controlnet_conditioning_scale=cn_scale,
        generator=generator,
    ).images[0]
    bg_rgb = np.array(result)
    print(f"  Generated in {time.time()-t0:.1f}s")

    del pipe, controlnet
    torch.cuda.empty_cache()

    # Save debug images
    out_dir = os.path.dirname(opts.get('_output_path', '')) or '.'
    debug_bg = os.path.join(out_dir, 'studio_bg.png')
    debug_depth = os.path.join(out_dir, 'studio_depth.png')
    cv2.imwrite(debug_bg, cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR))
    depth_vis = (depth_map * 255).astype(np.uint8)
    cv2.imwrite(debug_depth, depth_vis)
    print(f"  Saved: {debug_bg}")
    print(f"  Saved: {debug_depth}")

    return bg_rgb, depth_map


def parallax_warp(bg, depth, cam_dx, cam_dy, depth_scale, x_coords, y_coords):
    """Camera pan + depth-based 2.5D parallax warp.

    cam_dx/dy: base camera pan (all pixels shift by this amount).
    depth_scale: additional depth-dependent shift on top of base pan.
      Far objects (depth~0) shift by cam_dx only.
      Near objects (depth~1) shift by cam_dx * (1 + depth_scale).
    """
    map_x = (x_coords + cam_dx * (1.0 + depth * depth_scale)).astype(np.float32)
    map_y = (y_coords + cam_dy * (1.0 + depth * depth_scale)).astype(np.float32)
    warped = cv2.remap(bg, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    return warped


def studio_pipeline(input_path, alpha_path, output_path, opts):
    """Studio background replacement: analysis -> generation -> streaming render."""
    info = get_video_info(input_path)
    w, h, fps = info['width'], info['height'], info['fps']
    resolution = opts['resolution']

    if h > resolution:
        proc_h = resolution
        proc_w = int(w / h * proc_h)
        proc_w -= proc_w % 2
    else:
        proc_h, proc_w = h, w

    print(f"Studio pipeline:")
    print(f"  Input: {w}x{h} @ {fps:.2f}fps")
    print(f"  Processing at: {proc_w}x{proc_h}")

    # ── Phase 1: Analysis ────────────────────────────────────────
    print("\n═══ Phase 1: Analysis ═══")
    camera_info, exposure_ratios, light_info = analyze_video(
        input_path, alpha_path, proc_w, proc_h, opts)

    # ── Auto-compute bg_margin from camera drift ─────────────────
    cumulative = camera_info['cumulative']
    num_frames = camera_info['num_frames']
    ref_idx = num_frames // 2
    scale_x = camera_info['meas_scale_x']
    scale_y = camera_info['meas_scale_y']
    depth_scale = opts['depth_scale']

    # Max offset from reference frame, in proc pixels
    offsets_x = np.abs(cumulative[:, 0] - cumulative[ref_idx, 0]) * scale_x
    offsets_y = np.abs(cumulative[:, 1] - cumulative[ref_idx, 1]) * scale_y
    max_off_x = float(offsets_x.max()) * (1.0 + depth_scale)
    max_off_y = float(offsets_y.max()) * (1.0 + depth_scale)

    # Margin needed: both sides + 10% safety padding
    auto_margin_x = 1.0 + 2.0 * max_off_x / proc_w * 1.1
    auto_margin_y = 1.0 + 2.0 * max_off_y / proc_h * 1.1
    auto_margin = max(auto_margin_x, auto_margin_y)

    if opts['bg_margin'] > 0:
        margin = max(opts['bg_margin'], auto_margin)
    else:
        margin = max(1.3, auto_margin)  # at least 1.3
    opts['bg_margin'] = margin

    bg_w = round_to_8(int(proc_w * margin))
    bg_h = round_to_8(int(proc_h * margin))
    print(f"\n  Auto margin: {margin:.2f} (bg={bg_w}x{bg_h}, "
          f"max drift: ±{max_off_x:.0f}x ±{max_off_y:.0f}px)")

    # ── Phase 2: Generation ──────────────────────────────────────
    print("\n═══ Phase 2: Generation ═══")
    base_prompt = opts['prompt'] or "neon dance studio with mirrors and LED strip lighting"
    prompt = build_studio_prompt(base_prompt, light_info)
    bg_rgb, depth_map = generate_studio_bg(
        input_path, alpha_path, prompt, light_info, proc_w, proc_h, opts,
        bg_w=bg_w, bg_h=bg_h)

    # ── Phase 3: Streaming render ────────────────────────────────
    print("\n═══ Phase 3: Render ═══")
    bg_h, bg_w = bg_rgb.shape[:2]

    # Pre-compute coordinate grids for parallax warp (at oversized dims)
    y_grid, x_grid = np.mgrid[0:bg_h, 0:bg_w]
    x_coords = x_grid.astype(np.float32)
    y_coords = y_grid.astype(np.float32)

    # Crop rect: center region of oversized bg -> proc_w x proc_h
    crop_x0 = (bg_w - proc_w) // 2
    crop_y0 = (bg_h - proc_h) // 2

    bg_f = bg_rgb.astype(np.float32)

    start_args = ['-ss', str(opts['start'])] if opts['start'] else []
    dur_args = ['-t', str(opts['duration'])] if opts['duration'] else []

    # Decode input video
    decoder = subprocess.Popen([
        'ffmpeg', '-hwaccel', 'cuda', *start_args, *dur_args,
        '-i', input_path,
        '-vf', f'scale={proc_w}:{proc_h}',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-v', 'error', '-'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Decode alpha
    alpha_dec = subprocess.Popen([
        'ffmpeg', '-hwaccel', 'cuda', *start_args, *dur_args,
        '-i', alpha_path,
        '-vf', f'scale={proc_w}:{proc_h}',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-v', 'error', '-'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Encode output — copy audio from input
    qp = opts['qp']
    encoder = subprocess.Popen([
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{proc_w}x{proc_h}', '-r', str(fps), '-i', '-',
        *start_args, *dur_args, '-i', input_path,
        '-map', '0:v', '-map', '1:a?',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'h264_nvenc', '-rc', 'constqp', '-qp', str(qp),
        '-c:a', 'aac', '-b:a', '192k', '-v', 'error', output_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    rgb_bytes = proc_w * proc_h * 3
    alpha_bytes = proc_w * proc_h
    comp_f = np.empty((proc_h, proc_w, 3), dtype=np.float32)
    result = np.empty((proc_h, proc_w, 3), dtype=np.uint8)

    idx = 0
    t0 = time.time()
    print(f"  Rendering {num_frames} frames...")

    while True:
        raw_rgb = decoder.stdout.read(rgb_bytes)
        raw_alpha = alpha_dec.stdout.read(alpha_bytes)
        if len(raw_rgb) < rgb_bytes or len(raw_alpha) < alpha_bytes:
            break

        frame = np.frombuffer(raw_rgb, dtype=np.uint8).reshape(proc_h, proc_w, 3)
        alpha = np.frombuffer(raw_alpha, dtype=np.uint8).reshape(proc_h, proc_w).copy()

        # Filter alpha: remove stray background people from MatAnyone matte.
        # Keep only connected components >= 40% the size of the largest.
        thresh_mask = (alpha > 128).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            thresh_mask, connectivity=8)
        if num_labels > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = int(areas.max())
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] < max_area * 0.4:
                    alpha[labels == lbl] = 0

        alpha_f = alpha.astype(np.float32) / 255.0

        # Camera offset: negate so bg moves opposite to camera
        # phaseCorrelate gives negative cumulative when camera pans right,
        # we negate so remap samples rightward = image shifts left = correct
        fi = min(idx, num_frames - 1)
        cam_dx = -(cumulative[fi, 0] - cumulative[ref_idx, 0]) * scale_x
        cam_dy = -(cumulative[fi, 1] - cumulative[ref_idx, 1]) * scale_y

        # Parallax warp: base pan + depth-dependent shift
        warped = parallax_warp(bg_f, depth_map, cam_dx, cam_dy, depth_scale,
                               x_coords, y_coords)

        # Crop center region to proc dimensions
        warped_crop = warped[crop_y0:crop_y0+proc_h, crop_x0:crop_x0+proc_w]

        # Exposure adjustment
        exp_ratio = float(exposure_ratios[fi])
        if abs(exp_ratio - 1.0) > 0.01:
            warped_crop = warped_crop * exp_ratio

        # Alpha composite: fg * alpha + bg * (1 - alpha)
        for c in range(3):
            comp_f[:, :, c] = (frame[:, :, c].astype(np.float32) * alpha_f +
                               warped_crop[:, :, c] * (1.0 - alpha_f))

        np.clip(comp_f, 0, 255, out=comp_f)
        np.copyto(result, comp_f, casting='unsafe')
        encoder.stdin.write(result.tobytes())

        idx += 1
        if idx % 30 == 0:
            e = time.time() - t0
            print(f"  {idx}/{num_frames} ({idx/e:.1f} fps)", flush=True)

    try:
        encoder.stdin.close()
    except BrokenPipeError:
        pass
    decoder.wait()
    alpha_dec.wait()
    encoder.wait()

    print(f"  Encoded {idx} frames in {time.time()-t0:.1f}s")
    print(f"  Output: {output_path}")


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    input_path = sys.argv[2]
    alpha_path = sys.argv[3]
    output_path = sys.argv[4]
    opts = parse_opts(sys.argv[5:])

    if not os.path.exists(input_path):
        print(f"Not found: {input_path}")
        sys.exit(1)
    if alpha_path != 'auto' and not os.path.exists(alpha_path):
        print(f"Not found: {alpha_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f"Mode: {mode}\n")

    opts['_output_path'] = output_path

    if mode == 'plate':
        result, out_w, out_h = build_clean_plate(input_path, alpha_path, opts)
        ext = os.path.splitext(output_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                         [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            if ext != '.png':
                output_path = os.path.splitext(output_path)[0] + '.png'
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"\nOutput: {output_path} ({out_w}x{out_h})")

    elif mode == 'generate':
        result, out_w, out_h = generate_background(input_path, alpha_path, opts)
        ext = os.path.splitext(output_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                         [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            if ext != '.png':
                output_path = os.path.splitext(output_path)[0] + '.png'
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"\nOutput: {output_path} ({out_w}x{out_h})")

    elif mode == 'composite':
        composite_video(input_path, alpha_path, output_path, opts)

    elif mode == 'studio':
        studio_pipeline(input_path, alpha_path, output_path, opts)

    else:
        print(f"Unknown mode: {mode}")
        print("Available: plate, generate, composite, studio")
        sys.exit(1)

    print("Done!")


if __name__ == '__main__':
    main()
