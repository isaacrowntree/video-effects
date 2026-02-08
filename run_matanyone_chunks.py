#!/usr/bin/env python3
"""Process full video in chunks with MatAnyone to avoid OOM."""
import subprocess
import sys
import os
import time
import numpy as np
import cv2

CHUNK_DIR = 'tmp/matting/chunks'
MASK_360P = 'tmp/matting/first_frame_mask_360p.png'
OUTPUT_DIR = 'tmp/matting/matanyone'

def get_last_frame_alpha(alpha_video_path):
    """Extract last frame from alpha video as a mask image."""
    cap = cv2.VideoCapture(alpha_video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read last frame from {alpha_video_path}")
    # Convert to binary mask (white where alpha > 128)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return mask

def process_chunk(chunk_idx, chunk_path, mask_path, output_prefix):
    """Process a single chunk with MatAnyone."""
    from matanyone import InferenceCore

    print(f"\n=== Chunk {chunk_idx}: {chunk_path} ===")
    print(f"  Mask: {mask_path}")

    t0 = time.time()
    processor = InferenceCore('PeiqingYang/MatAnyone')
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    fgr, pha = processor.process_video(
        input_path=chunk_path,
        mask_path=mask_path,
        output_path=output_prefix,
        r_erode=5,
        r_dilate=5,
        max_size=360,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Alpha: {pha}")

    # Cleanup GPU memory
    import torch
    del processor
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return pha

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find chunks
    chunks = sorted([f for f in os.listdir(CHUNK_DIR) if f.startswith('chunk_') and f.endswith('.mp4')])
    print(f"Found {len(chunks)} chunks")

    alpha_videos = []
    current_mask = MASK_360P

    for i, chunk_file in enumerate(chunks):
        chunk_path = os.path.join(CHUNK_DIR, chunk_file)
        output_prefix = os.path.join(CHUNK_DIR, f'matanyone_chunk_{i}')

        # Save current mask for this chunk
        mask_path = os.path.join(CHUNK_DIR, f'mask_chunk_{i}.png')
        if i == 0:
            # Use original mask
            import shutil
            shutil.copy2(current_mask, mask_path)
        else:
            # Use last frame alpha from previous chunk as mask
            cv2.imwrite(mask_path, last_alpha_mask)

        # Process chunk
        pha_path = process_chunk(i, chunk_path, mask_path, output_prefix)
        alpha_videos.append(pha_path)

        # Extract last frame for next chunk's mask
        last_alpha_mask = get_last_frame_alpha(pha_path)
        print(f"  Extracted last-frame mask for next chunk")

    # Concatenate all alpha videos
    print(f"\n=== Concatenating {len(alpha_videos)} alpha videos ===")
    concat_list = os.path.join(CHUNK_DIR, 'concat_list.txt')
    with open(concat_list, 'w') as f:
        for v in alpha_videos:
            f.write(f"file '{os.path.abspath(v)}'\n")

    output_alpha = os.path.join(OUTPUT_DIR, 'c1595_full_360p_pha.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list,
        '-c', 'copy', output_alpha
    ], check=True, capture_output=True)
    print(f"Final alpha: {output_alpha}")

    # Also concat foreground videos
    fgr_videos = [v.replace('_pha.mp4', '_fgr.mp4') for v in alpha_videos]
    if all(os.path.exists(v) for v in fgr_videos):
        concat_list_fgr = os.path.join(CHUNK_DIR, 'concat_list_fgr.txt')
        with open(concat_list_fgr, 'w') as f:
            for v in fgr_videos:
                f.write(f"file '{os.path.abspath(v)}'\n")
        output_fgr = os.path.join(OUTPUT_DIR, 'c1595_full_360p_fgr.mp4')
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_fgr,
            '-c', 'copy', output_fgr
        ], check=True, capture_output=True)
        print(f"Final foreground: {output_fgr}")

    print("\nAll done!")

if __name__ == '__main__':
    main()
