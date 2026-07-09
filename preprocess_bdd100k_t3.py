"""
Preprocess BDD100K driving videos into t+3 triplets for IFRVP training.

For t+3 prediction, we need triplets: (frame_t, frame_t+1, frame_t+4)
- img0 = frame_t
- img1 = frame_t+1  
- imgt = frame_t+4 (ground truth, 3 frames ahead of img1)

This creates the dataset structure expected by IFRVP training:
  dataset_dir/sequences/XXXX/im1.png, im2.png, im3.png
"""

import os
import cv2
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random


def extract_triplets_from_video(args):
    """Extract t+3 triplets from a single video."""
    video_path, output_dir, target_size, skip_frames = args
    
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return 0
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to target size
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < 5:  # Need at least 5 frames for t+3
        return 0
    
    triplet_count = 0
    
    # Create triplets with stride to avoid too much redundancy
    # For t+3: we need frames at t, t+1, t+4
    for i in range(0, len(frames) - 4, skip_frames):
        img0 = frames[i]      # frame t
        img1 = frames[i + 1]  # frame t+1
        imgt = frames[i + 4]  # frame t+4 (target, 3 frames after img1)
        
        # Create unique triplet folder
        triplet_id = f"{video_name}_{i:06d}"
        triplet_dir = os.path.join(output_dir, "sequences", triplet_id)
        os.makedirs(triplet_dir, exist_ok=True)
        
        # Save frames (im1, im2, im3 convention from IFRVP)
        cv2.imwrite(os.path.join(triplet_dir, "im1.png"), img0)
        cv2.imwrite(os.path.join(triplet_dir, "im2.png"), img1)
        cv2.imwrite(os.path.join(triplet_dir, "im3.png"), imgt)
        
        triplet_count += 1
    
    return triplet_count


def main():
    parser = argparse.ArgumentParser(description='Preprocess BDD100K for t+3 prediction')
    parser.add_argument('--input_dir', type=str, 
                        default='/storage_drive_0/vivek/video/datasets/bdd100k_videos_train_00/bdd100k/videos/train',
                        help='Directory containing .mov video files')
    parser.add_argument('--output_dir', type=str,
                        default='/storage_drive_0/vivek/video/datasets/bdd100k_t3_triplets',
                        help='Output directory for triplets')
    parser.add_argument('--size', type=int, default=512,
                        help='Target size (will be square)')
    parser.add_argument('--skip_frames', type=int, default=5,
                        help='Skip frames between triplets (reduces redundancy)')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Max number of videos to process (for testing)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"BDD100K to t+3 Triplets Preprocessing")
    print(f"{'='*70}\n")
    
    # Find all video files
    video_dir = Path(args.input_dir)
    video_files = list(video_dir.glob("*.mov"))
    
    if args.max_videos:
        video_files = video_files[:args.max_videos]
    
    print(f"Found {len(video_files)} videos")
    print(f"Target size: {args.size}x{args.size}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, "sequences"), exist_ok=True)
    
    # Prepare arguments for parallel processing
    target_size = (args.size, args.size)
    process_args = [
        (str(v), args.output_dir, target_size, args.skip_frames) 
        for v in video_files
    ]
    
    # Process videos in parallel
    print(f"\nProcessing with {args.num_workers} workers...")
    
    total_triplets = 0
    with Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_triplets_from_video, process_args),
            total=len(video_files),
            desc="Processing videos"
        ))
        total_triplets = sum(results)
    
    # Create train.txt file listing all triplet folders
    sequences_dir = os.path.join(args.output_dir, "sequences")
    triplet_folders = sorted(os.listdir(sequences_dir))
    
    # Shuffle and split into train (no validation for now)
    random.seed(42)
    random.shuffle(triplet_folders)
    
    with open(os.path.join(args.output_dir, "tri_trainlist.txt"), "w") as f:
        for folder in triplet_folders:
            f.write(f"{folder}\n")
    
    print(f"\n{'='*70}")
    print(f"DONE!")
    print(f"{'='*70}")
    print(f"Total triplets created: {total_triplets}")
    print(f"Triplet list saved to: {args.output_dir}/tri_trainlist.txt")
    print(f"\nEstimated disk usage: ~{total_triplets * 3 * 0.5:.1f} GB")
    print(f"(assuming ~0.5 MB per {args.size}x{args.size} PNG)\n")


if __name__ == '__main__':
    main()

