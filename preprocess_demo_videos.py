"""
Preprocess demo videos into t+3 triplets for fine-tuning
Creates triplets: (frame_t, frame_t+1, frame_t+3)
"""
import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_triplets_from_video(video_path, output_dir, target_size=512, frame_skip=2):
    """
    Extract t+3 triplets from a video.
    
    Args:
        video_path: Path to video file
        output_dir: Base output directory
        target_size: Resize frames to this size
        frame_skip: Skip frames to reduce redundancy (default=2 means use every 2nd frame)
    
    Creates triplets where:
        im1.png = frame at time t
        im2.png = frame at time t+1  
        im3.png = frame at time t+3 (target for prediction)
    """
    video_name = Path(video_path).stem
    print(f"\nProcessing: {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"  Loaded {len(frames)} frames")
    
    # Create triplets
    triplet_count = 0
    
    # Use frame_skip to reduce redundancy
    for i in range(0, len(frames) - 4, frame_skip):
        # Triplet: frame i, frame i+1, frame i+3
        frame_t = frames[i]
        frame_t1 = frames[i + 1]
        frame_t3 = frames[i + 3]
        
        # Resize to target size
        frame_t = cv2.resize(frame_t, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        frame_t1 = cv2.resize(frame_t1, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        frame_t3 = cv2.resize(frame_t3, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Create output folder
        triplet_dir = os.path.join(output_dir, f"{video_name}_{triplet_count:05d}")
        os.makedirs(triplet_dir, exist_ok=True)
        
        # Save triplet
        cv2.imwrite(os.path.join(triplet_dir, "im1.png"), frame_t)
        cv2.imwrite(os.path.join(triplet_dir, "im2.png"), frame_t1)
        cv2.imwrite(os.path.join(triplet_dir, "im3.png"), frame_t3)
        
        triplet_count += 1
    
    print(f"  Created {triplet_count} triplets")
    return triplet_count

def main():
    parser = argparse.ArgumentParser(description='Preprocess demo videos for t+3 training')
    parser.add_argument('--input_dir', type=str, 
                        default='/storage_drive_0/vivek/video/demo videos',
                        help='Directory containing demo videos')
    parser.add_argument('--output_dir', type=str,
                        default='/storage_drive_0/vivek/video/datasets/demo_t3_triplets',
                        help='Output directory for triplets')
    parser.add_argument('--target_size', type=int, default=512,
                        help='Resize frames to this size')
    parser.add_argument('--frame_skip', type=int, default=2,
                        help='Use every Nth frame to reduce redundancy')
    args = parser.parse_args()
    
    print("="*60)
    print("Demo Video Preprocessing for t+3 Training")
    print("="*60)
    
    # Find all video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(args.input_dir).glob(f'*{ext}'))
    
    print(f"\nFound {len(video_files)} videos:")
    for v in video_files:
        print(f"  - {v.name}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each video
    total_triplets = 0
    for video_path in video_files:
        count = extract_triplets_from_video(
            str(video_path), 
            args.output_dir,
            args.target_size,
            args.frame_skip
        )
        total_triplets += count
    
    print("\n" + "="*60)
    print(f"COMPLETE!")
    print(f"  Total triplets created: {total_triplets}")
    print(f"  Output directory: {args.output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()






