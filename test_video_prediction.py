"""
Video prediction inference script for IFRVP.
Tests the arbitrary prediction method on real video data.
"""
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import argparse

# Add IFRVP directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IFRVP'))

def extract_frames(video_path, max_frames=None, resize=None):
    """Extract frames from video"""
    print(f"Extracting frames from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if specified
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
        
        frames.append(frame)
        frame_idx += 1
        
        if max_frames and frame_idx >= max_frames:
            break
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    return frames, fps

def frames_to_tensor(frames, device):
    """Convert list of numpy frames to torch tensor"""
    # frames: list of [H, W, 3] numpy arrays (0-255)
    # output: [N, 3, H, W] tensor (0-1)
    tensors = []
    for frame in frames:
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    return torch.stack(tensors).to(device)

def tensor_to_frame(tensor):
    """Convert torch tensor to numpy frame"""
    # tensor: [3, H, W] (0-1)
    # output: [H, W, 3] numpy array (0-255)
    frame = tensor.cpu().permute(1, 2, 0).numpy()
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    return frame

def predict_future_frames(model, frames_tensor, num_predictions=5, timestep_delta=1.0, device='cuda'):
    """
    Predict future frames using arbitrary prediction method.
    
    Args:
        model: IFRVP model
        frames_tensor: [N, 3, H, W] tensor of input frames
        num_predictions: number of future frames to predict
        timestep_delta: timestep increment (1.0 = next frame, 2.0 = 2 frames ahead, etc.)
        device: computation device
    
    Returns:
        predictions: list of predicted frames [num_predictions, 3, H, W]
        ground_truth: list of actual frames for comparison (if available)
    """
    model.eval()
    predictions = []
    ground_truth = []
    
    print(f"\nPredicting {num_predictions} future frames...")
    
    # Use the last two frames as context
    img0 = frames_tensor[-2].unsqueeze(0)  # [1, 3, H, W]
    img1 = frames_tensor[-1].unsqueeze(0)  # [1, 3, H, W]
    
    with torch.no_grad():
        for i in range(1, num_predictions + 1):
            # Timestep embedding: predict i * timestep_delta frames into the future
            t = i * timestep_delta
            embt = torch.tensor([t]).view(1, 1, 1, 1).to(device)
            
            # Run inference
            pred = model.inference(img0, img1, embt)
            predictions.append(pred.squeeze(0))
            
            print(f"  Predicted frame at t={t:.1f}")
    
    return predictions

def create_comparison_grid(img0, img1, predictions, ground_truth=None, max_cols=4):
    """Create a grid visualization comparing predictions"""
    frames = [img0, img1] + predictions
    labels = ['Frame t-1', 'Frame t'] + [f'Pred t+{i}' for i in range(1, len(predictions) + 1)]
    
    # Convert all to numpy
    frames_np = [tensor_to_frame(f) for f in frames]
    
    # Calculate grid dimensions
    n_frames = len(frames_np)
    n_cols = min(max_cols, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    # Get frame dimensions
    h, w = frames_np[0].shape[:2]
    
    # Add padding for labels
    label_height = 30
    frame_height = h + label_height
    
    # Create grid
    grid = np.ones((n_rows * frame_height, n_cols * w, 3), dtype=np.uint8) * 255
    
    for idx, (frame, label) in enumerate(zip(frames_np, labels)):
        row = idx // n_cols
        col = idx % n_cols
        
        # Add label
        label_img = np.ones((label_height, w, 3), dtype=np.uint8) * 255
        cv2.putText(label_img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Combine label and frame
        combined = np.vstack([label_img, frame])
        
        # Place in grid
        y_start = row * frame_height
        x_start = col * w
        grid[y_start:y_start + frame_height, x_start:x_start + w] = combined
    
    return grid

def save_predictions_as_video(predictions, output_path, fps=30):
    """Save predicted frames as a video"""
    if len(predictions) == 0:
        return
    
    # Get dimensions from first frame
    first_frame = tensor_to_frame(predictions[0])
    h, w = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for pred in predictions:
        frame = tensor_to_frame(pred)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved predictions video to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Test IFRVP video prediction')
    parser.add_argument('--video', type=str, default='../cam2_front.mp4',
                        help='Path to input video')
    parser.add_argument('--model_path', type=str, 
                        default='IFRVP/IFRVP_k+1_laploss/IFRNet_S_latest.pth',
                        help='Path to pre-trained model')
    parser.add_argument('--model_name', type=str, default='IFRNet_S',
                        help='Model architecture (IFRNet_S, IFRNet, IFRNet_L)')
    parser.add_argument('--max_frames', type=int, default=100,
                        help='Maximum number of frames to extract from video')
    parser.add_argument('--num_predictions', type=int, default=10,
                        help='Number of future frames to predict')
    parser.add_argument('--timestep_delta', type=float, default=1.0,
                        help='Timestep increment (1.0 = next frame, 2.0 = skip 1 frame)')
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize frames to this size (square)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"IFRVP Video Prediction Test")
    print(f"{'='*60}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    if args.model_name == 'IFRNet_S':
        from models.IFRNet_S import Model
    elif args.model_name == 'IFRNet':
        from models.IFRNet import Model
    elif args.model_name == 'IFRNet_L':
        from models.IFRNet_L import Model
    else:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    model = Model().to(device)
    
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded weights from: {args.model_path}")
    else:
        print(f"✗ Model weights not found at: {args.model_path}")
        return
    
    model.eval()
    
    # Extract frames from video
    resize = (args.resize, args.resize) if args.resize else None
    frames, fps = extract_frames(args.video, max_frames=args.max_frames, resize=resize)
    
    if len(frames) < 2:
        print("Error: Need at least 2 frames")
        return
    
    # Convert to tensors
    frames_tensor = frames_to_tensor(frames, device)
    
    # Predict future frames
    predictions = predict_future_frames(
        model, frames_tensor, 
        num_predictions=args.num_predictions,
        timestep_delta=args.timestep_delta,
        device=device
    )
    
    # Create visualization
    print("\nCreating visualizations...")
    
    # Save individual predictions
    for i, pred in enumerate(predictions):
        pred_frame = tensor_to_frame(pred)
        pred_img = Image.fromarray(pred_frame)
        pred_img.save(os.path.join(args.output_dir, f'prediction_{i+1:03d}.png'))
    
    # Create comparison grid
    grid = create_comparison_grid(
        frames_tensor[-2], frames_tensor[-1], predictions[:6]  # Show first 6 predictions
    )
    grid_img = Image.fromarray(grid)
    grid_path = os.path.join(args.output_dir, 'comparison_grid.png')
    grid_img.save(grid_path)
    print(f"✓ Saved comparison grid to: {grid_path}")
    
    # Save predictions as video
    video_path = os.path.join(args.output_dir, 'predictions.mp4')
    save_predictions_as_video(predictions, video_path, fps=fps)
    
    # Save context frames for reference
    context_frame0 = tensor_to_frame(frames_tensor[-2])
    context_frame1 = tensor_to_frame(frames_tensor[-1])
    Image.fromarray(context_frame0).save(os.path.join(args.output_dir, 'context_frame_0.png'))
    Image.fromarray(context_frame1).save(os.path.join(args.output_dir, 'context_frame_1.png'))
    
    print(f"\n{'='*60}")
    print(f"Testing complete! Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()

