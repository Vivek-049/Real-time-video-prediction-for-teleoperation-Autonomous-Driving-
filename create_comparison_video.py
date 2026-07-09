"""
Create side-by-side comparison video for teleoperation scenario:
- Left: Sender (Ground Truth) - Current frame at time t
- Middle: Receiver (Delayed) - Frame at time t-delay (what receiver sees due to network latency)
- Right: Predicted - Frame predicted from delayed frames to compensate and match sender
"""
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse
import imageio

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
    tensors = []
    for frame in frames:
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    return torch.stack(tensors).to(device)

def tensor_to_frame(tensor):
    """Convert torch tensor to numpy frame"""
    frame = tensor.cpu().permute(1, 2, 0).numpy()
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    return frame

def add_label_to_frame(frame, label, color=(255, 255, 255), extra_label=None):
    """Add text label to frame with optional extra label"""
    h, w = frame.shape[:2]
    labeled = frame.copy()
    
    # Add semi-transparent light blue bar at top
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (200, 200, 255), -1)
    cv2.addWeighted(overlay, 0.7, labeled, 0.3, 0, labeled)
    
    # Add main text (data ID)
    cv2.putText(labeled, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (200, 0, 0), 2, cv2.LINE_AA)
    
    # Add extra label if provided (prediction info)
    if extra_label:
        cv2.putText(labeled, extra_label, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (200, 0, 0), 2, cv2.LINE_AA)
    
    return labeled

def create_side_by_side_frame(sender_frame, receiver_frame, predicted_frame, 
                              sender_id, receiver_id, predicted_id, prediction_info=""):
    """Create side-by-side comparison of three frames"""
    # Add labels with data IDs and names
    sender_labeled = add_label_to_frame(sender_frame, f"sender data:{sender_id}")
    receiver_labeled = add_label_to_frame(receiver_frame, f"receiver data:{receiver_id}")
    predicted_labeled = add_label_to_frame(predicted_frame, f"predicted", 
                                           extra_label=f"predicted:{predicted_id}")
    
    # Add vertical separators (black)
    separator = np.ones((sender_labeled.shape[0], 2, 3), dtype=np.uint8) * 0
    
    # Concatenate horizontally
    combined = np.hstack([sender_labeled, separator, receiver_labeled, separator, predicted_labeled])
    
    return combined

def predict_sequence(model, img0, img1, num_frames, device):
    """Predict a sequence of future frames"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(1, num_frames + 1):
            embt = torch.tensor([float(i)]).view(1, 1, 1, 1).to(device)
            pred = model.inference(img0, img1, embt)
            predictions.append(pred.squeeze(0))
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Create comparison video for IFRVP')
    parser.add_argument('--video', type=str, default='../cam2_front.mp4',
                        help='Path to input video')
    parser.add_argument('--model_path', type=str, 
                        default='IFRVP/checkpoint_demo_finetune/IFRNet_S/2025-12-30_05-52-18/IFRNet_S_demo_finetuned.pth',
                        help='Path to pre-trained model')
    parser.add_argument('--model_name', type=str, default='IFRNet_S',
                        help='Model architecture')
    parser.add_argument('--start_frame', type=int, default=10,
                        help='Start frame index')
    parser.add_argument('--num_frames', type=int, default=60,
                        help='Number of frames to process')
    parser.add_argument('--delay_frames', type=int, default=3,
                        help='Network delay in frames (e.g., 3 frames = 0.2s at 15fps)')
    parser.add_argument('--resize', type=int, default=512,
                        help='Resize frames to this size')
    parser.add_argument('--output', type=str, default='output/teleop_finetuned_t3.mp4',
                        help='Output video path')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"IFRVP Comparison Video Generator")
    print(f"{'='*70}\n")
    
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
    
    # Extract frames
    resize = (args.resize, args.resize) if args.resize else None
    max_frames = args.start_frame + args.num_frames + args.delay_frames + 10
    frames, fps = extract_frames(args.video, max_frames=max_frames, resize=resize)
    
    if len(frames) < args.start_frame + args.num_frames + args.delay_frames:
        print(f"Error: Not enough frames. Need {args.start_frame + args.num_frames + args.delay_frames}, got {len(frames)}")
        return
    
    # Convert to tensors
    frames_tensor = frames_to_tensor(frames, device)
    
    print(f"\nTeleoperation simulation:")
    print(f"  Network delay: {args.delay_frames} frames ({args.delay_frames/fps:.3f} seconds)")
    print(f"  Processing {args.num_frames} frames starting from frame {args.start_frame}")
    print(f"  Video FPS: {fps}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create comparison video
    print(f"\nCreating teleoperation comparison video...")
    
    # Get dimensions
    h, w = frames[0].shape[:2]
    output_w = w * 3 + 6  # 3 frames + 2 separators
    output_h = h
    
    # Make dimensions even (required for some codecs)
    if output_w % 2 != 0:
        output_w += 1
    if output_h % 2 != 0:
        output_h += 1
    
    print(f"  Output resolution: {output_w}x{output_h}")
    
    # Create video writer with mp4v codec (most compatible)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (output_w, output_h))
    
    if not out.isOpened():
        print("ERROR: Could not open video writer!")
        return
    
    frame_count = 0
    
    # Process each frame in the sequence
    for i in range(args.num_frames):
        current_idx = args.start_frame + i
        
        # Sender (Ground Truth): Current frame at time t
        sender_frame = frames[current_idx]
        
        # Receiver (Delayed): Frame at time t - delay (what receiver sees due to network latency)
        delayed_idx = current_idx - args.delay_frames
        if delayed_idx >= 0:
            receiver_frame = frames[delayed_idx]
        else:
            receiver_frame = np.zeros_like(sender_frame)
        
        # Predicted: Use delayed frames to predict current frame (compensate for delay)
        # Use DIRECT t+3 prediction as model is now fine-tuned for it
        if delayed_idx >= 1:
            # Use frames at t-3 and t-2 to predict frame at t
            img0 = frames_tensor[delayed_idx - 1].unsqueeze(0)  # frame at t-3
            img1 = frames_tensor[delayed_idx].unsqueeze(0)      # frame at t-2
            
            # embt = 1.0 because the model is fine-tuned to predict 3 frames ahead with embt=1.0
            embt = torch.tensor([1.0]).view(1, 1, 1, 1).to(device)
            
            with torch.no_grad():
                # Single inference for direct t+3 prediction
                pred = model.inference(img0, img1, embt)
            
            predicted_frame = tensor_to_frame(pred.squeeze(0))
        else:
            predicted_frame = np.zeros_like(sender_frame)
        
        # Create side-by-side comparison with data IDs
        comparison = create_side_by_side_frame(
            sender_frame, 
            receiver_frame, 
            predicted_frame,
            sender_id=current_idx,
            receiver_id=delayed_idx,
            predicted_id=current_idx,
            prediction_info=f"{args.delay_frames} frames ahead"
        )
        
        # Resize if needed to match output dimensions
        if comparison.shape[1] != output_w or comparison.shape[0] != output_h:
            comparison = cv2.resize(comparison, (output_w, output_h))
        
        # Convert RGB to BGR for OpenCV
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        out.write(comparison_bgr)
        
        frame_count += 1
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{args.num_frames} frames")
    
    out.release()
    
    print(f"\n✓ Saved comparison video to: {args.output}")
    print(f"  Total frames: {frame_count}")
    print(f"  Resolution: {output_w}x{output_h}")
    print(f"  FPS: {fps}")
    
    print(f"\n{'='*70}")
    print(f"Done!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()

