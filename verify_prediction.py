"""
Verify that prediction is actually working and not just copying frames.
Creates a detailed comparison showing:
1. Sender frame
2. Receiver frame (delayed)
3. Predicted frame
4. Difference between sender and predicted (should be small if prediction works)
5. Difference between receiver and predicted (should be large)
"""
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IFRVP'))

def extract_frames(video_path, start_idx, num_frames, resize=None):
    """Extract specific frames from video"""
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    for i in range(start_idx + num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
        
        if i >= start_idx:
            frames.append(frame)
    
    cap.release()
    return frames

def compute_mse(img1, img2):
    """Compute Mean Squared Error between two images"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def compute_psnr(img1, img2):
    """Compute PSNR between two images"""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    print("\n" + "="*70)
    print("VERIFICATION: Is prediction actually working?")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    from models.IFRNet_S import Model
    model = Model().to(device)
    model_path = 'IFRVP/IFRVP_k+1_laploss/IFRNet_S_latest.pth'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✓ Model loaded\n")
    
    # Extract test frames
    video_path = '../cam2_front.mp4'
    start_idx = 20
    delay = 3
    
    frames = extract_frames(video_path, start_idx, delay + 5, resize=(512, 512))
    print(f"Extracted {len(frames)} frames\n")
    
    # Test case: predict frame at index delay using frames 0 and 1
    receiver_frame_0 = frames[0]  # Frame at t-3
    receiver_frame_1 = frames[1]  # Frame at t-2
    sender_frame = frames[delay]  # Frame at t (what we're trying to predict)
    
    print("Test scenario:")
    print(f"  Receiver has: frames 0, 1 (delayed)")
    print(f"  Sender has: frame {delay} (current)")
    print(f"  Model predicts: {delay} frames ahead\n")
    
    # Convert to tensors
    img0 = torch.from_numpy(receiver_frame_0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = torch.from_numpy(receiver_frame_1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img0 = img0.to(device)
    img1 = img1.to(device)
    
    # Predict
    embt = torch.tensor([float(delay)]).view(1, 1, 1, 1).to(device)
    
    with torch.no_grad():
        pred_tensor = model.inference(img0, img1, embt)
    
    predicted_frame = pred_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    predicted_frame = np.clip(predicted_frame * 255, 0, 255).astype(np.uint8)
    
    print("Prediction complete!\n")
    
    # Compute metrics
    print("="*70)
    print("METRICS (lower MSE / higher PSNR = more similar)")
    print("="*70)
    
    # 1. Predicted vs Sender (should be similar if prediction works)
    mse_pred_sender = compute_mse(predicted_frame, sender_frame)
    psnr_pred_sender = compute_psnr(predicted_frame, sender_frame)
    print(f"\n1. Predicted vs Sender (target):")
    print(f"   MSE:  {mse_pred_sender:.2f}")
    print(f"   PSNR: {psnr_pred_sender:.2f} dB")
    
    # 2. Receiver vs Sender (should be different due to motion)
    mse_recv_sender = compute_mse(receiver_frame_1, sender_frame)
    psnr_recv_sender = compute_psnr(receiver_frame_1, sender_frame)
    print(f"\n2. Receiver (delayed) vs Sender:")
    print(f"   MSE:  {mse_recv_sender:.2f}")
    print(f"   PSNR: {psnr_recv_sender:.2f} dB")
    
    # 3. Predicted vs Receiver (should be different)
    mse_pred_recv = compute_mse(predicted_frame, receiver_frame_1)
    psnr_pred_recv = compute_psnr(predicted_frame, receiver_frame_1)
    print(f"\n3. Predicted vs Receiver (delayed):")
    print(f"   MSE:  {mse_pred_recv:.2f}")
    print(f"   PSNR: {psnr_pred_recv:.2f} dB")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if mse_pred_sender == 0:
        print("\n❌ PROBLEM: Predicted frame is IDENTICAL to sender!")
        print("   This means we're copying, not predicting!")
    elif mse_pred_sender < mse_recv_sender:
        improvement = ((mse_recv_sender - mse_pred_sender) / mse_recv_sender) * 100
        print(f"\n✓ Prediction is working!")
        print(f"  Predicted frame is {improvement:.1f}% closer to sender than receiver frame")
        print(f"  Model successfully compensated for {delay} frames of delay")
    else:
        print(f"\n⚠️  Prediction is worse than just using delayed frame")
        print(f"  This might indicate the model isn't working well on this video")
    
    # Save comparison images
    output_dir = 'output/verification'
    os.makedirs(output_dir, exist_ok=True)
    
    Image.fromarray(receiver_frame_0).save(f'{output_dir}/1_receiver_t-3.png')
    Image.fromarray(receiver_frame_1).save(f'{output_dir}/2_receiver_t-2.png')
    Image.fromarray(predicted_frame).save(f'{output_dir}/3_predicted_t.png')
    Image.fromarray(sender_frame).save(f'{output_dir}/4_sender_t_ground_truth.png')
    
    # Create difference images
    diff_pred_sender = np.abs(predicted_frame.astype(float) - sender_frame.astype(float))
    diff_pred_sender = (diff_pred_sender * 5).clip(0, 255).astype(np.uint8)  # Amplify for visibility
    
    diff_recv_sender = np.abs(receiver_frame_1.astype(float) - sender_frame.astype(float))
    diff_recv_sender = (diff_recv_sender * 5).clip(0, 255).astype(np.uint8)
    
    Image.fromarray(diff_pred_sender).save(f'{output_dir}/5_diff_predicted_vs_sender.png')
    Image.fromarray(diff_recv_sender).save(f'{output_dir}/6_diff_receiver_vs_sender.png')
    
    print(f"\n✓ Saved verification images to: {output_dir}/")
    print("  Check these images to visually verify prediction is working")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()





















