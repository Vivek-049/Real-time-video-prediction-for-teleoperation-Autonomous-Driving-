"""
Compare pretrained k+1 model vs fine-tuned t+3 model performance.
"""
import os
import sys
import torch
import numpy as np
import cv2

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

def test_model(model, frames, delay, device, use_recurrent=False):
    """Test model on frames and return metrics"""
    receiver_frame_0 = frames[0]  # Frame at t-3
    receiver_frame_1 = frames[1]  # Frame at t-2
    sender_frame = frames[delay]  # Frame at t (what we're trying to predict)
    
    # Convert to tensors
    img0 = torch.from_numpy(receiver_frame_0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = torch.from_numpy(receiver_frame_1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img0 = img0.to(device)
    img1 = img1.to(device)
    
    with torch.no_grad():
        if use_recurrent:
            # Recurrent prediction for k+1 model
            prev_frame = img0
            curr_frame = img1
            embt = torch.tensor([1.0]).view(1, 1, 1, 1).to(device)
            
            for step in range(delay - 1):  # delay-1 because we start at t-2
                pred = model.inference(prev_frame, curr_frame, embt)
                prev_frame = curr_frame
                curr_frame = pred
            
            pred_tensor = pred
        else:
            # Direct prediction for fine-tuned t+3 model
            embt = torch.tensor([1.0]).view(1, 1, 1, 1).to(device)
            pred_tensor = model.inference(img0, img1, embt)
    
    predicted_frame = pred_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    predicted_frame = np.clip(predicted_frame * 255, 0, 255).astype(np.uint8)
    
    # Compute metrics
    mse_pred_sender = compute_mse(predicted_frame, sender_frame)
    psnr_pred_sender = compute_psnr(predicted_frame, sender_frame)
    
    mse_recv_sender = compute_mse(receiver_frame_1, sender_frame)
    psnr_recv_sender = compute_psnr(receiver_frame_1, sender_frame)
    
    return {
        'mse_pred': mse_pred_sender,
        'psnr_pred': psnr_pred_sender,
        'mse_recv': mse_recv_sender,
        'psnr_recv': psnr_recv_sender,
        'predicted_frame': predicted_frame,
        'sender_frame': sender_frame,
        'receiver_frame': receiver_frame_1
    }

def main():
    print("\n" + "="*70)
    print("COMPARISON: Pretrained k+1 vs Fine-tuned t+3 Model")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load both models
    from models.IFRNet_S import Model
    
    # Pretrained k+1 model
    print("\nLoading pretrained k+1 model...")
    model_pretrained = Model().to(device)
    pretrained_path = 'IFRVP/IFRVP_k+1_laploss/IFRNet_S_latest.pth'
    state_dict = torch.load(pretrained_path, map_location=device)
    model_pretrained.load_state_dict(state_dict, strict=False)
    model_pretrained.eval()
    print("✓ Pretrained model loaded")
    
    # Fine-tuned t+3 model
    print("\nLoading fine-tuned t+3 model...")
    model_finetuned = Model().to(device)
    finetuned_path = 'IFRVP/checkpoint_demo_finetune/IFRNet_S/2025-12-30_05-52-18/IFRNet_S_demo_finetuned.pth'
    state_dict = torch.load(finetuned_path, map_location=device)
    model_finetuned.load_state_dict(state_dict, strict=False)
    model_finetuned.eval()
    print("✓ Fine-tuned model loaded")
    
    # Test on demo video - first 40 seconds (600 frames at 15fps)
    video_path = '../demo videos/flir_camera_video_15fps (2).mp4'
    delay = 3
    
    # Test on multiple segments across first 40 seconds (0-600 frames)
    test_starts = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    
    pretrained_mses = []
    finetuned_mses = []
    pretrained_psnrs = []
    finetuned_psnrs = []
    baseline_mses = []
    baseline_psnrs = []
    
    print(f"\nTesting on {len(test_starts)} segments from: {video_path}")
    print(f"Delay: {delay} frames (200ms at 15fps)\n")
    
    for start_idx in test_starts:
        frames = extract_frames(video_path, start_idx, delay + 2, resize=(512, 512))
        if len(frames) < delay + 1:
            continue
        
        # Test pretrained with recurrent prediction
        result_pretrained = test_model(model_pretrained, frames, delay, device, use_recurrent=True)
        
        # Test fine-tuned with direct prediction
        result_finetuned = test_model(model_finetuned, frames, delay, device, use_recurrent=False)
        
        pretrained_mses.append(result_pretrained['mse_pred'])
        finetuned_mses.append(result_finetuned['mse_pred'])
        pretrained_psnrs.append(result_pretrained['psnr_pred'])
        finetuned_psnrs.append(result_finetuned['psnr_pred'])
        baseline_mses.append(result_pretrained['mse_recv'])
        baseline_psnrs.append(result_pretrained['psnr_recv'])
        
        print(f"Frame {start_idx}:")
        print(f"  Baseline (delayed):      MSE={result_pretrained['mse_recv']:.1f}, PSNR={result_pretrained['psnr_recv']:.2f} dB")
        print(f"  Pretrained (recurrent):  MSE={result_pretrained['mse_pred']:.1f}, PSNR={result_pretrained['psnr_pred']:.2f} dB")
        print(f"  Fine-tuned (direct t+3): MSE={result_finetuned['mse_pred']:.1f}, PSNR={result_finetuned['psnr_pred']:.2f} dB")
        print()
    
    # Summary
    print("="*70)
    print("SUMMARY (Average across all test segments)")
    print("="*70)
    
    avg_baseline_mse = np.mean(baseline_mses)
    avg_pretrained_mse = np.mean(pretrained_mses)
    avg_finetuned_mse = np.mean(finetuned_mses)
    
    avg_baseline_psnr = np.mean(baseline_psnrs)
    avg_pretrained_psnr = np.mean(pretrained_psnrs)
    avg_finetuned_psnr = np.mean(finetuned_psnrs)
    
    print(f"\nBaseline (no prediction - just use delayed frame):")
    print(f"  MSE:  {avg_baseline_mse:.1f}")
    print(f"  PSNR: {avg_baseline_psnr:.2f} dB")
    
    print(f"\nPretrained k+1 model (recurrent prediction):")
    print(f"  MSE:  {avg_pretrained_mse:.1f}")
    print(f"  PSNR: {avg_pretrained_psnr:.2f} dB")
    pretrained_improvement = ((avg_baseline_mse - avg_pretrained_mse) / avg_baseline_mse) * 100
    print(f"  Error reduction: {pretrained_improvement:.1f}%")
    
    print(f"\nFine-tuned t+3 model (direct prediction):")
    print(f"  MSE:  {avg_finetuned_mse:.1f}")
    print(f"  PSNR: {avg_finetuned_psnr:.2f} dB")
    finetuned_improvement = ((avg_baseline_mse - avg_finetuned_mse) / avg_baseline_mse) * 100
    print(f"  Error reduction: {finetuned_improvement:.1f}%")
    
    print(f"\n" + "-"*70)
    relative_improvement = ((avg_pretrained_mse - avg_finetuned_mse) / avg_pretrained_mse) * 100
    if avg_finetuned_mse < avg_pretrained_mse:
        print(f"✓ Fine-tuned model is {relative_improvement:.1f}% better than pretrained!")
    else:
        print(f"⚠ Pretrained model is {-relative_improvement:.1f}% better than fine-tuned")
    print("-"*70 + "\n")

if __name__ == '__main__':
    main()
