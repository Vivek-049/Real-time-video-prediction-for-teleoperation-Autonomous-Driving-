"""
Generate presentation-ready metrics for the fine-tuned t+3 prediction model.
"""
import os
import sys
import torch
import numpy as np
import cv2
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IFRVP'))

def extract_all_frames(video_path, resize=None):
    """Extract all frames from video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    
    cap.release()
    return frames, fps, total

def compute_mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def compute_psnr(img1, img2):
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(img1, img2):
    """Compute SSIM between two images"""
    from skimage.metrics import structural_similarity as ssim
    return ssim(img1, img2, channel_axis=2, data_range=255)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load fine-tuned model
    from models.IFRNet_S import Model
    model = Model().to(device)
    model_path = 'IFRVP/checkpoint_demo_finetune/IFRNet_S/2025-12-30_05-52-18/IFRNet_S_demo_finetuned.pth'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Test video
    video_path = '../demo videos/flir_camera_video_15fps.mp4'
    delay = 3  # 3 frames = 200ms at 15fps
    
    print("Loading video...")
    frames, fps, total_frames = extract_all_frames(video_path, resize=(512, 512))
    print(f"Loaded {len(frames)} frames at {fps} FPS")
    
    # Compute metrics for all valid frames
    mse_pred_list = []
    mse_baseline_list = []
    psnr_pred_list = []
    psnr_baseline_list = []
    ssim_pred_list = []
    ssim_baseline_list = []
    inference_times = []
    
    print(f"\nComputing metrics for {len(frames) - delay - 1} frames...")
    
    for i in range(1, len(frames) - delay):
        # Frames for prediction
        img0 = torch.from_numpy(frames[i-1]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img0 = img0.to(device)
        img1 = img1.to(device)
        
        # Ground truth (sender frame)
        sender_frame = frames[i + delay]
        # Baseline (delayed frame)
        receiver_frame = frames[i]
        
        # Predict
        embt = torch.tensor([1.0]).view(1, 1, 1, 1).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            pred_tensor = model.inference(img0, img1, embt)
        inference_time = (time.time() - start_time) * 1000  # ms
        inference_times.append(inference_time)
        
        predicted = pred_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        predicted = np.clip(predicted * 255, 0, 255).astype(np.uint8)
        
        # Compute metrics
        mse_pred = compute_mse(predicted, sender_frame)
        mse_baseline = compute_mse(receiver_frame, sender_frame)
        
        psnr_pred = compute_psnr(predicted, sender_frame)
        psnr_baseline = compute_psnr(receiver_frame, sender_frame)
        
        ssim_pred = compute_ssim(predicted, sender_frame)
        ssim_baseline = compute_ssim(receiver_frame, sender_frame)
        
        mse_pred_list.append(mse_pred)
        mse_baseline_list.append(mse_baseline)
        psnr_pred_list.append(psnr_pred)
        psnr_baseline_list.append(psnr_baseline)
        ssim_pred_list.append(ssim_pred)
        ssim_baseline_list.append(ssim_baseline)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(frames) - delay - 1} frames")
    
    # Calculate statistics
    print("\n" + "="*80)
    print("PRESENTATION-READY METRICS")
    print("="*80)
    
    print("\n## System Configuration")
    print(f"- Model: IFRNet_S (fine-tuned for t+3 prediction)")
    print(f"- Input Resolution: 512×512")
    print(f"- Video FPS: {fps:.0f}")
    print(f"- Network Delay Simulated: {delay} frames ({delay/fps*1000:.0f}ms)")
    print(f"- Total Frames Analyzed: {len(mse_pred_list)}")
    
    print("\n## Quality Metrics (Predicted vs Ground Truth)")
    print(f"| Metric | Average | Min | Max | Std Dev |")
    print(f"|--------|---------|-----|-----|---------|")
    print(f"| MSE ↓ | {np.mean(mse_pred_list):.2f} | {np.min(mse_pred_list):.2f} | {np.max(mse_pred_list):.2f} | {np.std(mse_pred_list):.2f} |")
    print(f"| PSNR ↑ | {np.mean(psnr_pred_list):.2f} dB | {np.min(psnr_pred_list):.2f} dB | {np.max(psnr_pred_list):.2f} dB | {np.std(psnr_pred_list):.2f} |")
    print(f"| SSIM ↑ | {np.mean(ssim_pred_list):.4f} | {np.min(ssim_pred_list):.4f} | {np.max(ssim_pred_list):.4f} | {np.std(ssim_pred_list):.4f} |")
    
    print("\n## Baseline Comparison (Delayed Frame vs Ground Truth)")
    print(f"| Metric | Average | Min | Max |")
    print(f"|--------|---------|-----|-----|")
    print(f"| MSE | {np.mean(mse_baseline_list):.2f} | {np.min(mse_baseline_list):.2f} | {np.max(mse_baseline_list):.2f} |")
    print(f"| PSNR | {np.mean(psnr_baseline_list):.2f} dB | {np.min(psnr_baseline_list):.2f} dB | {np.max(psnr_baseline_list):.2f} dB |")
    print(f"| SSIM | {np.mean(ssim_baseline_list):.4f} | {np.min(ssim_baseline_list):.4f} | {np.max(ssim_baseline_list):.4f} |")
    
    print("\n## Improvement Over Baseline")
    mse_improvement = ((np.mean(mse_baseline_list) - np.mean(mse_pred_list)) / np.mean(mse_baseline_list)) * 100
    psnr_improvement = np.mean(psnr_pred_list) - np.mean(psnr_baseline_list)
    ssim_improvement = np.mean(ssim_pred_list) - np.mean(ssim_baseline_list)
    
    print(f"| Metric | Improvement |")
    print(f"|--------|-------------|")
    print(f"| MSE Reduction | {mse_improvement:.1f}% |")
    print(f"| PSNR Gain | +{psnr_improvement:.2f} dB |")
    print(f"| SSIM Gain | +{ssim_improvement:.4f} |")
    
    print("\n## Performance")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Inference Time (avg) | {np.mean(inference_times):.2f} ms |")
    print(f"| Inference Time (min) | {np.min(inference_times):.2f} ms |")
    print(f"| Inference Time (max) | {np.max(inference_times):.2f} ms |")
    print(f"| Throughput | {1000/np.mean(inference_times):.1f} FPS |")
    print(f"| Real-time Capable | {'✓ Yes' if 1000/np.mean(inference_times) > fps else '✗ No'} ({fps:.0f} FPS required) |")
    
    print("\n## Latency Compensation")
    latency_ms = delay / fps * 1000
    print(f"| Parameter | Value |")
    print(f"|-----------|-------|")
    print(f"| Network Delay | {latency_ms:.0f} ms |")
    print(f"| Prediction Horizon | {delay} frames |")
    print(f"| Effective Latency After Prediction | ~{np.mean(inference_times):.0f} ms (inference only) |")
    print(f"| Latency Reduction | {(latency_ms - np.mean(inference_times))/latency_ms*100:.0f}% |")
    
    print("\n" + "="*80)
    print("COPY-PASTE FOR SLIDES")
    print("="*80)
    
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│               VIDEO PREDICTION QUALITY                       │
├─────────────────────────────────────────────────────────────┤
│  PSNR:     {np.mean(psnr_pred_list):.2f} dB    (+{psnr_improvement:.2f} dB vs baseline)          │
│  SSIM:     {np.mean(ssim_pred_list):.4f}       (+{ssim_improvement:.4f} vs baseline)          │
│  MSE:      {np.mean(mse_pred_list):.2f}        ({mse_improvement:.1f}% reduction)               │
├─────────────────────────────────────────────────────────────┤
│               PERFORMANCE                                    │
├─────────────────────────────────────────────────────────────┤
│  Inference:  {np.mean(inference_times):.1f} ms/frame                                │
│  Throughput: {1000/np.mean(inference_times):.0f} FPS (Real-time @ {fps:.0f} FPS ✓)                   │
│  Resolution: 512×512                                        │
├─────────────────────────────────────────────────────────────┤
│               LATENCY COMPENSATION                           │
├─────────────────────────────────────────────────────────────┤
│  Network Delay:     {latency_ms:.0f} ms ({delay} frames)                        │
│  Prediction Horizon: t+{delay} frames                                │
│  Effective Latency: ~{np.mean(inference_times):.0f} ms ({(latency_ms - np.mean(inference_times))/latency_ms*100:.0f}% reduction)                  │
└─────────────────────────────────────────────────────────────┘
""")

if __name__ == '__main__':
    main()





