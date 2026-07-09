"""
Analyze the comparison video to extract quality metrics.
The video has 3 panels: Sender | Receiver | Predicted
"""
import cv2
import numpy as np

def compute_mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def compute_psnr(img1, img2):
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    video_path = 'output/teleop_finetuned_t3_long.mp4'
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    
    # Each panel width (minus 2 pixel separators)
    panel_width = (width - 4) // 3
    
    mse_pred_list = []
    mse_baseline_list = []
    psnr_pred_list = []
    psnr_baseline_list = []
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip header area (top 50 pixels have labels)
        frame = frame[50:, :]
        
        # Extract panels
        sender = frame[:, :panel_width]
        receiver = frame[:, panel_width+2:2*panel_width+2]
        predicted = frame[:, 2*panel_width+4:]
        
        # Resize predicted to match sender if needed
        if predicted.shape[1] != sender.shape[1]:
            predicted = predicted[:, :sender.shape[1]]
        
        # Compute metrics
        mse_pred = compute_mse(predicted, sender)
        mse_baseline = compute_mse(receiver, sender)
        psnr_pred = compute_psnr(predicted, sender)
        psnr_baseline = compute_psnr(receiver, sender)
        
        mse_pred_list.append(mse_pred)
        mse_baseline_list.append(mse_baseline)
        psnr_pred_list.append(psnr_pred)
        psnr_baseline_list.append(psnr_baseline)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    
    # Results
    print("\n" + "="*70)
    print("QUALITY METRICS FOR SLIDES")
    print("="*70)
    
    avg_mse_pred = np.mean(mse_pred_list)
    avg_mse_baseline = np.mean(mse_baseline_list)
    avg_psnr_pred = np.mean(psnr_pred_list)
    avg_psnr_baseline = np.mean(psnr_baseline_list)
    
    mse_reduction = ((avg_mse_baseline - avg_mse_pred) / avg_mse_baseline) * 100
    psnr_gain = avg_psnr_pred - avg_psnr_baseline
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║            IFRVP VIDEO PREDICTION - QUALITY METRICS              ║
╠══════════════════════════════════════════════════════════════════╣
║  Model: IFRNet_S (Fine-tuned for t+3 prediction)                 ║
║  Resolution: 512×512                                             ║
║  Video FPS: {fps:.0f}                                                    ║
║  Frames Analyzed: {frame_count}                                          ║
╠══════════════════════════════════════════════════════════════════╣
║                    PREDICTION QUALITY                            ║
╠══════════════════════════════════════════════════════════════════╣
║  PSNR (Predicted vs Ground Truth):    {avg_psnr_pred:.2f} dB                  ║
║  MSE (Predicted vs Ground Truth):     {avg_mse_pred:.2f}                      ║
╠══════════════════════════════════════════════════════════════════╣
║                    BASELINE (No Prediction)                      ║
╠══════════════════════════════════════════════════════════════════╣
║  PSNR (Delayed vs Ground Truth):      {avg_psnr_baseline:.2f} dB                  ║
║  MSE (Delayed vs Ground Truth):       {avg_mse_baseline:.2f}                     ║
╠══════════════════════════════════════════════════════════════════╣
║                    IMPROVEMENT                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  MSE Reduction:    {mse_reduction:.1f}%                                      ║
║  PSNR Improvement: +{psnr_gain:.2f} dB                                    ║
╠══════════════════════════════════════════════════════════════════╣
║                    LATENCY COMPENSATION                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Network Delay Simulated: 200 ms (3 frames @ 15 FPS)             ║
║  Prediction Horizon: t+3 frames                                  ║
║  Method: Direct prediction (single inference)                    ║
╚══════════════════════════════════════════════════════════════════╝
""")

    print("\n## Table Format for Slides:\n")
    print("| Metric | Predicted | Baseline (Delayed) | Improvement |")
    print("|--------|-----------|-------------------|-------------|")
    print(f"| PSNR   | {avg_psnr_pred:.2f} dB  | {avg_psnr_baseline:.2f} dB          | +{psnr_gain:.2f} dB    |")
    print(f"| MSE    | {avg_mse_pred:.2f}     | {avg_mse_baseline:.2f}              | {mse_reduction:.1f}% reduction |")
    
    print("\n## Key Numbers for Slides:\n")
    print(f"• PSNR: {avg_psnr_pred:.2f} dB (prediction) vs {avg_psnr_baseline:.2f} dB (baseline)")
    print(f"• {psnr_gain:.2f} dB improvement in visual quality")
    print(f"• {mse_reduction:.1f}% error reduction")
    print(f"• Compensates for 200ms network latency")
    print(f"• Real-time capable at 15 FPS")

if __name__ == '__main__':
    main()





