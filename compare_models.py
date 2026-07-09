"""
Compare trained t+3 model vs pretrained k+1 model
Calculate PSNR and MSE metrics
"""
import os
import sys
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IFRVP'))

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_mse(img1, img2):
    """Calculate MSE between two images"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def extract_frames(video_path, max_frames=None, resize=None):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
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

def evaluate_model(model, frames, frames_tensor, delay_frames, start_frame, num_frames, device, model_name):
    """Evaluate model predictions"""
    psnr_sender_delayed = []
    psnr_sender_predicted = []
    mse_sender_delayed = []
    mse_sender_predicted = []
    
    model.eval()
    
    for i in range(num_frames):
        current_idx = start_frame + i
        delayed_idx = current_idx - delay_frames
        
        if delayed_idx < 1:
            continue
            
        sender_frame = frames[current_idx]
        receiver_frame = frames[delayed_idx]
        
        # Predict using recurrent approach
        prev_frame = frames_tensor[delayed_idx - 1].unsqueeze(0)
        curr_frame = frames_tensor[delayed_idx].unsqueeze(0)
        embt = torch.tensor([1.0]).view(1, 1, 1, 1).to(device)
        
        with torch.no_grad():
            for step in range(delay_frames):
                pred = model.inference(prev_frame, curr_frame, embt)
                prev_frame = curr_frame
                curr_frame = pred
        
        predicted_frame = tensor_to_frame(pred.squeeze(0))
        
        # Calculate metrics
        psnr_sd = calculate_psnr(sender_frame, receiver_frame)
        psnr_sp = calculate_psnr(sender_frame, predicted_frame)
        mse_sd = calculate_mse(sender_frame, receiver_frame)
        mse_sp = calculate_mse(sender_frame, predicted_frame)
        
        psnr_sender_delayed.append(psnr_sd)
        psnr_sender_predicted.append(psnr_sp)
        mse_sender_delayed.append(mse_sd)
        mse_sender_predicted.append(mse_sp)
    
    return {
        'model': model_name,
        'psnr_delayed': np.mean(psnr_sender_delayed),
        'psnr_predicted': np.mean(psnr_sender_predicted),
        'mse_delayed': np.mean(mse_sender_delayed),
        'mse_predicted': np.mean(mse_sender_predicted),
        'psnr_improvement': np.mean(psnr_sender_predicted) - np.mean(psnr_sender_delayed),
        'mse_reduction': (1 - np.mean(mse_sender_predicted) / np.mean(mse_sender_delayed)) * 100
    }

def main():
    print("="*70)
    print("Model Comparison: Trained t+3 vs Pretrained k+1")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    video_path = '../cam2_front.mp4'
    delay_frames = 3
    resize = (512, 512)
    start_frame = 10
    num_frames = 100
    
    # Extract frames
    print(f"\nExtracting frames from video...")
    max_frames = start_frame + num_frames + delay_frames + 10
    frames, fps = extract_frames(video_path, max_frames=max_frames, resize=resize)
    print(f"Extracted {len(frames)} frames at {fps} FPS")
    
    frames_tensor = frames_to_tensor(frames, device)
    
    results = []
    
    # Test 1: Trained t+3 model (IFRNet)
    print(f"\n--- Evaluating: Trained t+3 Model (IFRNet) ---")
    from models.IFRNet import Model as IFRNetModel
    model_trained = IFRNetModel().to(device)
    model_trained.load_state_dict(torch.load(
        'IFRVP/checkpoint_bdd100k_t3/IFRNet/2025-12-30_01-21-50/IFRNet_final.pth',
        map_location=device
    ), strict=False)
    
    result_trained = evaluate_model(
        model_trained, frames, frames_tensor, 
        delay_frames, start_frame, num_frames, device,
        "Trained t+3 (IFRNet)"
    )
    results.append(result_trained)
    del model_trained
    torch.cuda.empty_cache()
    
    # Test 2: Pretrained k+1 model (IFRNet_S)
    print(f"\n--- Evaluating: Pretrained k+1 Model (IFRNet_S) ---")
    from models.IFRNet_S import Model as IFRNetSModel
    model_pretrained = IFRNetSModel().to(device)
    model_pretrained.load_state_dict(torch.load(
        'IFRVP/IFRVP_k+1_laploss/IFRNet_S_latest.pth',
        map_location=device
    ), strict=False)
    
    result_pretrained = evaluate_model(
        model_pretrained, frames, frames_tensor,
        delay_frames, start_frame, num_frames, device,
        "Pretrained k+1 (IFRNet_S)"
    )
    results.append(result_pretrained)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"Test: {num_frames} frames, {delay_frames}-frame delay (200ms at 15fps)")
    print("-"*70)
    
    print(f"\n{'Metric':<25} {'Delayed':<15} {'Trained t+3':<15} {'Pretrained k+1':<15}")
    print("-"*70)
    
    print(f"{'PSNR (dB)':<25} {results[0]['psnr_delayed']:>12.2f} dB {results[0]['psnr_predicted']:>12.2f} dB {results[1]['psnr_predicted']:>12.2f} dB")
    print(f"{'MSE':<25} {results[0]['mse_delayed']:>15.1f} {results[0]['mse_predicted']:>15.1f} {results[1]['mse_predicted']:>15.1f}")
    
    print("\n" + "-"*70)
    print("IMPROVEMENT OVER DELAYED FRAME:")
    print("-"*70)
    print(f"{'Model':<30} {'PSNR Gain':<20} {'MSE Reduction':<20}")
    for r in results:
        print(f"{r['model']:<30} {r['psnr_improvement']:>+.2f} dB           {r['mse_reduction']:>+.1f}%")
    
    print("\n" + "-"*70)
    print("TRAINED vs PRETRAINED:")
    print("-"*70)
    psnr_diff = results[0]['psnr_predicted'] - results[1]['psnr_predicted']
    mse_diff = (1 - results[0]['mse_predicted'] / results[1]['mse_predicted']) * 100
    print(f"Trained model PSNR improvement: {psnr_diff:+.2f} dB")
    print(f"Trained model MSE reduction: {mse_diff:+.1f}%")
    
    if psnr_diff > 0:
        print(f"\n✓ Trained t+3 model is BETTER by {psnr_diff:.2f} dB PSNR")
    else:
        print(f"\n✗ Pretrained k+1 model is better by {-psnr_diff:.2f} dB PSNR")
    
    print("="*70)

if __name__ == '__main__':
    main()






