"""
DEBUG: Thorough verification that prediction is using correct inputs
and the pretrained model is working properly.
"""
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IFRVP'))

def main():
    print("\n" + "="*80)
    print("DEBUG VERIFICATION - Checking if model is working correctly")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    from models.IFRNet_S import Model
    model = Model().to(device)
    model_path = 'IFRVP/IFRVP_k+1_laploss/IFRNet_S_latest.pth'
    
    print(f"\n1. Loading pretrained model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    
    # Check what keys are in the state dict
    print(f"   State dict has {len(state_dict)} keys")
    print(f"   First 5 keys: {list(state_dict.keys())[:5]}")
    
    # Load and check for missing/unexpected keys
    missing, unexpected = [], []
    model_state = model.state_dict()
    for k in model_state.keys():
        if k not in state_dict and 'vgg' not in k.lower():
            missing.append(k)
    for k in state_dict.keys():
        if k not in model_state:
            unexpected.append(k)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"   Missing keys (excluding VGG): {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"   Missing: {missing[:5]}...")
    print("   ✓ Model loaded\n")
    
    # Extract frames
    print("2. Extracting test frames...")
    cap = cv2.VideoCapture('../cam2_front.mp4')
    frames = []
    for i in range(30):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512))
            frames.append(frame)
    cap.release()
    print(f"   Extracted {len(frames)} frames\n")
    
    # Test scenario
    delay = 3
    current_idx = 20  # Sender's current frame
    
    print("3. Test scenario:")
    print(f"   Sender frame index: {current_idx}")
    print(f"   Receiver frame index: {current_idx - delay} (delayed by {delay} frames)")
    print(f"   Model input frames: {current_idx - delay - 1}, {current_idx - delay}")
    print(f"   Expected prediction target: frame {current_idx}\n")
    
    # Get specific frames
    sender_frame = frames[current_idx]
    receiver_frame = frames[current_idx - delay]
    input_frame_0 = frames[current_idx - delay - 1]
    input_frame_1 = frames[current_idx - delay]
    
    print("4. Frame index verification:")
    print(f"   sender_frame = frames[{current_idx}]")
    print(f"   receiver_frame = frames[{current_idx - delay}]")
    print(f"   input_frame_0 (img0) = frames[{current_idx - delay - 1}]")
    print(f"   input_frame_1 (img1) = frames[{current_idx - delay}]")
    
    # Check if frames are different
    print("\n5. Checking if input frames are different from sender:")
    mse_input0_sender = np.mean((input_frame_0.astype(float) - sender_frame.astype(float))**2)
    mse_input1_sender = np.mean((input_frame_1.astype(float) - sender_frame.astype(float))**2)
    mse_input0_input1 = np.mean((input_frame_0.astype(float) - input_frame_1.astype(float))**2)
    
    print(f"   MSE(input_0, sender): {mse_input0_sender:.2f}")
    print(f"   MSE(input_1, sender): {mse_input1_sender:.2f}")
    print(f"   MSE(input_0, input_1): {mse_input0_input1:.2f}")
    
    if mse_input0_sender < 1 or mse_input1_sender < 1:
        print("   ❌ WARNING: Input frames are nearly identical to sender!")
        print("      This would mean prediction is trivial (no motion)")
    else:
        print("   ✓ Input frames are different from sender (motion present)")
    
    # Convert to tensors
    img0 = torch.from_numpy(input_frame_0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = torch.from_numpy(input_frame_1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img0 = img0.to(device)
    img1 = img1.to(device)
    
    print("\n6. Running model inference:")
    print(f"   img0 shape: {img0.shape}")
    print(f"   img1 shape: {img1.shape}")
    print(f"   embt value: {delay} (predict {delay} frames ahead)")
    
    embt = torch.tensor([float(delay)]).view(1, 1, 1, 1).to(device)
    
    with torch.no_grad():
        pred_tensor = model.inference(img0, img1, embt)
    
    print(f"   Output shape: {pred_tensor.shape}")
    print(f"   Output range: [{pred_tensor.min():.3f}, {pred_tensor.max():.3f}]")
    
    predicted_frame = pred_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    predicted_frame = np.clip(predicted_frame * 255, 0, 255).astype(np.uint8)
    
    # Verify prediction is different from inputs
    print("\n7. Checking if prediction is different from inputs:")
    mse_pred_input0 = np.mean((predicted_frame.astype(float) - input_frame_0.astype(float))**2)
    mse_pred_input1 = np.mean((predicted_frame.astype(float) - input_frame_1.astype(float))**2)
    mse_pred_sender = np.mean((predicted_frame.astype(float) - sender_frame.astype(float))**2)
    
    print(f"   MSE(predicted, input_0): {mse_pred_input0:.2f}")
    print(f"   MSE(predicted, input_1): {mse_pred_input1:.2f}")
    print(f"   MSE(predicted, sender):  {mse_pred_sender:.2f}")
    
    if mse_pred_input0 < 1 and mse_pred_input1 < 1:
        print("   ❌ PROBLEM: Prediction is identical to inputs!")
        print("      Model might not be extrapolating properly")
    elif mse_pred_sender < 1:
        print("   ❌ PROBLEM: Prediction is identical to sender!")
        print("      This would indicate a bug - using sender as prediction")
    else:
        print("   ✓ Prediction is different from both inputs")
    
    # Check if prediction is closer to sender than input
    print("\n8. Quality assessment:")
    print(f"   Input_1 → Sender MSE: {mse_input1_sender:.2f}")
    print(f"   Predicted → Sender MSE: {mse_pred_sender:.2f}")
    
    if mse_pred_sender < mse_input1_sender:
        improvement = (1 - mse_pred_sender / mse_input1_sender) * 100
        print(f"   ✓ Prediction is {improvement:.1f}% closer to sender than delayed frame")
        print("   ✓ Model is successfully predicting forward!")
    else:
        print("   ⚠ Prediction is not closer to sender than delayed frame")
        print("   Model might not be working optimally for this video")
    
    # Save debug images
    print("\n9. Saving debug images...")
    output_dir = 'output/debug'
    os.makedirs(output_dir, exist_ok=True)
    
    Image.fromarray(input_frame_0).save(f'{output_dir}/01_input_frame_0_idx{current_idx-delay-1}.png')
    Image.fromarray(input_frame_1).save(f'{output_dir}/02_input_frame_1_idx{current_idx-delay}.png')
    Image.fromarray(predicted_frame).save(f'{output_dir}/03_predicted.png')
    Image.fromarray(sender_frame).save(f'{output_dir}/04_sender_ground_truth_idx{current_idx}.png')
    Image.fromarray(receiver_frame).save(f'{output_dir}/05_receiver_delayed_idx{current_idx-delay}.png')
    
    # Create side-by-side comparison
    comparison = np.hstack([input_frame_0, input_frame_1, predicted_frame, sender_frame])
    Image.fromarray(comparison).save(f'{output_dir}/06_comparison_inputs_pred_sender.png')
    
    # Create difference images (amplified)
    diff_pred_sender = np.abs(predicted_frame.astype(float) - sender_frame.astype(float))
    diff_pred_sender = np.clip(diff_pred_sender * 10, 0, 255).astype(np.uint8)
    
    diff_input1_sender = np.abs(input_frame_1.astype(float) - sender_frame.astype(float))
    diff_input1_sender = np.clip(diff_input1_sender * 10, 0, 255).astype(np.uint8)
    
    Image.fromarray(diff_pred_sender).save(f'{output_dir}/07_diff_predicted_vs_sender_x10.png')
    Image.fromarray(diff_input1_sender).save(f'{output_dir}/08_diff_input1_vs_sender_x10.png')
    
    print(f"   Saved to: {output_dir}/")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
    Model Path: {model_path}
    Model Loaded: ✓ (with {len(state_dict)} parameters)
    
    Input Frames Used:
      - img0: frame {current_idx - delay - 1}
      - img1: frame {current_idx - delay}
    
    Prediction Target: frame {current_idx}
    
    MSE Results:
      - Input_1 → Sender: {mse_input1_sender:.2f}
      - Predicted → Sender: {mse_pred_sender:.2f}
      - Improvement: {(1 - mse_pred_sender/mse_input1_sender)*100:.1f}%
    
    Conclusion: {"Model IS working correctly!" if mse_pred_sender < mse_input1_sender else "Check the model/video"}
    """)
    
    print("Check the debug images in output/debug/ to visually verify!\n")

if __name__ == '__main__':
    main()



















