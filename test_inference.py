"""
Simple inference test script for IFRVP model.
Tests model loading and basic forward pass without requiring datasets.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

# Add IFRVP directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IFRVP'))

def create_dummy_frames(height=256, width=256):
    """Create dummy frames for testing"""
    # Create two simple test frames with gradients
    frame0 = np.zeros((height, width, 3), dtype=np.uint8)
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some pattern to frame0 (red gradient)
    for i in range(height):
        frame0[i, :, 0] = int(255 * i / height)
    
    # Add some pattern to frame1 (shifted red gradient)
    for i in range(height):
        frame1[i, :, 0] = int(255 * (i + 20) / height) % 255
    
    return frame0, frame1

def test_model_loading(model_path, model_name='IFRNet_S'):
    """Test loading the model"""
    print(f"\n{'='*60}")
    print(f"Testing IFRVP Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Import the model
    if model_name == 'IFRNet_S':
        from models.IFRNet_S import Model
    elif model_name == 'IFRNet':
        from models.IFRNet import Model
    elif model_name == 'IFRNet_L':
        from models.IFRNet_L import Model
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create model
    print("1. Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = Model().to(device)
    print(f"   ✓ Model created successfully")
    
    # Load weights if available
    if model_path and os.path.exists(model_path):
        print(f"\n2. Loading weights from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device)
            # Load with strict=False to ignore VGG loss weights (only used for training)
            model.load_state_dict(state_dict, strict=False)
            print(f"   ✓ Weights loaded successfully")
            print(f"   Note: VGG perceptual loss weights not loaded (only needed for training)")
        except Exception as e:
            print(f"   ✗ Failed to load weights: {e}")
            return None
    else:
        print(f"\n2. No pre-trained weights found at: {model_path}")
        print(f"   Using random initialization for testing")
    
    model.eval()
    return model, device

def test_inference(model, device):
    """Test inference with dummy data"""
    print(f"\n3. Testing inference...")
    
    # Create dummy frames
    frame0, frame1 = create_dummy_frames(256, 256)
    
    # Convert to torch tensors
    img0 = torch.from_numpy(frame0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    img0 = img0.to(device)
    img1 = img1.to(device)
    
    # Time embedding (predict middle frame, t=0.5)
    embt = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
    
    print(f"   Input shapes: img0={img0.shape}, img1={img1.shape}, embt={embt.shape}")
    
    # Run inference
    try:
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if start_time:
                start_time.record()
            
            output = model.inference(img0, img1, embt)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                print(f"   Inference time: {elapsed_time:.2f} ms")
            
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"   ✓ Inference successful!")
        
        return output
        
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_multiple_timesteps(model, device):
    """Test prediction at multiple timesteps"""
    print(f"\n4. Testing multiple timesteps...")
    
    frame0, frame1 = create_dummy_frames(256, 256)
    img0 = torch.from_numpy(frame0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img0 = img0.to(device)
    img1 = img1.to(device)
    
    timesteps = [0.25, 0.5, 0.75, 1.5, 2.0]  # Including extrapolation (>1.0)
    
    for t in timesteps:
        embt = torch.tensor([t]).view(1, 1, 1, 1).to(device)
        try:
            with torch.no_grad():
                output = model.inference(img0, img1, embt)
            status = "✓"
        except Exception as e:
            status = "✗"
            output = None
        
        if output is not None:
            print(f"   t={t:.2f}: {status} Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        else:
            print(f"   t={t:.2f}: {status} Failed")

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n5. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

def main():
    # Path to pre-trained model
    model_path = os.path.join(os.path.dirname(__file__), 
                              'IFRVP/IFRVP_k+1_laploss/IFRNet_S_latest.pth')
    
    # Test model loading
    result = test_model_loading(model_path, model_name='IFRNet_S')
    if result is None:
        print("\n✗ Model loading failed. Cannot proceed with testing.")
        return
    
    model, device = result
    
    # Count parameters
    count_parameters(model)
    
    # Test basic inference
    output = test_inference(model, device)
    
    if output is not None:
        # Test multiple timesteps
        test_multiple_timesteps(model, device)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()

