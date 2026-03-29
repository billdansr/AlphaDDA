#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import torch
import torch.onnx
import os
import sys

# Ensure we can import nn and parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from nn import Net

def export():
    # Check if a specific checkpoint was requested (e.g., python onnx_export_unity.py 100)
    checkpoint_idx = None
    if len(sys.argv) > 1:
        try:
            checkpoint_idx = int(sys.argv[1])
            print(f"Targeting checkpoint_{checkpoint_idx}.model")
        except ValueError:
            pass

    # Determine filename
    if checkpoint_idx is not None:
        checkpoint_path = os.path.join(script_dir, f"checkpoint_{checkpoint_idx}.model")
    else:
        checkpoint_path = os.path.join(script_dir, "checkpoint.model")

    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Initialize network
    net = Net()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
        iteration = checkpoint.get('iteration', 'unknown')
        print(f"Loaded checkpoint from iteration {iteration}")
    else:
        net.load_state_dict(checkpoint)
        print("Loaded checkpoint (state_dict only format)")
        
    net.eval()

    # Create dummy input with the correct shape: (batch, channels, x, y) -> (1, 3, 2, 8)
    # 3 channels: Current Player, Opponent Player, Perspective
    dummy_input = torch.randn(1, 3, 2, 8)

    # Export to ONNX
    onnx_file = os.path.join(script_dir, "CongklakAlphaDDA.onnx")
    print(f"Exporting to {onnx_file}...")
    
    torch.onnx.export(
        net,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=15, # Unity 6 Sentis (Inference Engine) works well with opset 15+
        do_constant_folding=True,
        input_names=['input_board'],
        output_names=['output_pi', 'output_v'],
        # Dynamic batch size allows for flexible inference batching if ever needed
        dynamic_axes={
            'input_board': {0: 'batch_size'}, 
            'output_pi': {0: 'batch_size'}, 
            'output_v': {0: 'batch_size'}
        }
    )

    print(f"Successfully exported {onnx_file}")
    print("\nNext Steps for Unity:")
    print("1. Drag 'CongklakAlphaDDA.onnx' into your Unity Assets folder.")
    print("2. Ensure the 'Inference Engine' (Sentis) package is installed.")
    print("3. Use the ModelInference C# script to load and run this model.")

if __name__ == "__main__":
    export()
