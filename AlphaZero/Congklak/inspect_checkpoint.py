import torch
import sys
import os

def check_checkpoint(file_path):
    """
    Loads a PyTorch checkpoint file and prints its iteration number.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Inspecting '{file_path}'...")
    try:
        # Load the checkpoint on CPU to avoid CUDA requirements just for checking
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Check structure based on nn.py save_checkpoint format
        if isinstance(checkpoint, dict) and 'iteration' in checkpoint:
            print(f"  -> Iteration: {checkpoint['iteration']}")
        elif isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
            # Likely an old format where the file is just the state_dict
            print("  -> Format: Old (Weight Dictionary only).")
            print("  -> Iteration: Unknown (treated as 0 by load_checkpoint).")
        else:
            print("  -> Format: Dictionary with weights but no iteration key (Legacy).")

    except Exception as e:
        print(f"  -> Error reading file: {e}")

if __name__ == "__main__":
    # Default to 'checkpoint.model' if no argument is provided
    target_file = sys.argv[1] if len(sys.argv) > 1 else "checkpoint.model"
    check_checkpoint(target_file)