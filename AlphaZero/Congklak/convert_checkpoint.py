import torch
import sys
import os

def convert_checkpoint(input_path, output_path, iteration_num):
    """
    Converts an old-style PyTorch checkpoint (state_dict only) to the new format
    which includes an iteration number.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    if os.path.exists(output_path):
        overwrite = input(f"Warning: Output file '{output_path}' already exists. Overwrite? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("Conversion cancelled.")
            return

    print(f"Loading checkpoint from '{input_path}'...")
    try:
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

        state_dict = checkpoint

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print(f"Info: File is already in new format (Current Iteration: {checkpoint.get('iteration', 'Unknown')}).")
            print("Updating iteration number as requested...")
            state_dict = checkpoint['state_dict']
        elif not isinstance(checkpoint, dict):
             print(f"Error: The file '{input_path}' is not a valid dictionary/state_dict.")
             return

        print(f"Saving with new iteration number: {iteration_num}")
        new_checkpoint = {
            'state_dict': state_dict,
            'iteration': iteration_num
        }

        torch.save(new_checkpoint, output_path)
        print(f"Successfully saved new checkpoint to '{output_path}'.")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_checkpoint.py <input_model_path> <output_model_path> <iteration_number>")
        print("Example: python convert_checkpoint.py checkpoint_19mar.model checkpoint_19mar_iter0.model 0")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        iteration = int(sys.argv[3])
    except ValueError:
        print("Error: Iteration number must be an integer.")
        sys.exit(1)

    convert_checkpoint(input_file, output_file, iteration)
