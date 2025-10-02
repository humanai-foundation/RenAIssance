import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run TrOCR training with EMA")
    parser.add_argument("--ema_decay", type=float, default=0.9999, 
                        help="Decay rate for EMA (0.9-0.9999)")
    parser.add_argument("--model_name", type=str, default="qantev/trocr-large-spanish",
                        help="Base model to use")
    parser.add_argument("--output_dir", type=str, 
                        default="../../../models/spanish_large_ema",
                        help="Output directory for the model")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the command to run the training script
    cmd = [
        "python", os.path.join(script_dir, "train_ema.py"),
        f"--ema_decay={args.ema_decay}",
        f"--model_name={args.model_name}",
        f"--model_output_dir={args.output_dir}",
        f"--batch_size={args.batch_size}",
        f"--learning_rate={args.learning_rate}",
        f"--epochs={args.epochs}"
    ]
    
    # Print the command
    print("Running command:", " ".join(cmd))
    
    # Execute the command
    os.system(" ".join(cmd))

if __name__ == "__main__":
    main()