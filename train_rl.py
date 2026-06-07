"""
Orbit Wars - RL Agent Training Script
Train the hybrid RL agent on your laptop.
"""

import argparse
import sys
from rl_agent import RLTrainer, train_rl_agent

def parse_args():
    parser = argparse.ArgumentParser(description="Train Orbit Wars RL Agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--player", type=int, default=1, help="Player ID (0 or 1)")
    parser.add_argument("--opponent", type=str, default="easy", choices=["easy", "medium", "hard", "elite"], help="Opponent difficulty")
    parser.add_argument("--save", type=str, default="rl_model.pth", help="Model save path")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--buffer", type=int, default=10000, help="Replay buffer size")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("Orbit Wars - RL Agent Training")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Player: {args.player}")
    print(f"Opponent: {args.opponent}")
    print(f"Save path: {args.save}")
    print("=" * 60)
    
    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: PyTorch not installed. Install with: pip install torch")
        sys.exit(1)
    
    # Train the agent
    try:
        train_rl_agent(
            player_id=args.player,
            num_episodes=args.episodes,
            save_path=args.save,
            opponent_difficulty=args.opponent
        )
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Model saved to: {args.save}")
        print("=" * 60)
        print("\nTo use the trained agent:")
        print(f"  python main.py --difficulty rl --seed <seed>")
        print(f"  # Or in web app, select 'rl' difficulty")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
