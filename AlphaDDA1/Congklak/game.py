#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from congklak import Congklak
from player import Random_player
from classical_MCTS import MCTS
from AlphaDDA1 import A_MCTS
from nn import NNetWrapper
from parameters import Parameters
from minimax import Minimax
import time

def play():
    # Ensure we look for checkpoints in the script's directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    params = Parameters()
    net = NNetWrapper(params=params)
    try:
        print(f"Checking for models in: {os.getcwd()}")
        # Try finding checkpoint.model or the latest numbered one
        checkpoint_file = "checkpoint.model"
        if not os.path.exists(checkpoint_file):
            checkpoints = [f for f in os.listdir(".") if f.startswith("checkpoint_") and f.endswith(".model")]
            if checkpoints:
                # Get the one with the highest number
                checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
                checkpoint_file = checkpoints[0]
                idx = int(checkpoint_file.split('_')[1].split('.')[0])
                net.load_checkpoint(idx)
                print(f"Loaded {checkpoint_file}")
            else:
                print("No checkpoint found. Using untrained network.")
        else:
            net.load_checkpoint()
            print("Loaded checkpoint.model")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Using untrained network.")

    g = Congklak()
    g.Ini_board()
    
    # Choose opponent
    print("\n" + "="*40)
    print("      CONGKLAK EVALUATION GAME")
    print("="*40)
    print("Select Opponent:")
    print(" 1: Random Player")
    print(" 2: Classical MCTS")
    print(" 3: Minimax (Fixed depth 3)")
    print(" 4: AlphaDDA1 (Dynamic Difficulty)")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
        if choice == '1': opp_type = "random"
        elif choice == '2': opp_type = "mcts"
        elif choice == '3': opp_type = "minimax"
        else: opp_type = "alphadda1"
    except (EOFError, ValueError):
        opp_type = "alphadda1"
        
    print(f"\n>> Playing vs {opp_type}")

    turn = 0
    while not g.Check_game_end():
        turn += 1
        g.Print_board()
        
        valid_moves = g.Get_valid_moves()
        current_side = 'P1' if g.current_player == 1 else 'P2'
        print(f"\nTurn {turn} - Player {current_side}")
        
        if g.current_player == 1:
            # Human Player
            print(f"Valid moves: {valid_moves}")
            move = -1
            while move not in valid_moves:
                try:
                    move_input = input("Enter your move (0-6): ")
                    move = int(move_input)
                except (ValueError, EOFError):
                    move = valid_moves[0]
                    print(f"Defaulting to move {move}")
                    break
        else:
            # AI Player
            print(f"{opp_type} is thinking...")
            start_time = time.time()
            if opp_type == "random":
                move = Random_player().action(g)
            elif opp_type == "mcts":
                move = ClassicalMCTS(g).Run()
            elif opp_type == "minimax":
                move = Minimax(g).Run()
            else:
                dda = A_MCTS(game=g, net=net, params=params)
                dda.num_moves = turn
                move = dda.Run()
            print(f"{opp_type} chose move {move} (Time: {time.time() - start_time:.2f}s)")
            
        g.Play_action(move)
        
    g.Print_board()
    winner = g.Get_winner()
    print("\n" + "="*40)
    if winner == 1:
        print("Winner: Player 1 (You!)")
    elif winner == -1:
        print(f"Winner: Player 2 ({opp_type})")
    else:
        print("It's a Draw!")
    print("="*40)

if __name__ == '__main__':
    play()
