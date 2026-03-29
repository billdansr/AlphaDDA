#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
from congklak import Congklak
from AlphaZero_mcts import A_MCTS
from nn import NNetWrapper
from parameters import Parameters
import numpy as np
import sys

def play():
    # Ensure we look for checkpoints in the script's directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    params = Parameters()
    net = NNetWrapper(params=params)
    
    # Try finding checkpoint.model or the latest numbered one
    try:
        if len(sys.argv) > 1:
            try:
                checkpoint_idx = int(sys.argv[1])
                net.load_checkpoint(checkpoint_idx)
                print(f"Loaded 'checkpoint_{checkpoint_idx}.model'")
            except ValueError:
                print(f"Invalid index provided. Falling back to search.")
                raise Exception("Search fallback")
        else:
            checkpoint_file = "checkpoint.model"
            if not os.path.exists(checkpoint_file):
                checkpoints = [f for f in os.listdir(".") if f.startswith("checkpoint_") and f.endswith(".model")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
                    latest_checkpoint = checkpoints[0]
                    idx = int(latest_checkpoint.split('_')[1].split('.')[0])
                    net.load_checkpoint(idx)
                    print(f"Loaded {latest_checkpoint}")
                else:
                    print("No checkpoint found. AI will be untrained!")
            else:
                net.load_checkpoint()
                print("Loaded latest 'checkpoint.model'")
    except Exception as e:
        print(f"Using default 'checkpoint.model' or untrained network.")
        try:
            net.load_checkpoint()
        except:
            pass

    g = Congklak()
    g.Ini_board()
    
    print("\n" + "="*40)
    print("      WELCOME TO CONGKLAK AI")
    print("="*40)
    
    # Choose sides
    try:
        human_first = input("Do you want to go first (P1)? (y/n): ").lower().strip() == 'y'
    except EOFError:
        human_first = True
        
    human_player = 1 if human_first else -1
    
    while not g.Check_game_end():
        g.Print_board()
        
        if g.current_player == human_player:
            print(f"\n>> YOUR TURN (Player {'1' if human_player==1 else '2'})")
            valid_moves = g.Get_valid_moves()
            print(f"Valid holes: {valid_moves}")
            
            move = -1
            while move not in valid_moves:
                try:
                    move_in = input(f"Select a hole (0-6): ")
                    move = int(move_in)
                    if move not in valid_moves:
                        print("Invalid move. Choose a hole with shells in it.")
                except ValueError:
                    print("Please enter a number between 0 and 6.")
                except EOFError:
                    sys.exit(0)
        else:
            print(f"\n>> AI is thinking... (Iteration {checkpoint_idx if checkpoint_idx else 'Latest'})")
            # Increase simulation for better play
            az = A_MCTS(game=g, net=net, params=params)
            az.num_moves = 200 # Higher simulation for "expert" human play
            move = az.Run()
            print(f"AI selected hole: {move}")
            
        g.Play_action(move)
    
    g.Print_board()
    winner = g.Get_winner()
    print("\n" + "="*40)
    if winner == 0:
        print("      IT'S A DRAW!")
    elif winner == human_player:
        print("      CONGRATULATIONS! YOU WON!")
    else:
        print("      THE AI WON! REVENGE NEXT TIME?")
    print("="*40)

if __name__ == "__main__":
    play()
