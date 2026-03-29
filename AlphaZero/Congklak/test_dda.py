#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from congklak import Congklak
from player import Random_player
from classical_MCTS import MCTS as ClassicalMCTS
from AlphaZero_mcts import A_MCTS as AlphaZeroMCTS
from nn import NNetWrapper
import multiprocessing as mp
from parameters import Parameters
import time

class Evaluator():
    def __init__(self, num_games=10):
        # Ensure we look for checkpoints in the script's directory
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        self.num_games = num_games
        self.params = Parameters()
        self.net = NNetWrapper(params=self.params)
        # Try finding checkpoint.model or the latest numbered one
        try:
            checkpoint_file = "checkpoint.model"
            if not os.path.exists(checkpoint_file):
                checkpoints = [f for f in os.listdir(".") if f.startswith("checkpoint_") and f.endswith(".model")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
                    latest_checkpoint = checkpoints[0]
                    idx = int(latest_checkpoint.split('_')[1].split('.')[0])
                    self.net.load_checkpoint(idx)
                    print(f"Loaded {latest_checkpoint}")
                else:
                    print("No checkpoint found. Using untrained network.")
            else:
                self.net.load_checkpoint()
                print("Loaded latest checkpoint.model")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using untrained network.")

    def play_game(self, p1_type, p2_type):
        g = Congklak()
        g.Ini_board()
        
        # Initialize players
        players = {
            self.params.p1: p1_type,
            self.params.p2: p2_type
        }

        while not g.Check_game_end():
            current_player_type = players[g.current_player]
            
            if current_player_type == "random":
                move = Random_player().action(g)
            elif current_player_type == "mcts":
                move = ClassicalMCTS(g).Run()
            elif current_player_type == "alphazero":
                # For evaluation, we usually use higher simulation count or competitive settings
                az = AlphaZeroMCTS(game=g, net=self.net, params=self.params)
                az.num_moves = 100 # arbitrary large number to indicate competitive play
                move = az.Run()
            
            g.Play_action(move)
            
        return g.Get_winner()

    def evaluate(self, opponent="random"):
        print(f"Evaluating AlphaZero vs {opponent} ({self.num_games} games)...")
        # Use test opening (greedy) so evaluation is deterministic / competitive
        self.params.opening = self.params.opening_test
        
        results = {"win": 0, "loss": 0, "draw": 0}
        wins_as_p1, wins_as_p2 = 0, 0
        games_as_p1, games_as_p2 = 0, 0
        
        for i in range(self.num_games):
            # Alternate who goes first
            if i % 2 == 0:
                p1, p2 = "alphazero", opponent
                az_side = self.params.p1
                games_as_p1 += 1
            else:
                p1, p2 = opponent, "alphazero"
                az_side = self.params.p2
                games_as_p2 += 1
            
            winner = self.play_game(p1, p2)
            
            if winner == az_side:
                results["win"] += 1
                if az_side == self.params.p1:
                    wins_as_p1 += 1
                else:
                    wins_as_p2 += 1
            elif winner == 0:
                results["draw"] += 1
            else:
                results["loss"] += 1
            
            print(f"Game {i+1}/{self.num_games}: Winner={winner} (AZ was {'P1' if az_side==1 else 'P2'})")

        print(f"\nFinal Results vs {opponent}:")
        print(f"Wins:   {results['win']}  (as P1: {wins_as_p1}/{games_as_p1}, as P2: {wins_as_p2}/{games_as_p2})")
        print(f"Losses: {results['loss']}")
        print(f"Draws:  {results['draw']}")
        win_rate = (results['win'] + 0.5 * results['draw']) / self.num_games * 100
        print(f"Win Rate: {win_rate:.2f}%")
        return win_rate

if __name__ == '__main__':
    import sys
    
    # Check if a specific checkpoint was requested (e.g., python test_dda.py 65)
    checkpoint_idx = None
    if len(sys.argv) > 1:
        try:
            checkpoint_idx = int(sys.argv[1])
            print(f"Targeting checkpoint_{checkpoint_idx}.model")
        except ValueError:
            pass

    # Increase simulations for evaluation if needed
    evaluator = Evaluator(num_games=50)
    
    # Load the specific checkpoint if provided
    if checkpoint_idx is not None:
        try:
            evaluator.net.load_checkpoint(checkpoint_idx)
            print(f"Successfully loaded checkpoint_{checkpoint_idx}.model")
        except:
            print(f"Error: Could not find checkpoint_{checkpoint_idx}.model")
            sys.exit(1)
    
    # Test against Random
    evaluator.evaluate(opponent="random")
