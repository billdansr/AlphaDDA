#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import os
import sys
import numpy as np

from congklak import Congklak
from player import Random_player
from nn import NNetWrapper
from parameters import Parameters
from AlphaZero_mcts import A_MCTS as AlphaZeroMCTS
from minimax import Minimax

class AlphaZeroEvaluator():
    def __init__(self, num_games=20):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir:
            os.chdir(script_dir)

        self.num_games = num_games
        self.params = Parameters()
        self.params.opening = 0 # Force deterministic play for evaluation
        self.net = NNetWrapper(params=self.params)
        
        # Load checkpoint from command line argument if provided
        target_checkpoint = None
        if len(sys.argv) > 1:
            try:
                target_checkpoint = int(sys.argv[1])
            except ValueError:
                pass

        self.load_model(target_checkpoint)

    def load_model(self, checkpoint_idx):
        try:
            if checkpoint_idx is not None:
                self.net.load_checkpoint(checkpoint_idx)
                print(f"--- Validating AlphaZero Strength: Checkpoint {checkpoint_idx} ---")
            else:
                # Fallback: if checkpoint.model doesn't exist, look for the highest numbered one
                if not os.path.exists("checkpoint.model"):
                    checkpoints = [f for f in os.listdir(".") if f.startswith("checkpoint_") and f.endswith(".model")]
                    if checkpoints:
                        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
                        latest_idx = int(checkpoints[0].split('_')[1].split('.')[0])
                        self.net.load_checkpoint(latest_idx)
                        print(f"--- Validating AlphaZero Strength: Found latest numbered Checkpoint {latest_idx} ---")
                    else:
                        print("--- No checkpoints found. Using untrained network. ---")
                else:
                    self.net.load_checkpoint()
                    print("--- Validating AlphaZero Strength: Latest Checkpoint (checkpoint.model) ---")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using untrained network.")

    def play_game(self, p1_type, p2_type):
        g = Congklak()
        g.Ini_board()
        players = {self.params.p1: p1_type, self.params.p2: p2_type}
        turn = 0
        while not g.Check_game_end():
            turn += 1
            current_type = players[g.current_player]
            if current_type == "random":
                move = Random_player().action(g)
            elif current_type == "minimax":
                move = Minimax(g).Run()
            elif current_type == "alphazero":
                az = AlphaZeroMCTS(game=g, net=self.net, params=self.params)
                az.num_moves = turn
                move = az.Run()
            g.Play_action(move)
        return g.Get_winner()

    def evaluate(self, opponent_type):
        print(f"\nEvaluating AlphaZero vs {opponent_type} over {self.num_games} games...")
        wins, losses, draws = 0, 0, 0
        
        for i in range(self.num_games):
            # Alternate sides
            if i % 2 == 0:
                winner = self.play_game("alphazero", opponent_type)
                az_side = self.params.p1
            else:
                winner = self.play_game(opponent_type, "alphazero")
                az_side = self.params.p2

            if winner == az_side:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
            
            sys.stdout.write(f"\rGame {i+1}/{self.num_games} | Wins: {wins} Losses: {losses} Draws: {draws}")
            sys.stdout.flush()

        print(f"\nFinal Results vs {opponent_type}:")
        print(f"  Win Rate:  {(wins/self.num_games)*100:.1f}%")
        print(f"  Loss Rate: {(losses/self.num_games)*100:.1f}%")
        print(f"  Draw Rate: {(draws/self.num_games)*100:.1f}%")
        
        if wins > losses:
            print(f"  STATUS: AlphaZero is DOMINATING {opponent_type}.")
        else:
            print(f"  STATUS: AlphaZero is STRUGGLING against {opponent_type}.")

if __name__ == '__main__':
    # 20 games is usually enough to see if the model is "broken" or "trained"
    evaluator = AlphaZeroEvaluator(num_games=20)
    
    # Test 1: Sanity Check (Should be near 100% win rate)
    evaluator.evaluate("random")
    
    # Test 2: Strategic Check (Should ideally beat Minimax depth 3)
    evaluator.evaluate("minimax")