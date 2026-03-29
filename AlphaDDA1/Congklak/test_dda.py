#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from congklak import Congklak
from player import Random_player
from classical_MCTS import MCTS as ClassicalMCTS
from AlphaZero_mcts import A_MCTS as AlphaZeroMCTS
from AlphaDDA1 import A_MCTS as AlphaDDA1MCTS
from nn import NNetWrapper
from parameters import Parameters
from minimax import Minimax
import os
import time

class Evaluator():
    def __init__(self, num_games=10):
        # Ensure we look for checkpoints in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        self.num_games = num_games
        self.params = Parameters()
        self.net = NNetWrapper(params=self.params)
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
        players = {
            self.params.p1: p1_type,
            self.params.p2: p2_type
        }
        turn = 0
        while not g.Check_game_end():
            turn += 1
            current_player_type = players[g.current_player]
            if current_player_type == "random":
                move = Random_player().action(g)
            elif current_player_type == "mcts":
                move = ClassicalMCTS(g).Run()
            elif current_player_type == "minimax":
                move = Minimax(g).Run()
            elif current_player_type == "alphazero":
                az = AlphaZeroMCTS(game=g, net=self.net, params=self.params)
                az.num_moves = turn
                move = az.Run()
            elif current_player_type == "alphadda1":
                adda = AlphaDDA1MCTS(game=g, net=self.net, params=self.params)
                adda.num_moves = turn
                move = adda.Run()
            g.Play_action(move)
        return g.Get_winner()

    def evaluate(self, p1_type, p2_type):
        print(f"\nEvaluating {p1_type} vs {p2_type} ({self.num_games} games)...")
        results = {"win_p1": 0, "win_p2": 0, "draw": 0}
        for i in range(self.num_games):
            # Alternate who goes first
            if i % 2 == 0:
                p1, p2 = p1_type, p2_type
                p1_side = self.params.p1
                p2_side = self.params.p2
                is_p1_p1 = True
            else:
                p1, p2 = p2_type, p1_type
                p1_side = self.params.p2
                p2_side = self.params.p1
                is_p1_p1 = False
            
            winner = self.play_game(p1, p2)
            if winner == p1_side:
                results["win_p1"] += 1
            elif winner == p2_side:
                results["win_p2"] += 1
            else:
                results["draw"] += 1
            
            # Print intermediate progress
            print(f"Game {i+1}/{self.num_games}: Winner={winner} ({p1_type} was {'P1' if is_p1_p1 else 'P2'})")

        print(f"\nFinal Statistics for {p1_type} vs {p2_type}:")
        total = self.num_games
        w1 = results['win_p1']
        w2 = results['win_p2']
        d = results['draw']
        print(f"  {p1_type} Wins: {w1} ({w1/total*100:.1f}%)")
        print(f"  {p2_type} Wins: {w2} ({w2/total*100:.1f}%)")
        print(f"  Draws:      {d} ({d/total*100:.1f}%)")
        
        balance_metric = abs(50 - (w1/total*100))
        print(f"  Balance Metric (Target 50%): {balance_metric:.1f} offset")
        return results

if __name__ == '__main__':
    # Adjust num_games for better statistical significance (e.g., 20)
    # 4 games is just for a quick smoke test
    evaluator = Evaluator(num_games=4)
    
    opponents = ["random", "mcts", "minimax"]
    for opp in opponents:
        evaluator.evaluate("alphadda1", opp)
