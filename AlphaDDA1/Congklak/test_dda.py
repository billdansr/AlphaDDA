#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import os
import sys

from congklak import Congklak
from player import Random_player
from classical_MCTS import MCTS as ClassicalMCTS
from nn import NNetWrapper
from parameters import Parameters

# These are the local experiment files
from AlphaZero_mcts import A_MCTS as AlphaZeroMCTS
from AlphaDDA1 import A_MCTS as AlphaDDA1MCTS
from minimax import Minimax

class Evaluator():
    def __init__(self, num_games=10):
        # Ensure we look for checkpoints in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir:
            os.chdir(script_dir)

        self.num_games = num_games
        self.params = Parameters()
        self.net = NNetWrapper(params=self.params)
        
        # Load checkpoint
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
                    print("No checkpoint found in directory. Using untrained network.")
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
                # N_MAX=800 is the full strength from the paper.
                adda = AlphaDDA1MCTS(game=g, net=self.net, params=self.params, N_MAX=800)
                adda.num_moves = turn
                move = adda.Run()
            g.Play_action(move)
        
        # Return winner and stores for margin analysis
        return g.Get_winner(), g.board[7], g.board[15]

    def evaluate(self, target_ai, opponent_type):
        print(f"\nEvaluating {target_ai} vs {opponent_type} ({self.num_games} games)...")
        wins, losses, draws = 0, 0, 0
        p1_margins, p2_margins = [], []
        
        for i in range(self.num_games):
            # Alternate who goes first
            if i % 2 == 0:
                winner, s1, s2 = self.play_game(target_ai, opponent_type)
                ai_side = self.params.p1
                p1_margins.append(s1 - s2)
            else:
                winner, s1, s2 = self.play_game(opponent_type, target_ai)
                ai_side = self.params.p2
                p2_margins.append(s2 - s1) # Margin relative to the target_ai
            
            if winner == ai_side:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
            
            sys.stdout.write(f"\rGame {i+1}/{self.num_games} | Wins: {wins} Losses: {losses} Draws: {draws}")
            sys.stdout.flush()

        avg_p1_margin = sum(p1_margins) / len(p1_margins) if p1_margins else 0
        avg_p2_margin = sum(p2_margins) / len(p2_margins) if p2_margins else 0

        print(f"\nFinal Statistics for {target_ai} vs {opponent_type}:")
        print(f"  Win Rate:  {(wins/self.num_games)*100:.1f}%")
        print(f"  Avg Margin as P1: {avg_p1_margin:+.1f}")
        print(f"  Avg Margin as P2: {avg_p2_margin:+.1f}")
        
        balance_metric = abs(50 - (wins/self.num_games*100))
        print(f"  DDA Balance Offset (Target 50% Win Rate): {balance_metric:.1f}%")
        return wins, losses, draws

if __name__ == '__main__':
    # 20 games for a statistically meaningful test of the DDA balance
    evaluator = Evaluator(num_games=20)
    
    # We test both AlphaZero (Fixed Strength) and AlphaDDA1 (Adaptive)
    # to compare how much closer DDA gets to the 50% "Fun" target.
    test_types = ["alphazero", "alphadda1"]
    opponents = ["random", "minimax"]
    
    for ai in test_types:
        for opp in opponents:
            evaluator.evaluate(ai, opp)
