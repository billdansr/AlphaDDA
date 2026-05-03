#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import os
import sys
import numpy as np
import multiprocessing as mp
import random
from statistics import mean

from congklak import Congklak
from player import Random_player
from classical_MCTS import MCTS as ClassicalMCTS
from nn import NNetWrapper
from parameters import Parameters

# These are the local experiment files
from AlphaZero_mcts import A_MCTS as AlphaZeroMCTS
from AlphaDDA1 import A_MCTS as AlphaDDA1MCTS
from minimax import Minimax

class GridEvaluator():
    def __init__(self, num_mean=5, N_MAX=800):
        self.params = Parameters()
        self.num_mean = num_mean
        self.N_MAX = N_MAX
        self.num_games = 50 # Games per pairing per side (Total 100) — matches paper's protocol
        
        # We initialize the net but we will reload it in each process for safety
        self.net = None 

    def play_single_game(self, args):
        """
        Worker function for multiprocessing.
        args: (p1_type, p2_type, device)
        """
        p1_type, p2_type, device = args
        
        # Local net initialization for the process
        params = Parameters()
        net = NNetWrapper(params=params, device=device)
        net.load_checkpoint()
        
        g = Congklak()
        g.Ini_board()
        players = {params.p1: p1_type, params.p2: p2_type}
        
        turn = 0
        while not g.Check_game_end():
            turn += 1
            current_player_type = players[g.current_player]
            
            if current_player_type == "random":
                move = Random_player().action(g)
            elif current_player_type == "minimax":
                move = Minimax(g).Run()
            elif current_player_type == "mcts":
                move = ClassicalMCTS(g).Run()
            elif current_player_type == "alphazero":
                # Ensure fair comparison with DDA1 max sims
                params.num_mcts_sims = self.N_MAX
                az = AlphaZeroMCTS(game=g, net=net, params=params)
                az.num_moves = turn
                move = az.Run()
            elif current_player_type == "alphadda1":
                # Paper defaults: A=1000, X0=0.0, num_mean=1, N_MAX=300
                adda = AlphaDDA1MCTS(game=g, net=net, params=params, num_mean=self.num_mean, A=10.0, X0=0.0, N_MAX=self.N_MAX)
                adda.num_moves = turn
                move = adda.Run()
            
            g.Play_action(move)
            
        winner = g.Get_winner()
        p1_score = g.board[7]
        p2_score = g.board[15]
        return winner, p1_score, p2_score

    def run_bulk_test(self, target_ai, opponent_type):
        print(f"Grid Test: {target_ai} vs {opponent_type}...", flush=True)
        
        # Prepare schedule: half as P1, half as P2
        schedule = []
        num_cores = mp.cpu_count()
        # Use available GPUs if possible, otherwise CPU
        device = "cuda:0" if net_has_cuda() else "cpu"
        
        for _ in range(self.num_games):
            schedule.append((target_ai, opponent_type, device)) # target is P1
            schedule.append((opponent_type, target_ai, device)) # target is P2

        # Execute in parallel
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(self.play_single_game, schedule)

        # Aggregate results
        ai_wins = 0
        ai_margins = []
        
        for i, (winner, p1_s, p2_s) in enumerate(results):
            if i % 2 == 0: # Target was P1
                ai_side = self.params.p1
                ai_margins.append(p1_s - p2_s)
            else: # Target was P2
                ai_side = self.params.p2
                ai_margins.append(p2_s - p1_s)
            
            if winner == ai_side:
                ai_wins += 1

        total_games = len(results)
        win_rate = (ai_wins / total_games) * 100
        avg_margin = sum(ai_margins) / len(ai_margins)
        
        print(f"RESULT | WinRate: {win_rate:.1f}% | AvgMargin: {avg_margin:+.2f}", flush=True)
        return win_rate, avg_margin

def net_has_cuda():
    import torch
    return torch.cuda.is_available()

if __name__ == '__main__':
    # Usage: python test_dda_grid.py [num_mean] [N_MAX]
    n_mean = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    
    # We must use 'spawn' for CUDA multiprocessing compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    evaluator = GridEvaluator(num_mean=n_mean, N_MAX=n_max)
    
    # Run the grid search pairings
    print(f"--- Starting Bulk Grid Evaluation (N_MAX={n_max}, Window={n_mean}) ---", flush=True)
    
    # Test against all 4 opponents defined in the paper
    evaluator.run_bulk_test("alphadda1", "random")
    evaluator.run_bulk_test("alphadda1", "minimax")
    evaluator.run_bulk_test("alphadda1", "mcts")
    evaluator.run_bulk_test("alphadda1", "alphazero")
