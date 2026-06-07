#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import os
import sys
import numpy as np
import multiprocessing as mp
import random
import csv
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
    def __init__(self, num_mean=5, N_MAX=800, model_path="checkpoint.model"):
        self.params = Parameters()
        self.num_mean = num_mean
        self.N_MAX = N_MAX
        self.num_games = 50 # Games per pairing per side (Total 100) — matches paper's protocol
        self.model_path = model_path
        
        # We initialize the net but we will reload it in each process for safety
        self.net = None 

    def play_single_game(self, args):
        """
        Worker function for multiprocessing.
        args: (p1_type, p2_type, device, custom_A, custom_X0)
        """
        p1_type, p2_type, device, custom_A, custom_X0 = args
        
        # Local net initialization for the process
        params = Parameters()
        net = NNetWrapper(params=params, device=device)
        
        # Load from absolute path if provided
        if os.path.exists(self.model_path):
            import torch
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                net.net.load_state_dict(checkpoint['state_dict'])
            else:
                net.net.load_state_dict(checkpoint)
            net.net.eval()
        else:
            net.load_checkpoint()
        
        g = Congklak()
        g.Ini_board()
        players = {params.p1: p1_type, params.p2: p2_type}
        
        turn = 0
        while not g.Check_game_end():
            if len(g.Get_valid_moves()) == 0:
                g.current_player *= -1
                continue
                
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
                # Use custom A and X0 if provided, else use defaults (Optimal for Congklak: A=1.0, X0=-2.0)
                a_val = custom_A if custom_A is not None else 1.0
                x0_val = custom_X0 if custom_X0 is not None else -2.0
                adda = AlphaDDA1MCTS(game=g, net=net, params=params, num_mean=self.num_mean, A=a_val, X0=x0_val, N_MAX=self.N_MAX)
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
            schedule.append((target_ai, opponent_type, device, None, None)) # target is P1
            schedule.append((opponent_type, target_ai, device, None, None)) # target is P2

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

    def run_bulk_test_custom(self, target_ai, opponent_type, a_val, x0_val):
        """ Specialized version for Grid Search with custom A and X0 """
        schedule = []
        num_cores = mp.cpu_count()
        device = "cuda:0" if net_has_cuda() else "cpu"
        
        for _ in range(self.num_games):
            schedule.append((target_ai, opponent_type, device, a_val, x0_val)) # target is P1
            schedule.append((opponent_type, target_ai, device, a_val, x0_val)) # target is P2

        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(self.play_single_game, schedule)

        ai_wins = 0
        ai_margins = []
        for i, (winner, p1_s, p2_s) in enumerate(results):
            if i % 2 == 0: ai_side = self.params.p1; ai_margins.append(p1_s - p2_s)
            else: ai_side = self.params.p2; ai_margins.append(p2_s - p1_s)
            if winner == ai_side: ai_wins += 1

        win_rate = (ai_wins / len(results)) * 100
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
    opponents = ["random", "minimax", "mcts", "alphazero"]
    results_to_save = []

    for opp in opponents:
        win_rate, avg_margin = evaluator.run_bulk_test("alphadda1", opp)
        results_to_save.append({
            "Opponent": opp,
            "WinRate": f"{win_rate:.1f}%",
            "AvgMargin": f"{avg_margin:+.2f}"
        })

    # Save to CSV for thesis reporting
    csv_file = "alphadda1_eval.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Opponent", "WinRate", "AvgMargin"])
        writer.writeheader()
        writer.writerows(results_to_save)

    print(f"\n--- Evaluation Complete. Results saved to {csv_file} ---")
