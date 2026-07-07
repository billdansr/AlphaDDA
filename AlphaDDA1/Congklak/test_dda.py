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
    def __init__(self, num_mean=1, N_MAX=300, model_path="checkpoint.model"):
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
            elif current_player_type == "minimax1":
                move = Minimax(g).Run()
            elif current_player_type == "mcts1":
                mcts_ai = ClassicalMCTS(g)
                mcts_ai.num_sim = 300
                move = mcts_ai.Run()
            elif current_player_type == "alphazero":
                # Ensure fair comparison with DDA1 max sims
                params.num_mcts_sims = self.N_MAX
                az = AlphaZeroMCTS(game=g, net=net, params=params)
                az.num_moves = turn
                move = az.Run()
            elif current_player_type == "alphadda1":
                # Use custom A and X0 if provided, else use defaults (Optimal for Congklak: A=1.5, X0=-2.5)
                a_val = custom_A if custom_A is not None else 1.5
                x0_val = custom_X0 if custom_X0 is not None else -2.5
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
        f_wins, f_draws, f_losses = 0, 0, 0
        s_wins, s_draws, s_losses = 0, 0, 0
        
        for i, (winner, p1_s, p2_s) in enumerate(results):
            if i % 2 == 0: # Target was P1 (First Player)
                if winner == self.params.p1:
                    f_wins += 1
                elif winner == 0:
                    f_draws += 1
                else:
                    f_losses += 1
            else: # Target was P2 (Second Player)
                if winner == self.params.p2:
                    s_wins += 1
                elif winner == 0:
                    s_draws += 1
                else:
                    s_losses += 1

        f_win_rate = f_wins / self.num_games
        f_loss_rate = f_losses / self.num_games
        f_draw_rate = f_draws / self.num_games

        s_win_rate = s_wins / self.num_games
        s_loss_rate = s_losses / self.num_games
        s_draw_rate = s_draws / self.num_games
        
        print(f"first: {self.num_mean} 1.5 -2.5 {self.N_MAX} {opponent_type} {f_wins} {f_losses} {f_draws}", flush=True)
        print(f"second: {self.num_mean} 1.5 -2.5 {self.N_MAX} {opponent_type} {s_wins} {s_losses} {s_draws}", flush=True)
        print(f"total: {self.num_mean} 1.5 -2.5 {self.N_MAX} {opponent_type} {f_wins+s_wins} {f_losses+s_losses} {f_draws+s_draws}", flush=True)

        return (f_win_rate, f_loss_rate, f_draw_rate), (s_win_rate, s_loss_rate, s_draw_rate)

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
        ai_draws = 0
        ai_losses = 0
        ai_margins = []
        win_margins = []
        loss_margins = []

        for i, (winner, p1_s, p2_s) in enumerate(results):
            if i % 2 == 0: 
                ai_side = self.params.p1
                margin = p1_s - p2_s
            else: 
                ai_side = self.params.p2
                margin = p2_s - p1_s
            
            ai_margins.append(margin)
            
            if winner == ai_side:
                ai_wins += 1
                win_margins.append(margin)
            elif winner == 0:
                ai_draws += 1
            else:
                ai_losses += 1
                loss_margins.append(margin)

        total_games = len(results)
        win_rate = (ai_wins / total_games) * 100 if total_games > 0 else 0.0
        draw_rate = (ai_draws / total_games) * 100 if total_games > 0 else 0.0
        loss_rate = (ai_losses / total_games) * 100 if total_games > 0 else 0.0
        avg_margin = sum(ai_margins) / total_games if total_games > 0 else 0.0
        
        avg_win_margin = sum(win_margins) / len(win_margins) if win_margins else 0
        avg_loss_margin = sum(loss_margins) / len(loss_margins) if loss_margins else 0

        print(f"RESULT vs {opponent_type.upper()} | W: {win_rate:.0f}% L: {loss_rate:.0f}% D: {draw_rate:.0f}%")
        print(f"  > Avg Total Margin: {avg_margin:+.2f}")
        print(f"  > Avg Win Margin  : {avg_win_margin:+.2f} (Forgiveness)")
        print(f"  > Avg Loss Margin : {avg_loss_margin:+.2f} (Opponent Power)")
        
        return win_rate, loss_rate, draw_rate, avg_margin

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

    # Deteksi Path Model (Colab vs Local)
    base_path = "./"
    if os.path.exists('/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'):
        base_path = '/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'

    # Cari model (Prioritas: checkpoint.model -> latest numbered checkpoint seperti checkpoint_75.model)
    model_path = os.path.join(base_path, "checkpoint.model")
    if not os.path.exists(model_path):
        checkpoints = [f for f in os.listdir(base_path) if f.startswith("checkpoint_") and f.endswith(".model")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            model_path = os.path.join(base_path, checkpoints[0])
            print(f"Model checkpoint.model tidak ditemukan. Menggunakan model terbaru: {model_path}")

    evaluator = GridEvaluator(num_mean=n_mean, N_MAX=n_max, model_path=model_path)
    
    # Run the evaluation pairings
    print(f"--- Starting Bulk Grid Evaluation (N_MAX={n_max}, Window={n_mean}) ---", flush=True)
    
    # Test against all opponents defined in the paper (filtered to MCTS1 and Minimax1 + others)
    opponents = ["alphazero", "mcts1", "minimax1", "random"]
    
    csv_file = os.path.join(base_path, "alphadda1_eval.csv")
    results_to_save = []
    
    # Cek apakah ada file resume
    existing_opps = set()
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results_to_save.append(row)
                existing_opps.add(row["Opponent"])
        print(f"Resume: Ditemukan {len(existing_opps)} lawan yang sudah diuji sebelumnya.")

    for opp in opponents:
        if opp in existing_opps:
            continue
            
        f_res, s_res = evaluator.run_bulk_test("alphadda1", opp)
        
        # Simpan hasil First dan Second
        results_to_save.append({
            "Opponent": opp,
            "F_WinRate": f"{f_res[0]:.2f}",
            "F_LossRate": f"{f_res[1]:.2f}",
            "F_DrawRate": f"{f_res[2]:.2f}",
            "S_WinRate": f"{s_res[0]:.2f}",
            "S_LossRate": f"{s_res[1]:.2f}",
            "S_DrawRate": f"{s_res[2]:.2f}"
        })

        # Auto-save setelah setiap opponent selesai (Mencegah hilang karena disconnect)
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Opponent", "F_WinRate", "F_LossRate", "F_DrawRate", "S_WinRate", "S_LossRate", "S_DrawRate"])
            writer.writeheader()
            writer.writerows(results_to_save)

    print(f"\n--- Evaluation Complete. Results saved to {csv_file} ---")
