#-------------------------------------------------------------------
# run_elo_tournament.py
# Perhitungan Elo Rating Riil Menggunakan Turnamen Round-Robin 50 Kali
#-------------------------------------------------------------------
import os
import sys
import numpy as np
import multiprocessing as mp
import csv
import argparse
from datetime import datetime
from test_dda import GridEvaluator

def calculate_elo_update(rating_a, rating_b, score_a, score_b, k_factor=8):
    """
    Menghitung update Elo rating untuk Pemain A dan Pemain B.
    score_a: poin aktual Pemain A (Menang=1.0, Seri=0.5, Kalah=0.0)
    score_b: poin aktual Pemain B (Menang=1.0, Seri=0.5, Kalah=0.0)
    k_factor: K-Factor (nilai default dari paper K=8)
    """
    # Menghitung probabilitas kemenangan masing-masing pemain (Persamaan 16)
    p_a_defeats_b = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    p_b_defeats_a = 1.0 - p_a_defeats_b
    
    # Update Elo rating (Persamaan 17 dengan NG = 1 game per pencocokan aktual)
    # Karena kita memperhitungkan score_a & score_b per single game secara sekuensial:
    new_rating_a = rating_a + k_factor * (score_a - p_a_defeats_b)
    new_rating_b = rating_b + k_factor * (score_b - p_b_defeats_a)
    
    return new_rating_a, new_rating_b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Spesifik nama file model, misalnya checkpoint_75.model', default=None)
    args = parser.parse_args()

    # Pastikan start method 'spawn' diaktifkan untuk kompabilitas CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Setup parameter pengujian
    NUM_TOURNAMENTS = 50
    K_FACTOR = 8
    INITIAL_ELO = 1500.0
    N_MAX = 400

    # --- Lokasi Model (bisa dari /content/ atau Drive) ---
    model_dir = "./"
    if os.path.exists('/content/AlphaDDA/AlphaZero/Congklak'):
        model_dir = '/content/AlphaDDA/AlphaZero/Congklak'
    elif os.path.exists('/content/AlphaDDA/AlphaDDA1/Congklak'):
        model_dir = '/content/AlphaDDA/AlphaDDA1/Congklak'
    elif os.path.exists('/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'):
        model_dir = '/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'
    elif os.path.exists('/content/drive/MyDrive/Colab Notebooks/AlphaDDA1/Congklak'):
        model_dir = '/content/drive/MyDrive/Colab Notebooks/AlphaDDA1/Congklak'

    # --- Lokasi Penyimpanan CSV (selalu prioritaskan Google Drive agar tidak hilang saat disconnect) ---
    save_dir = model_dir  # fallback ke model_dir jika Drive tidak ditemukan
    if os.path.exists('/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'):
        save_dir = '/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'
    elif os.path.exists('/content/drive/MyDrive/Colab Notebooks/AlphaDDA1/Congklak'):
        save_dir = '/content/drive/MyDrive/Colab Notebooks/AlphaDDA1/Congklak'

    print(f"Model dir : {model_dir}")
    print(f"Save dir  : {save_dir}")
    
    if args.model:
        model_path = os.path.join(model_dir, args.model)
        if not os.path.exists(model_path):
            print(f"Peringatan: {model_path} tidak ditemukan. Evaluator akan menggunakan model acak/untrained.")
        else:
            print(f"Menggunakan model spesifik: {model_path}")
    else:
        model_path = os.path.join(model_dir, "checkpoint.model")
        if not os.path.exists(model_path):
            checkpoints = [f for f in os.listdir(model_dir) if f.startswith("checkpoint_") and f.endswith(".model")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
                model_path = os.path.join(model_dir, checkpoints[0])
                print(f"Menggunakan model terbaru: {model_path}")
            else:
                print("Peringatan: checkpoint.model tidak ditemukan. Evaluator akan menggunakan model acak/untrained.")

    # Evaluator instance untuk menjalankan game
    evaluator = GridEvaluator(num_mean=1, N_MAX=N_MAX, model_path=model_path)

    # Daftar agent yang bertanding (HANYA 5 baseline agents sesuai paper - Tabel 4)
    # AlphaDDA1 TIDAK dimasukkan karena kekuatannya dinamis, bukan agen fixed-strength
    # Elo rating baseline ini kemudian digunakan sebagai referensi pada win-loss-draw evaluation
    agents = ["alphazero", "mcts", "minimax", "random"]
    
    # Inisialisasi Elo Rating ke 1,500 (seperti di paper)
    elo_ratings = {agent: INITIAL_ELO for agent in agents}
    
    csv_file = os.path.join(save_dir, "elo_tournament_results.csv")
    start_t = 1
    
    # Cek apakah ada progres turnamen sebelumnya untuk dilanjutkan
    if os.path.exists(csv_file):
        print(f"Ditemukan file {csv_file}. Melanjutkan progres turnamen...")
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row is not None:
                start_t = int(last_row["Tournament"]) + 1
                for agent in agents:
                    elo_ratings[agent] = float(last_row[agent])
        if start_t > NUM_TOURNAMENTS:
            print(f"Turnamen sudah selesai (telah mencapai {NUM_TOURNAMENTS}).")
            sys.exit(0)
    else:
        # Jika file belum ada, inisialisasi file CSV dengan header
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Tournament"] + agents)
            writer.writeheader()

    print("=" * 60)
    print(f"MEMULAI TURNAMEN ROUND-ROBIN ELO RATING (Congklak)")
    print(f"Mulai dari Turnamen: {start_t} / {NUM_TOURNAMENTS}")
    print(f"K-Factor         : {K_FACTOR}")
    print(f"Model checkpoint : {model_path}")
    print("=" * 60)

    # Siapkan list pasangan unik (A, B) untuk round-robin
    pairings = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            pairings.append((agents[i], agents[j]))

    # Kita jalankan turnamen
    history_ratings = []
    
    # Multiprocessing setup
    num_cores = mp.cpu_count()
    device = "cuda:0" if evaluator.play_single_game.__globals__['net_has_cuda']() else "cpu"

    for t in range(start_t, NUM_TOURNAMENTS + 1):
        print(f"\n--- Turnamen ke-{t}/{NUM_TOURNAMENTS} ---")
        
        # Buat schedule pertandingan untuk turnamen ini
        # Setiap pasangan bermain 2 games (1 kali A sebagai P1, 1 kali B sebagai P1)
        # Jadi total 20 games per turnamen
        schedule = []
        
        for (agent_a, agent_b) in pairings:
            # Game 1: A vs B (A is P1, B is P2)
            schedule.append((agent_a, agent_b, device, 1.5, -2.5)) # DDA parameters default
            
            # Game 2: B vs A (B is P1, A is P2)
            schedule.append((agent_b, agent_a, device, 1.5, -2.5))

        # Jalankan secara paralel di CPU/GPU
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(evaluator.play_single_game, schedule)

        # Proses hasil game secara berurutan dan update Elo
        print("  Memproses hasil game dan mengupdate rating Elo...")
        
        for p_idx, (agent_a, agent_b) in enumerate(pairings):
            # Ambil hasil dari Game 1 dan Game 2 untuk pairing ini
            # Game 1: A vs B (A=P1, B=P2) -> index = 2 * p_idx
            # Game 2: B vs A (B=P1, A=P2) -> index = 2 * p_idx + 1
            
            g1_winner, _, _ = results[2 * p_idx]
            g2_winner, _, _ = results[2 * p_idx + 1]
            
            # Game 1 (A is P1, B is P2)
            # Winner: 1 = P1 (A), -1 = P2 (B), 0 = Draw
            score_a_g1 = 1.0 if g1_winner == 1 else (0.5 if g1_winner == 0 else 0.0)
            score_b_g1 = 1.0 if g1_winner == -1 else (0.5 if g1_winner == 0 else 0.0)
            
            # Game 2 (B is P1, A is P2)
            # Winner: 1 = P1 (B), -1 = P2 (A), 0 = Draw
            score_a_g2 = 1.0 if g2_winner == -1 else (0.5 if g2_winner == 0 else 0.0)
            score_b_g2 = 1.0 if g2_winner == 1 else (0.5 if g2_winner == 0 else 0.0)
            
            # Total score over NG = 2 games
            total_score_a = score_a_g1 + score_a_g2
            total_score_b = score_b_g1 + score_b_g2
            
            # Update rating untuk pasangan ini
            rating_a = elo_ratings[agent_a]
            rating_b = elo_ratings[agent_b]
            
            p_a_defeats_b = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
            p_b_defeats_a = 1.0 - p_a_defeats_b
            
            new_rating_a = rating_a + K_FACTOR * (total_score_a - 2.0 * p_a_defeats_b)
            new_rating_b = rating_b + K_FACTOR * (total_score_b - 2.0 * p_b_defeats_a)
            
            elo_ratings[agent_a] = new_rating_a
            elo_ratings[agent_b] = new_rating_b

        # Print rating setelah turnamen ini selesai
        print("  Peringkat Elo saat ini:")
        for name, rating in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {name:12s} : {rating:.2f}")

        # Catat history dan append langsung ke CSV per turnamen
        row = {"Tournament": t}
        for agent in agents:
            row[agent] = elo_ratings[agent]
            
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Tournament"] + agents)
            writer.writerow(row)

    print("\n" + "=" * 60)
    print(f"TURNAMEN SELESAI!")
    print(f"Hasil akhir peringkat Elo:")
    for name, rating in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:12s} : {rating:.1f}")
    print(f"Data riwayat Elo disimpan ke {csv_file}")
    print("=" * 60)
