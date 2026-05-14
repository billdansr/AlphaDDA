#---------------------------------------
# grid_search_dda.py
# Mencari kombinasi Window dan N_MAX terbaik untuk Win Rate 50%
#---------------------------------------
import os
import csv
import multiprocessing as mp
from test_dda import GridEvaluator

if __name__ == '__main__':
    # Konfigurasi Grid: Pergeseran Drastis untuk menyeimbangkan Bias Congklak
    # Perhitungan: A=1.0, X0=-2.5 memberikan ~40 sims di awal dan 300 sims saat v=0
    a_sim_list = [1.0, 1.5, 2.0]
    x0_list = [-2.5, -2.25, -2.0, -1.75]
    
    fixed_window = 1
    fixed_n_max = 300
    target_opponent = "alphazero" # Sesuai protokol Connect4 Fujita
    
    # 1. Setup Absolute Paths (Crucial for multiprocessing in Colab)
    base_path = "./"
    if os.path.exists('/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'):
        base_path = '/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'
        print(f"Colab detected. Using absolute path: {base_path}")

    csv_file = os.path.join(base_path, "grid_search_fujita.csv")
    model_path = os.path.join(base_path, "checkpoint.model")
    results_to_save = []

    # 1. Muat hasil yang sudah ada jika ada (Resume Capability)
    existing_configs = set()
    
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results_to_save.append(row)
                existing_configs.add((float(row['A_sim']), float(row['X0'])))
        print(f"--- Resuming Grid Search: {len(existing_configs)} combinations already done ---")

    # Setup Multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"--- Starting Fujita Grid Search ($A_{{sim}}$ vs $X_0$) ---")
    print(f"Fixed Params: Window={fixed_window}, $N_{{max}}$={fixed_n_max}")
    
    # Gunakan standar paper: 50 P1 + 50 P2 = 100 total per combo
    evaluator = GridEvaluator(num_mean=fixed_window, N_MAX=fixed_n_max, model_path=model_path)
    evaluator.num_games = 50

    for a_sim in a_sim_list:
        for x0 in x0_list:
            if (a_sim, x0) in existing_configs:
                continue
                
            print(f"Testing: $A_{{sim}}$={a_sim}, $X_0$={x0}...", end=" ", flush=True)
            win_rate, avg_margin = evaluator.run_bulk_test_custom("alphadda1", target_opponent, a_sim, x0)
            
            new_result = {
                "A_sim": a_sim,
                "X0": x0,
                "WinRate": win_rate,
                "AvgMargin": avg_margin,
                "DiffFrom50": abs(50.0 - win_rate)
            }
            results_to_save.append(new_result)

            # Simpan setiap iterasi (Auto-save)
            with open(csv_file, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["A_sim", "X0", "WinRate", "AvgMargin", "DiffFrom50"])
                writer.writeheader()
                writer.writerows(results_to_save)

    # Tentukan pemenang
    best_config = min(results_to_save, key=lambda x: x['DiffFrom50'])

    print(f"\n--- Grid Search Complete ---")
    print(f"BEST CONFIGURATION (Fujita Method):")
    print(f"A_sim: {best_config['A_sim']}")
    print(f"X0: {best_config['X0']}")
    print(f"WinRate: {best_config['WinRate']}%")
    print(f"Results saved to {csv_file}")
