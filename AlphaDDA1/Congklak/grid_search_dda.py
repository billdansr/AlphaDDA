#---------------------------------------
# grid_search_dda.py
# Mencari kombinasi Window dan N_MAX terbaik untuk Win Rate 50%
#---------------------------------------
import os
import csv
import multiprocessing as mp
from test_dda import GridEvaluator

if __name__ == '__main__':
    # Konfigurasi Grid sesuai metodologi Fujita (AlphaDDA1)
    a_sim_list = [5.0, 10.0, 15.0, 20.0]
    x0_list = [-0.5, -0.25, 0.0, 0.25, 0.5]
    
    fixed_window = 1
    fixed_n_max = 300
    target_opponent = "minimax" # Fokus menyeimbangkan melawan Minimax
    
    # Setup Multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    results_to_save = []
    csv_file = "grid_search_fujita.csv"

    print(f"--- Starting Fujita Grid Search ($A_{{sim}}$ vs $X_0$) ---")
    print(f"Fixed Params: Window={fixed_window}, $N_{{max}}$={fixed_n_max}")
    print(f"Goal: Find WinRate closest to 50%\n")

    for a_sim in a_sim_list:
        for x0 in x0_list:
            print(f"Testing: $A_{{sim}}$={a_sim}, $X_0$={x0}...", end=" ", flush=True)
            
            # Kita buat evaluator dengan window dan n_max tetap
            evaluator = GridEvaluator(num_mean=fixed_window, N_MAX=fixed_n_max)
            
            # Modifikasi: Kita perlu test_dda menggunakan a_sim dan x0 yang sedang diuji
            # Kita akan mengupdate fungsi run_bulk_test untuk menerima a_sim dan x0
            win_rate, avg_margin = evaluator.run_bulk_test_custom("alphadda1", target_opponent, a_sim, x0)
            
            results_to_save.append({
                "A_sim": a_sim,
                "X0": x0,
                "WinRate": win_rate,
                "AvgMargin": avg_margin,
                "DiffFrom50": abs(50.0 - win_rate)
            })

    # Simpan hasil ke CSV
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
