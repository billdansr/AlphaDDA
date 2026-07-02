#---------------------------------------
# grid_search_dda.py
# Mencari kombinasi Nh, A_sim, X0, dan N_max terbaik untuk Win Rate 50%
# Mengikuti metodologi Fujita (2022) Table A2 yang diadaptasi untuk Congklak
#---------------------------------------
import os
import csv
import multiprocessing as mp
from test_dda import GridEvaluator

if __name__ == '__main__':
    # Konfigurasi Grid: Mengikuti Table A2 Fujita (2022) dengan penyesuaian bias Congklak
    nh_list = [1, 2, 3, 4, 5]
    a_sim_list = [1.4, 1.6, 2.0, 2.4, 2.8]
    x0_list = [-1.8, -1.6, -1.5, -1.4, -1.3, -1.2]
    n_max_list = [200, 300, 400]
    
    target_opponent = "alphazero" # Sesuai protokol Connect4 Fujita
    
    # Setup Absolute Paths (Crucial for multiprocessing in Colab)
    base_path = "./"
    if os.path.exists('/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'):
        base_path = '/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'
        print(f"Colab detected. Using absolute path: {base_path}")

    csv_file = os.path.join(base_path, "grid_search_fujita.csv")
    model_path = os.path.join(base_path, "checkpoint.model")
    results_to_save = []

    # Muat hasil yang sudah ada jika ada (Resume Capability)
    existing_configs = set()
    
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results_to_save.append(row)
                existing_configs.add((int(row['Nh']), float(row['A_sim']), float(row['X0']), int(row['N_max'])))
        print(f"--- Resuming Grid Search: {len(existing_configs)} combinations already done ---")

    # Setup Multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"--- Starting Fujita Grid Search ($N_h$ vs $A_{{sim}}$ vs $X_0$ vs $N_{{max}}$) ---")
    
    # Gunakan standar paper: 50 P1 + 50 P2 = 100 total per combo
    evaluator = GridEvaluator(model_path=model_path)
    evaluator.num_games = 50

    for nh in nh_list:
        for a_sim in a_sim_list:
            for x0 in x0_list:
                for n_max in n_max_list:
                    if (nh, a_sim, x0, n_max) in existing_configs:
                        continue
                        
                    print(f"Testing: Nh={nh}, A_sim={a_sim}, X0={x0}, N_max={n_max}...", end=" ", flush=True)
                    
                    # Update evaluator parameters dynamically
                    evaluator.num_mean = nh
                    evaluator.N_MAX = n_max
                    
                    win_rate, loss_rate, draw_rate, avg_margin = evaluator.run_bulk_test_custom("alphadda1", target_opponent, a_sim, x0)
                    
                    new_result = {
                        "Nh": nh,
                        "A_sim": a_sim,
                        "X0": x0,
                        "N_max": n_max,
                        "WinRate": win_rate,
                        "AvgMargin": avg_margin,
                        "DiffFrom50": abs(50.0 - win_rate)
                    }
                    results_to_save.append(new_result)

                    # Simpan setiap iterasi (Auto-save)
                    with open(csv_file, mode='w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=["Nh", "A_sim", "X0", "N_max", "WinRate", "AvgMargin", "DiffFrom50"])
                        writer.writeheader()
                        writer.writerows(results_to_save)

    # Tentukan pemenang
    best_config = min(results_to_save, key=lambda x: float(x['DiffFrom50']))

    print(f"\n--- Grid Search Complete ---")
    print(f"BEST CONFIGURATION (Fujita Method):")
    print(f"Nh: {best_config['Nh']}")
    print(f"A_sim: {best_config['A_sim']}")
    print(f"X0: {best_config['X0']}")
    print(f"N_max: {best_config['N_max']}")
    print(f"WinRate: {best_config['WinRate']}%")
    print(f"Results saved to {csv_file}")

