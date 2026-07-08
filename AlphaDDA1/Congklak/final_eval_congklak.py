#-------------------------------------------------------------------
# final_eval_congklak.py
# Pengujian Akhir AlphaDDA1 dengan Parameter Optimal (A=1.5, X0=-2.5)
#-------------------------------------------------------------------
import os, csv, argparse, multiprocessing as mp
from datetime import datetime
from test_dda import GridEvaluator

if __name__ == '__main__':
    # Force 'spawn' start method for CUDA compatibility in multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Final Evaluation of AlphaDDA1 Congklak AI.")
    parser.add_argument('--A', type=float, help="Override sensitivity A. If not provided, loads from grid_search_fujita.csv.")
    parser.add_argument('--X0', type=float, help="Override offset X0. If not provided, loads from grid_search_fujita.csv.")
    parser.add_argument('--num_games', type=int, default=50, help="Number of games per pairing for evaluation.")
    
    args = parser.parse_args()

    NUM_GAMES = args.num_games

    # 1. Setup Path (Colab vs Local)
    base_path = "./"
    if os.path.exists('/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'):
        base_path = '/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'
        print(f"Colab detected. Using: {base_path}")

    model_path = os.path.join(base_path, "checkpoint.model")
    csv_results = os.path.join(base_path, "grid_search_fujita.csv")
    
    # Gunakan nilai dari argumen jika disediakan, jika tidak gunakan dari CSV
    if args.A is not None and args.X0 is not None:
        BEST_A = args.A
        BEST_X0 = args.X0
    else:
        # 2. Loading Parameter Terbaik secara Dinamis dari CSV
        if not os.path.exists(csv_results):
            print(f"Error: {csv_results} tidak ditemukan. Jalankan grid_search_dda.py terlebih dahulu atau sediakan --A dan --X0!")
            exit()

        grid_data = []
        with open(csv_results, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                grid_data.append(row)

        # Map DiffFrom50 and AvgMargin dynamically for robust loading
        for row in grid_data:
            # Fallback for DiffFrom50
            diff_key = 'MeanDiffFrom50' if 'MeanDiffFrom50' in row else ('DiffFrom50' if 'DiffFrom50' in row else None)
            row['computed_diff'] = float(row[diff_key]) if diff_key else 999.0
            
            # Fallback for AvgMargin
            # If we have specific opponent margins, compute their average
            m_keys = ["Margin_Random", "Margin_Minimax", "Margin_MCTS", "Margin_AlphaZero", "Margin_Minimax1", "Margin_MCTS1"]
            margins = []
            for k in m_keys:
                if k in row:
                    margins.append(float(row[k]))
            if not margins and 'AvgMargin' in row:
                margins.append(float(row['AvgMargin']))
                
            row['computed_margin'] = sum(margins) / len(margins) if margins else 0.0

        best_config = min(grid_data, key=lambda x: (x['computed_diff'], abs(x['computed_margin'] + 15.0)))
        
        BEST_A = float(best_config['A_sim'])
        BEST_X0 = float(best_config['X0'])
    
    # Tambahkan flag untuk menunjukkan apakah parameter di-override
    override_status = ""
    if args.A is not None or args.X0 is not None:
        override_status = " (OVERRIDDEN by args)"

    N_MAX = 300
    
    
    
    opponents = ["random", "minimax1", "mcts1", "alphazero"]
    
    print("="*50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"STARTING FINAL EVALUATION")
    print(f"Target Model: {model_path}")
    print(f"Params: A={BEST_A}, X0={BEST_X0}, N_max={N_MAX}{override_status}")
    print("="*50)

    results_to_save = [] # Inisialisasi di luar loop agar semua hasil tersimpan
    for opp in opponents:
        print(f"\n--- Testing against: {opp.upper()} ---")

        # Inisialisasi Evaluator
        evaluator = GridEvaluator(
            model_path=model_path,
            num_mean=1,
            N_MAX=N_MAX
        )
        evaluator.num_games = NUM_GAMES
        
        # Jalankan Test menggunakan parameter dinamis
        # run_bulk_test_custom sekarang mengembalikan win_rate, loss_rate, draw_rate, avg_margin
        win_rate, loss_rate, draw_rate, avg_margin = evaluator.run_bulk_test_custom("alphadda1", opp, BEST_A, BEST_X0)
        
        print(f"RESULT vs {opp.upper()}:")
        print(f"Win Rate  : {win_rate:.1f}%")
        print(f"Loss Rate : {loss_rate:.1f}%")
        print(f"Draw Rate : {draw_rate:.1f}%")
        print(f"Avg Margin: {avg_margin:+.2f}")
        print("-" * 30)

        results_to_save.append({
            "Opponent": opp,
            "A_sim": BEST_A,
            "X0": BEST_X0,
            "WinRate": f"{win_rate:.1f}%",
            "LossRate": f"{loss_rate:.1f}%",
            "DrawRate": f"{draw_rate:.1f}%",
            "AvgMargin": f"{avg_margin:+.2f}",
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    # Simpan hasil ke CSV
    csv_output_file = os.path.join(base_path, f"final_eval_A{BEST_A}_X0{BEST_X0}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Opponent", "A_sim", "X0", "WinRate", "LossRate", "DrawRate", "AvgMargin", "Timestamp"])
        writer.writeheader()
        writer.writerows(results_to_save)

    print(f"\n[DONE] Data Final Evaluasi disimpan ke {csv_output_file}. Siap dimasukkan ke Tabel Skripsi.")
