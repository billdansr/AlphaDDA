#-------------------------------------------------------------------
# final_eval_congklak.py
# Pengujian Akhir AlphaDDA1 dengan Parameter Optimal (A=1.5, X0=-2.5)
#-------------------------------------------------------------------
import os
from test_dda import GridEvaluator

if __name__ == '__main__':
    # 1. Setup Path (Colab vs Local)
    base_path = "./"
    if os.path.exists('/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'):
        base_path = '/content/drive/MyDrive/Colab Notebooks/AlphaZero/Congklak'
        print(f"Colab detected. Using: {base_path}")

    model_path = os.path.join(base_path, "checkpoint.model")
    
    # 2. Parameter Terbaik Hasil Grid Search
    BEST_A = 1.5
    BEST_X0 = -2.5
    NUM_GAMES = 50 # Menaikkan jumlah gim agar data lebih solid untuk Bab IV
    
    opponents = ["random", "mcts", "alphazero"]
    
    print("="*50)
    print(f"STARTING FINAL EVALUATION")
    print(f"Params: A={BEST_A}, X0={BEST_X0}, N_max=300")
    print("="*50)

    for opp in opponents:
        print(f"\n--- Testing against: {opp.upper()} ---")
        
        # Inisialisasi Evaluator
        evaluator = GridEvaluator(
            model_path=model_path,
            target_opponent=opp,
            num_games=NUM_GAMES
        )
        
        # Jalankan Test
        win_rate, avg_margin = evaluator.evaluate(BEST_A, BEST_X0)
        
        print(f"RESULT vs {opp.upper()}:")
        print(f"Win Rate  : {win_rate}%")
        print(f"Avg Margin: {avg_margin}")
        print("-" * 30)

    print("\n[DONE] Data Final Evaluasi siap dimasukkan ke Tabel Skripsi.")
