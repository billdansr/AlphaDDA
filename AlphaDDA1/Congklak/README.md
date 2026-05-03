# AlphaDDA1 Congklak Research Tools

Repositori ini berisi implementasi resmi dan perangkat evaluasi untuk penelitian AlphaDDA1 pada permainan Congklak.

## Struktur File Utama

| File | Deskripsi |
| :--- | :--- |
| `AlphaDDA1.py` | Implementasi inti algoritma AlphaDDA1 (Fujita, 2022). |
| `test_dda.py` | Skrip evaluasi performa AI vs AI dengan ekspor hasil otomatis ke CSV. |
| `grid_search_dda.py` | Alat kalibrasi parameter $A_{sim}$ dan $X_0$ menggunakan metode *brute-force*. |
| `onnx_export_unity.py` | Skrip untuk mengonversi model PyTorch ke format ONNX untuk integrasi Unity. |

## Panduan Eksperimen

### 1. Kalibrasi Parameter (Grid Search)
Untuk mencari kombinasi parameter $A_{sim}$ dan $X_0$ yang paling seimbang (Win Rate ~50%):
```bash
python grid_search_dda.py
```
Hasil akan disimpan di `grid_search_fujita.csv`.

### 2. Evaluasi Performa Akhir
Setelah mendapatkan parameter terbaik, jalankan pengujian bulk melawan semua jenis lawan (Random, Minimax, MCTS, AlphaZero):
```bash
python test_dda.py [window] [n_max]
```
Contoh: `python test_dda.py 1 300`

### 3. Integrasi ke Unity (Android)
Untuk memperbarui otak AI di aplikasi Android:
1. Jalankan `python onnx_export_unity.py`.
2. Salin file `CongklakAlphaDDA.onnx` yang dihasilkan ke dalam folder `Assets/` di project Unity.

---
*Penelitian ini dikembangkan sebagai bagian dari tugas akhir/skripsi menggunakan metodologi Design Science Research Method (DSRM).*
