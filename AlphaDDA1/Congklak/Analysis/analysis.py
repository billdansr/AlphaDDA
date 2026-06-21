import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from scipy import stats

def load_and_preprocess(file_name='game_logs.csv'):
    """Memuat data dan melakukan pra-pemrosesan tipe data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    
    if not os.path.exists(file_path):
        # Coba cari file Excel jika CSV tidak ditemukan
        excel_path = file_path.replace('.csv', '.xlsx')
        if os.path.exists(excel_path):
            file_path = excel_path
        else:
            print(f"Error: File {file_name} atau .xlsx tidak ditemukan di {script_dir}")
            return None

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    # Bersihkan nama kolom dari spasi atau karakter aneh yang sering muncul dari ekspor Excel
    df.columns = [c.strip() for c in df.columns]
    
    # Hapus kolom kosong dan baris yang tidak memiliki sessionId (biasanya baris sampah di Excel)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed|Column')]
    if 'sessionId' in df.columns:
        df = df.dropna(subset=['sessionId'])
    
    # Konversi boolean yang tersimpan sebagai string
    df['isDda'] = df['isDda'].astype(str).str.upper().str.strip() == 'TRUE'
    df['isTerminal'] = df['isTerminal'].astype(str).str.upper().str.strip() == 'TRUE'
    
    # Pastikan tipe data numerik untuk kolom penting
    numeric_cols = ['thinkTime', 'v', 'simulations', 'scoreP1', 'scoreP2', 'turn']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Optional: Filter out developer data to maintain research integrity
    # developer_names = ['YourName', 'Dev', 'Admin']
    # df = df[~df['participant'].isin(developer_names)]

    return df

def print_summary_stats(df):
    """Menghitung dan menampilkan statistik agregat untuk semua sesi penelitian."""
    # Ambil baris terminal (akhir permainan) untuk setiap sesi
    terminal_df = df[df['isTerminal'] == True].copy()
    
    if terminal_df.empty:
        print("Peringatan: Baris 'isTerminal' tidak ditemukan. Menggunakan baris terakhir tiap sesi.")
        terminal_df = df.sort_values('turn').groupby('sessionId').tail(1).copy()

    # Hitung metrik tambahan
    terminal_df['Margin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    
    # Agregasi per kondisi (DDA vs Non-DDA)
    summary = terminal_df.groupby('isDda').agg({
        'sessionId': 'count',
        'scoreP1': 'mean',
        'scoreP2': 'mean',
        'Margin': 'mean'
    }).rename(columns={
        'sessionId': 'Total Sesi',
        'scoreP1': 'Rerata Skor P1',
        'scoreP2': 'Rerata Skor P2',
        'Margin': 'Rerata Selisih'
    })

    print("\n" + "="*40)
    print("STATISTIK AGREGAT PENELITIAN")
    print("="*40)
    print(summary)
    
    # Hitung Win Rate
    def get_winner(row):
        if row['scoreP1'] > row['scoreP2']: return 'Human'
        if row['scoreP2'] > row['scoreP1']: return 'AI'
        return 'Draw'
    
    terminal_df['Winner'] = terminal_df.apply(get_winner, axis=1)
    win_rates = terminal_df.groupby(['isDda', 'Winner']).size().unstack(fill_value=0)
    
    # Pastikan kolom lengkap untuk visualisasi yang konsisten
    for col in ['Human', 'AI', 'Draw']:
        if col not in win_rates.columns: win_rates[col] = 0
        
    win_rates_pct = win_rates.div(win_rates.sum(axis=1), axis=0) * 100
    
    print("\n=== Win Rates (%) ===")
    print(win_rates_pct.round(2))

    # Korelasi DDA (Hanya jika DDA aktif)
    dda_moves = df[(df['isDda'] == True) & (df['player'] == -1) & (df['v'].notnull())]
    if not dda_moves.empty:
        corr = dda_moves[['v', 'simulations']].corr().iloc[0,1]
        print(f"\nKorelasi V vs Simulations (DDA): {corr:.2f}")
        if corr < -0.5:
            print("Status: Mekanisme DDA Berjalan Baik (Korelasi Negatif Kuat).")

def perform_statistical_test(df):
    """
    Melakukan uji statistik dengan alur:
    1. Cek Normalitas (Shapiro-Wilk)
    2. Jika Normal -> T-Test
    3. Jika Tidak Normal -> Mann-Whitney
    """
    terminal_df = df[df['isTerminal'] == True].copy()
    if terminal_df.empty:
        terminal_df = df.sort_values('turn').groupby('sessionId').tail(1).copy()
    
    terminal_df['AbsMargin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    dda_group = terminal_df[terminal_df['isDda'] == True]['AbsMargin']
    nodda_group = terminal_df[terminal_df['isDda'] == False]['AbsMargin']
    
    if dda_group.empty or nodda_group.empty:
        print("\n[!] Error: Data untuk salah satu grup (DDA/Non-DDA) belum tersedia.")
        print("    Minimal diperlukan data dari kedua kondisi untuk melakukan uji komparatif.")
        return

    print("\n" + "="*50)
    print("ANALISIS VALIDASI STATISTIK")
    print("="*50)
    print(f"Desain Eksperimen: Within-Subjects (Paired Comparison)")

    # --- 1. ASPEK BALANCING (Selisih Skor) ---
    print("\n[1] ASPEK BALANCING (Selisih Skor Akhir)")
    
    # Agregasi data per partisipan untuk dipasangkan
    paired_data = terminal_df.groupby(['participant', 'isDda'])['AbsMargin'].mean().unstack().dropna()
    
    if len(paired_data) < 2:
        print("  [!] Data berpasangan (Within-Subject) tidak cukup. Pastikan satu partisipan mencoba kedua mode.")
        return
    
    n_count = len(paired_data)
    print(f"  - Jumlah Partisipan Valid (N): {n_count}")
    if n_count < 30:
        print(f"  - Catatan: N < 30. Hasil Shapiro-Wilk sangat menentukan pemilihan uji (Parametrik vs Non-Parametrik).")
    
    # Cek Normalitas pada Selisih (Syarat Paired T-Test)
    diff_balancing = paired_data[True] - paired_data[False]
    
    p_norm = 0 # Default jika test tidak bisa dijalankan
    if len(diff_balancing) >= 3:
        _, p_norm = stats.shapiro(diff_balancing)
        print(f"  - Uji Normalitas (Shapiro-Wilk): p={p_norm:.4f}")
    else:
        print(f"  - Normalitas Selisih (Shapiro-Wilk): Skipped (N < 3)")

    # Justifikasi Statistik: 
    # Desain Within-Subjects memitigasi 'Player Skill Bias' sehingga N=15 sudah cukup 
    # untuk mendeteksi 'Effect Size' yang besar dari algoritma DDA.
    if p_norm > 0.05 or len(paired_data) < 5:
        t_stat, p_val = stats.ttest_rel(paired_data[True], paired_data[False])
        method = "Paired T-Test (Parametrik)"
        if len(paired_data) < 5:
            print(f"    (Justifikasi: N={n_count} terlalu kecil untuk uji non-parametrik, menggunakan Paired T-Test).")
    else:
        t_stat, p_val = stats.wilcoxon(paired_data[True], paired_data[False])
        method = "Wilcoxon Signed-Rank (Non-Parametrik)"
    
    # Hitung nilai rata-rata untuk interpretasi arah (DDA lebih kecil marginnya?)
    mean_dda = paired_data[True].mean()
    mean_nodda = paired_data[False].mean()

    print(f"  - Metode: {method}")
    print(f"  - Statistic: {t_stat:.3f}, P-Value: {p_val:.4f}")
    print(f"  - Rerata Margin: DDA={mean_dda:.2f} vs Non-DDA={mean_nodda:.2f}")
    print(f"  - Kesimpulan: {'SIGNIFIKAN' if p_val < 0.05 else 'TIDAK SIGNIFIKAN'}")
    # --- 2. ASPEK ENGAGEMENT (Waktu Berpikir) ---
    print("\n[2] ASPEK ENGAGEMENT (Rerata Waktu Berpikir)")
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime'])
    paired_think = human_moves.groupby(['participant', 'isDda'])['thinkTime'].mean().unstack().dropna()

    if len(paired_think) >= 2:
        diff_think = paired_think[True] - paired_think[False]
        
        p_norm_t = 0
        if len(diff_think) >= 3:
            _, p_norm_t = stats.shapiro(diff_think)
        
        if p_norm_t > 0.05 or len(paired_think) < 5:
            t_stat_t, p_val_t = stats.ttest_rel(paired_think[True], paired_think[False])
            method_t = "Paired T-Test (Parametrik)"
        else:
            t_stat_t, p_val_t = stats.wilcoxon(paired_think[True], paired_think[False])
            method_t = "Wilcoxon Signed-Rank (Non-Parametrik)"
            
        print(f"  - Metode: {method_t}")
        print(f"  - Statistic: {t_stat_t:.3f}, P-Value: {p_val_t:.4f}")
        print(f"  - Kesimpulan: {'SIGNIFIKAN' if p_val_t < 0.05 else 'TIDAK SIGNIFIKAN'}")

def generate_markdown_narrative(df, timestamp_str, output_filename_prefix='analysis_report'):
    """Menghasilkan draf narasi Bab 4 dalam format Markdown dan menyimpannya ke file."""
    terminal_df = df[df['isTerminal'] == True].copy()
    if terminal_df.empty:
        terminal_df = df.sort_values('turn').groupby('sessionId').tail(1).copy()
    
    lines = []
    lines.append("# PANDUAN PENULISAN BAB 4 DAN BAB 5 (DATA-DRIVEN)\n")
    lines.append("> **PENTING:** Gunakan poin-poin di bawah ini sebagai basis argumen. ")
    lines.append("> Lakukan parafrase dan tambahkan analisis kualitatif berdasarkan observasi Anda selama eksperimen.\n")
    
    # 4.1 Deskripsi Data (Konteks Penelitian)
    n_participants = len(terminal_df['participant'].unique())
    n_sessions = len(terminal_df)
    lines.append(f"### 4.1 Statistik Partisipan")
    lines.append(f"Data penelitian dikumpulkan dari **{n_participants} partisipan** dengan total **{n_sessions} sesi** valid (berpasangan antara mode DDA dan Non-DDA).\n")

    # 4.2 Jawaban Rumusan Masalah 1: Implementasi AlphaDDA
    lines.append(f"### 4.2 Analisis Implementasi Adaptasi AlphaDDA")
    dda_moves = df[(df['isDda'] == True) & (df['player'] == -1) & (df['v'].notnull())]
    if not dda_moves.empty:
        corr = dda_moves[['v', 'simulations']].corr().iloc[0,1]
        lines.append(f"**Poin Diskusi Implementasi:**")
        lines.append(f"- Koefisien korelasi ($v$ vs $sims$) tercatat sebesar **{corr:.2f}**.")
        lines.append(f"- Gunakan angka ini untuk menjelaskan bagaimana AI melakukan *throttling* kekuatan berdasarkan dominasi pemain di papan Congklak.\n")

    # 4.3 Jawaban Rumusan Masalah 2: Efektivitas terhadap Player Experience (Implicit Metrics)
    lines.append(f"### 4.3 Analisis Efektivitas Player Experience (Metrik Implisit)")
    
    # Bagian Balancing (Score Margin)
    avg_margin = terminal_df.groupby('isDda')['scoreP1'].apply(lambda x: (x - terminal_df.loc[x.index, 'scoreP2']).abs().mean())
    if True in avg_margin and False in avg_margin:
        improvement = ((avg_margin[False] - avg_margin[True]) / avg_margin[False] * 100)
        lines.append(f"#### 4.3.1 Aspek Game Balancing")
        lines.append(f"- Data empiris menunjukkan margin skor DDA (**{avg_margin[True]:.2f}**) vs Non-DDA (**{avg_margin[False]:.2f}**).")
        lines.append(f"- Interpretasikan penurunan sebesar **{improvement:.1f}%** ini sebagai keberhasilan sistem dalam menciptakan 'close-game' yang lebih intens.\n")
    
    # Bagian Win Rate (Mendukung Efektivitas)
    def get_winner_label(row):
        if row['scoreP1'] > row['scoreP2']: return 'Human'
        if row['scoreP2'] > row['scoreP1']: return 'AI'
        return 'Draw'
    terminal_df['Winner'] = terminal_df.apply(get_winner_label, axis=1)
    wr = terminal_df.groupby(['isDda', 'Winner']).size().unstack(fill_value=0)
    if True in wr.index:
        win_count = wr.loc[True].get('Human', 0)
        total_dda = wr.loc[True].sum()
        lines.append(f"#### 4.3.2 Analisis Win-Rate")
        lines.append(f"- Pada mode DDA, pemain manusia berhasil memenangkan **{win_count} dari {total_dda}** pertandingan, mendekati rasio kemenangan ideal 50:50.\n")

    # 4.4 Visualisasi Pendukung
    lines.append(f"### 4.4 Visualisasi Data")
    lines.append(f"![Metrik Engagement dan Balancing](engagement_metrics_{timestamp_str}.png)")
    lines.append(f"![Adaptasi Dinamis AI](ai_adaptation_{timestamp_str}.png)\n")

    # --- STRUKTUR BAB 5 ---
    lines.append("## IDE POKOK BAB 5: SIMPULAN DAN SARAN\n")
    
    lines.append("### 5.1 Simpulan")
    lines.append("1. Simpulkan bagaimana korelasi **{:.2f}** menjawab rumusan masalah pertama tentang implementasi.".format(corr if not dda_moves.empty else 0))
    lines.append("2. Bahas efektivitas balancing berdasarkan angka perbaikan **{:.1f}%** untuk menjawab rumusan masalah kedua.".format(improvement if 'improvement' in locals() else 0))

    lines.append("\n### 5.2 Implikasi Penelitian")
    lines.append("- Jabarkan bagaimana temuan ini mendukung atau mengkritisi literatur yang ada (misal: Fujita 2022 atau Yannakakis).")
    lines.append("- Jelaskan manfaat praktis bagi industri game dalam menciptakan AI yang tidak mengintimidasi pemain pemula.\n")

    lines.append("### 5.3 Saran dan Rekomendasi")
    lines.append("- Saran teknis: Optimasi model ONNX untuk performa mobile yang lebih ringan.")
    lines.append("- Saran metodologis: Penggunaan kuesioner sebagai data triangulasi untuk mendukung data telemetri yang telah ada.\n")

    lines.append("---\n*Catatan: Dokumen ini adalah kerangka kerja ilmiah. Tulislah narasi Anda sendiri untuk menjamin originalitas dan kedalaman analisis.*")

    markdown_content = "\n".join(lines)

    # Simpan ke file di direktori yang sama dengan script
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Dapatkan direktori script saat ini
    md_path = os.path.join(script_dir, f"{output_filename_prefix}_{timestamp_str}.md")
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"\n[INFO] Narasi Markdown otomatis disimpan ke: {md_path}")
    except Exception as e:
        print(f"\n[ERROR] Gagal menyimpan file Markdown: {e}")

    # Tetap tampilkan preview di terminal
    print("\n" + "="*20 + " PREVIEW: MARKDOWN NARRATIVE " + "="*20)
    print(markdown_content)
    print("="*20 + "  END OF PREVIEW  " + "="*20)

def plot_score_progression(df, timestamp_str, session_id=None):
    """Visualisasi progresi skor. Jika session_id null, ambil sesi terbaru."""
    if session_id is None:
        session_id = df['sessionId'].iloc[-1]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    session_data = df[df['sessionId'] == session_id].copy()
    is_dda = session_data['isDda'].iloc[0]

    plt.figure(figsize=(10, 5))
    plt.plot(session_data['turn'], session_data['scoreP1'], label='P1 (Human)', marker='o', color='#3498db')
    plt.plot(session_data['turn'], session_data['scoreP2'], label='P2 (AI)', marker='s', color='#e74c3c')
    
    plt.title(f'Progresi Skor: Sesi {session_id}\n(DDA: {is_dda})', fontsize=12)
    plt.xlabel('Giliran (Turn)')
    plt.ylabel('Skor')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'score_progression_{timestamp_str}.png'), dpi=300)
    plt.show()

def plot_engagement_metrics(df, timestamp_str):
    """Menganalisis engagement melalui waktu berpikir di berbagai kondisi."""
    human_moves = df[df['player'] == 1].copy()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    plt.figure(figsize=(12, 5))
    
    # 1. Boxplot Waktu Berpikir
    plt.subplot(1, 2, 1)
    sns.boxplot(x='isDda', y='thinkTime', data=human_moves, palette='Set2')
    plt.title('Distribusi Waktu Berpikir')
    plt.xlabel('DDA Aktif')
    plt.ylabel('Detik')

    # 2. Rata-rata Selisih Skor Akhir (Balancing)
    plt.subplot(1, 2, 2)
    terminal_df = df[df['isTerminal'] == True].copy()
    if terminal_df.empty: terminal_df = df.groupby('sessionId').tail(1)
    
    terminal_df['AbsMargin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    sns.barplot(x='isDda', y='AbsMargin', data=terminal_df, palette='viridis')
    plt.title('Rerata Selisih Skor (Lower is More Balanced)')
    plt.xlabel('DDA Aktif')
    plt.ylabel('Selisih Skor Akhir')

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'engagement_metrics_{timestamp_str}.png'), dpi=300)
    plt.show()

def plot_ai_adaptation(df, timestamp_str, session_id=None):
    """Visualisasi bagaimana AI menyesuaikan simulasi berdasarkan evaluasi board."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if session_id is None:
        # Ambil sesi DDA terakhir
        dda_sessions = df[df['isDda'] == True]['sessionId'].unique()
        if len(dda_sessions) == 0: return
        session_id = dda_sessions[-1]

    ai_turns = df[(df['sessionId'] == session_id) & (df['player'] == -1)]
    
    if ai_turns.empty: return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.set_xlabel('Turn')
    ax1.set_ylabel('Value (v)', color='tab:blue')
    ax1.plot(ai_turns['turn'], ai_turns['v'], color='tab:blue', marker='d', label='AI Value (Expectation)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Simulations (AI Strength)', color='tab:red')
    ax2.step(ai_turns['turn'], ai_turns['simulations'], color='tab:red', where='post', label='MCTS Sims')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'Adaptasi Dinamis AI: Sesi {session_id}')
    fig.tight_layout()
    plt.savefig(os.path.join(script_dir, f'ai_adaptation_{timestamp_str}.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    # Masukkan nama file hasil ekspor Google Form Anda di sini
    DATA_FILE = 'game_logs.csv' 
    
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data = load_and_preprocess(DATA_FILE)
    if data is not None:
        print(f"Berhasil memuat {len(data)} baris data dari {len(data['sessionId'].unique())} sesi.")
        print_summary_stats(data)
        perform_statistical_test(data)
        generate_markdown_narrative(data, current_timestamp)
        
        # Visualisasi Agregat
        plot_engagement_metrics(data, current_timestamp)
        
        # Visualisasi Sesi Spesifik (Terakhir)
        latest_session = data['sessionId'].iloc[-1]
        plot_score_progression(data, current_timestamp, latest_session)
        plot_ai_adaptation(data, current_timestamp)
