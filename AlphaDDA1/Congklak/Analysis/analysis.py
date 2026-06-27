import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW

# ──────────────────────────────────────────────────────────────
# GLOBAL APA STYLE CONFIGURATION
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

APA_ALPHA = 0.05

# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_and_preprocess(file_name='game_logs.csv'):
    """Memuat data dan melakukan pra-pemrosesan tipe data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    
    if not os.path.exists(file_path):
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

    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed|Column')]
    if 'sessionId' in df.columns:
        df = df.dropna(subset=['sessionId'])
    
    df['isDda'] = df['isDda'].astype(str).str.upper().str.strip() == 'TRUE'
    df['isTerminal'] = df['isTerminal'].astype(str).str.upper().str.strip() == 'TRUE'
    
    numeric_cols = ['thinkTime', 'v', 'simulations', 'scoreP1', 'scoreP2', 'turn']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ──────────────────────────────────────────────────────────────
# HELPER: APA FORMATTING
# ──────────────────────────────────────────────────────────────
def _apa_effect_size_cohens_d(mean_diff, std_diff):
    if std_diff == 0:
        return 0.0
    return mean_diff / std_diff

def _apa_effect_size_r(z_stat, n):
    if n == 0:
        return 0.0
    return abs(z_stat) / np.sqrt(n)

def _apa_interpret_cohens_d(d):
    ad = abs(d)
    if ad < 0.2:
        return "sangat kecil"
    elif ad < 0.5:
        return "kecil"
    elif ad < 0.8:
        return "sedang"
    else:
        return "besar"

def _apa_interpret_p(p):
    if p < 0.001:
        return "p < .001"
    else:
        return f"p = {p:.3f}"

def _apa_latex_math(t_or_z, df_val, stat_val, p_val, effect_symbol=None, effect_value=None):
    base = f"{t_or_z}({df_val}) = {float(stat_val):.3f}, {_apa_interpret_p(p_val)}"
    if effect_symbol and effect_value is not None:
        base += f", {effect_symbol} = {effect_value:.2f}"
    return base

def _apa_decision(p):
    if p < APA_ALPHA:
        return "SIGNIFIKAN"
    else:
        return "TIDAK SIGNIFIKAN"

def _apa_normality_decision(p):
    if p > APA_ALPHA:
        return "normal (p > .05)"
    else:
        return "tidak normal (p <= .05)"


# ──────────────────────────────────────────────────────────────
# 1. DESCRIPTIVE STATISTICS (APA STYLE)
# ──────────────────────────────────────────────────────────────
def print_descriptive_statistics(df):
    terminal_df = df[df['isTerminal'] == True].copy()
    if terminal_df.empty:
        terminal_df = df.sort_values('turn').groupby('sessionId').tail(1).copy()

    terminal_df['AbsMargin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime'])

    print("\n" + "="*70)
    print("STATISTIK DESKRIPTIF (APA STYLE)")
    print("="*70)

    # ── 1a. Demografi Partisipan ──
    n_participants = terminal_df['participant'].nunique()
    n_sessions = len(terminal_df)
    n_dda = terminal_df['isDda'].sum()
    n_nodda = n_sessions - n_dda
    print(f"\n1. DEMOGRAFI PARTISIPAN")
    print(f"   N partisipan = {n_participants}")
    print(f"   Total sesi   = {n_sessions} ({n_dda} DDA, {n_nodda} Non-DDA)")
    sesi_dist = terminal_df.groupby(['participant', 'isDda']).size().unstack(fill_value=0)
    print(f"   Distribusi sesi per partisipan:")
    for part, row in sesi_dist.iterrows():
        print(f"     {part}: {row.get(False,0)} Non-DDA, {row.get(True,0)} DDA")
    print(f"   Catatan: Analisis paired (within-subjects) menggunakan mean per partisipan,")
    print(f"   sehingga partisipan dengan >1 sesi per kondisi tetap berkontribusi 1 data point.")

    # ── 1b. Skor Akhir (participant-level) ──
    print(f"\n2. SKOR AKHIR (P1 = Human, P2 = AI) — Participant-level (N = {n_participants})")
    for dda_label, dda_name in [(True, "DDA"), (False, "Non-DDA")]:
        part_agg = terminal_df[terminal_df['isDda'] == dda_label].groupby('participant').agg({
            'scoreP1': 'mean', 'scoreP2': 'mean', 'AbsMargin': 'mean'
        })
        m_p1 = part_agg['scoreP1'].mean()
        sd_p1 = part_agg['scoreP1'].std()
        med_p1 = part_agg['scoreP1'].median()
        q1_p1 = part_agg['scoreP1'].quantile(0.25)
        q3_p1 = part_agg['scoreP1'].quantile(0.75)
        m_p2 = part_agg['scoreP2'].mean()
        sd_p2 = part_agg['scoreP2'].std()
        med_p2 = part_agg['scoreP2'].median()
        q1_p2 = part_agg['scoreP2'].quantile(0.25)
        q3_p2 = part_agg['scoreP2'].quantile(0.75)
        m_margin = part_agg['AbsMargin'].mean()
        sd_margin = part_agg['AbsMargin'].std()
        med_margin = part_agg['AbsMargin'].median()
        q1_margin = part_agg['AbsMargin'].quantile(0.25)
        q3_margin = part_agg['AbsMargin'].quantile(0.75)
        print(f"   [{dda_name}]")
        print(f"     P1: M = {m_p1:.2f}, SD = {sd_p1:.2f}, Mdn = {med_p1:.2f}, Q1-Q3 = [{q1_p1:.2f}, {q3_p1:.2f}]")
        print(f"     P2: M = {m_p2:.2f}, SD = {sd_p2:.2f}, Mdn = {med_p2:.2f}, Q1-Q3 = [{q1_p2:.2f}, {q3_p2:.2f}]")
        print(f"     Margin: M = {m_margin:.2f}, SD = {sd_margin:.2f}, Mdn = {med_margin:.2f}, Q1-Q3 = [{q1_margin:.2f}, {q3_margin:.2f}]")

    # ── 1c. Waktu Berpikir (Think Time, participant-level) ──
    print(f"\n3. WAKTU BERPIKIR (Think Time, sekon) — Participant-level (N = {n_participants})")
    for dda_label, dda_name in [(True, "DDA"), (False, "Non-DDA")]:
        part_mean = human_moves[human_moves['isDda'] == dda_label].groupby('participant')['thinkTime'].mean()
        m = part_mean.mean()
        sd = part_mean.std()
        med = part_mean.median()
        q1 = part_mean.quantile(0.25)
        q3 = part_mean.quantile(0.75)
        print(f"   [{dda_name}] M = {m:.2f}, SD = {sd:.2f}, Mdn = {med:.2f}, Q1-Q3 = [{q1:.2f}, {q3:.2f}]")

    # ── 1d. Win Rate (participant-level) ──
    print(f"\n4. WIN RATE — Participant-level (N = {n_participants})")
    def get_winner(row):
        if row['scoreP1'] > row['scoreP2']: return 'Human'
        if row['scoreP2'] > row['scoreP1']: return 'AI'
        return 'Draw'
    terminal_df['Winner'] = terminal_df.apply(get_winner, axis=1)
    part_winner = terminal_df.groupby(['participant', 'isDda', 'Winner']).size().unstack(fill_value=0)
    for col in ['Human', 'AI', 'Draw']:
        if col not in part_winner.columns: part_winner[col] = 0
    part_winner_pct = part_winner.div(part_winner.sum(axis=1), axis=0) * 100
    for dda_label, dda_name in [(True, "DDA"), (False, "Non-DDA")]:
        subset = part_winner_pct[part_winner_pct.index.get_level_values('isDda') == dda_label]
        if not subset.empty:
            h = subset['Human'].mean()
            a = subset['AI'].mean()
            d = subset['Draw'].mean()
            print(f"   [{dda_name}] Human = {h:.1f}%, AI = {a:.1f}%, Draw = {d:.1f}%")

    # ── 1e. DDA Metrics ──
    dda_moves = df[(df['isDda'] == True) & (df['player'] == -1) & (df['v'].notnull())]
    if not dda_moves.empty:
        corr, p_corr = stats.pearsonr(dda_moves['v'], dda_moves['simulations'])
        print(f"\n5. MEKANISME DDA (Korelasi Value vs Simulations)")
        print(f"   r({len(dda_moves)-2}) = {corr:.3f}, {_apa_interpret_p(p_corr)}")
        print(f"   Interpretasi: {'Negatif kuat (DDA bekerja)' if corr < -0.5 else 'Lemah/tidak sesuai'}")

    print("\n" + "="*70)


# ──────────────────────────────────────────────────────────────
# 2. INFERENTIAL STATISTICS — HYPOTHESIS TESTING (APA STYLE)
# ──────────────────────────────────────────────────────────────
def print_hypothesis_testing(df):
    terminal_df = df[df['isTerminal'] == True].copy()
    if terminal_df.empty:
        terminal_df = df.sort_values('turn').groupby('sessionId').tail(1).copy()

    terminal_df['AbsMargin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime'])

    print("\n" + "="*70)
    print("PENGUJIAN HIPOTESIS (APA STYLE)")
    print("="*70)

    # ──────────────────────────────────────────────────────────
    # H1: Game Balancing — Margin skor DDA < Non-DDA
    # ──────────────────────────────────────────────────────────
    print("\n" + "-"*70)
    print("HIPOTESIS 1 (H1): GAME BALANCING")
    print("H0: mu_Delta_score_DDA = mu_Delta_score_non-DDA")
    print("    (Tidak terdapat perbedaan selisih skor antara kondisi DDA dan Non-DDA)")
    print("Ha: mu_Delta_score_DDA < mu_Delta_score_non-DDA")
    print("    (Selisih skor pada kondisi DDA lebih kecil dibandingkan Non-DDA)")
    print("-"*70)

    paired_margin = terminal_df.groupby(['participant', 'isDda'])['AbsMargin'].mean().unstack().dropna()
    n_h1 = len(paired_margin)

    if n_h1 >= 2:
        dda_vals = paired_margin[True]
        nodda_vals = paired_margin[False]
        diff_h1 = dda_vals - nodda_vals
        mean_diff = diff_h1.mean()
        std_diff = diff_h1.std()

        if n_h1 >= 3:
            _, p_shapiro = stats.shapiro(diff_h1)
        else:
            p_shapiro = 0.0

        print(f"\n   A. UJI ASUMSI (Normalitas Selisih)")
        print(f"      Shapiro-Wilk: p = {p_shapiro:.4f}")
        print(f"      Keputusan: Data selisih {_apa_normality_decision(p_shapiro)}")

        if p_shapiro > APA_ALPHA or n_h1 < 5:
            paired_data = pd.DataFrame({'DDA': dda_vals, 'Non-DDA': nodda_vals})
            paired_test = DescrStatsW(paired_data['DDA'] - paired_data['Non-DDA'])
            t_stat = paired_test.ttest_mean(0)[0]
            p_val_two = paired_test.ttest_mean(0)[1]
            d_cohen = _apa_effect_size_cohens_d(mean_diff, std_diff)
            effect_desc = _apa_interpret_cohens_d(d_cohen)
            p_val = p_val_two / 2 if mean_diff < 0 else 1 - p_val_two / 2
            method = "Paired Samples t-Test (statsmodels, one-tailed)"
            df_method = n_h1 - 1

            print(f"\n   B. UJI HIPOTESIS (Parametrik, One-Tailed)")
            print(f"      Metode: {method}")
            print(f"      t({df_method}) = {t_stat:.3f}, {_apa_interpret_p(p_val)}, d = {d_cohen:.2f} ({effect_desc})")
        else:
            t_stat, p_val_two = stats.wilcoxon(dda_vals, nodda_vals)
            p_val = p_val_two / 2 if mean_diff < 0 else 1 - p_val_two / 2
            method = "Wilcoxon Signed-Rank Test (scipy, one-tailed)"
            n_pairs = len(diff_h1)
            z_stat = (t_stat - n_pairs * (n_pairs + 1) / 4) / np.sqrt(n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24)
            r_effect = _apa_effect_size_r(z_stat, n_pairs)

            print(f"\n   B. UJI HIPOTESIS (Non-Parametrik, One-Tailed)")
            print(f"      Metode: {method}")
            print(f"      W = {t_stat:.1f}, Z = {z_stat:.2f}, {_apa_interpret_p(p_val)}")
            print(f"      Effect size r = {r_effect:.2f}")

        mean_dda = dda_vals.mean()
        mean_nodda = nodda_vals.mean()
        print(f"\n   C. DESKRIPTIF")
        print(f"      DDA:      M = {mean_dda:.2f}, SD = {dda_vals.std():.2f}")
        print(f"      Non-DDA:  M = {mean_nodda:.2f}, SD = {nodda_vals.std():.2f}")
        print(f"      Selisih:  Delta M = {mean_diff:.2f}")

        print(f"\n   D. KEPUTUSAN")
        if p_val < APA_ALPHA:
            print(f"      H0 DITOLAK. Terdapat perbedaan signifikan ({_apa_interpret_p(p_val)}).")
            if mean_dda < mean_nodda:
                print(f"      -> DDA menghasilkan selisih skor lebih kecil, mendukung H1.")
            else:
                print(f"      -> Arah berlawanan: DDA justru meningkatkan selisih.")
        else:
            print(f"      H0 GAGAL DITOLAK. Tidak ada perbedaan signifikan ({_apa_interpret_p(p_val)}).")
    else:
        print("   Data tidak mencukupi untuk uji H1.")

    # ──────────────────────────────────────────────────────────
    # H2: Engagement — Waktu Berpikir DDA vs Non-DDA (Two-tailed)
    # ──────────────────────────────────────────────────────────
    print("\n" + "-"*70)
    print("HIPOTESIS 2 (H2): ENGAGEMENT (Waktu Berpikir)")
    print("H0: mu_thinkTime_DDA = mu_thinkTime_non-DDA")
    print("    (Tidak terdapat perbedaan rata-rata waktu berpikir antara DDA dan Non-DDA)")
    print("Ha: mu_thinkTime_DDA != mu_thinkTime_non-DDA")
    print("    (Terdapat perbedaan rata-rata waktu berpikir antara DDA dan Non-DDA)")
    print("-"*70)

    paired_think = human_moves.groupby(['participant', 'isDda'])['thinkTime'].mean().unstack().dropna()
    n_h2 = len(paired_think)

    if n_h2 >= 2:
        dda_t = paired_think[True]
        nodda_t = paired_think[False]
        diff_t = dda_t - nodda_t
        mean_diff_t = diff_t.mean()
        std_diff_t = diff_t.std()

        if n_h2 >= 3:
            _, p_shapiro_t = stats.shapiro(diff_t)
        else:
            p_shapiro_t = 0.0

        print(f"\n   A. UJI ASUMSI (Normalitas Selisih)")
        print(f"      Shapiro-Wilk: p = {p_shapiro_t:.4f}")
        print(f"      Keputusan: Data selisih {_apa_normality_decision(p_shapiro_t)}")

        if p_shapiro_t > APA_ALPHA or n_h2 < 5:
            paired_data_t = pd.DataFrame({'DDA': dda_t, 'Non-DDA': nodda_t})
            paired_test_t = DescrStatsW(paired_data_t['DDA'] - paired_data_t['Non-DDA'])
            t_stat_t = paired_test_t.ttest_mean(0)[0]
            p_val_t = paired_test_t.ttest_mean(0)[1]
            method_t = "Paired Samples t-Test (statsmodels, two-tailed)"
            df_t = n_h2 - 1
            d_cohen_t = _apa_effect_size_cohens_d(mean_diff_t, std_diff_t)
            effect_desc_t = _apa_interpret_cohens_d(d_cohen_t)

            print(f"\n   B. UJI HIPOTESIS (Parametrik, Two-Tailed)")
            print(f"      Metode: {method_t}")
            print(f"      t({df_t}) = {t_stat_t:.3f}, {_apa_interpret_p(p_val_t)}, d = {d_cohen_t:.2f} ({effect_desc_t})")
        else:
            t_stat_t, p_val_t = stats.wilcoxon(dda_t, nodda_t)
            method_t = "Wilcoxon Signed-Rank Test (scipy, two-tailed)"
            n_pairs_t = len(diff_t)
            z_stat_t = (t_stat_t - n_pairs_t * (n_pairs_t + 1) / 4) / np.sqrt(n_pairs_t * (n_pairs_t + 1) * (2 * n_pairs_t + 1) / 24)
            r_effect_t = _apa_effect_size_r(z_stat_t, n_pairs_t)

            print(f"\n   B. UJI HIPOTESIS (Non-Parametrik, Two-Tailed)")
            print(f"      Metode: {method_t}")
            print(f"      W = {t_stat_t:.1f}, Z = {z_stat_t:.2f}, {_apa_interpret_p(p_val_t)}")
            print(f"      Effect size r = {r_effect_t:.2f}")

        mean_dda_t = dda_t.mean()
        mean_nodda_t = nodda_t.mean()
        print(f"\n   C. DESKRIPTIF")
        print(f"      DDA:      M = {mean_dda_t:.2f}, SD = {dda_t.std():.2f}")
        print(f"      Non-DDA:  M = {mean_nodda_t:.2f}, SD = {nodda_t.std():.2f}")
        print(f"      Selisih:  Delta M = {mean_diff_t:.2f}")

        print(f"\n   D. KEPUTUSAN")
        if p_val_t < APA_ALPHA:
            print(f"      H0 DITOLAK. Terdapat perbedaan signifikan ({_apa_interpret_p(p_val_t)}).")
            if mean_dda_t > mean_nodda_t:
                print(f"      -> Waktu berpikir DDA lebih lama.")
            else:
                print(f"      -> Waktu berpikir Non-DDA lebih lama.")
        else:
            print(f"      H0 GAGAL DITOLAK. Tidak ada perbedaan signifikan ({_apa_interpret_p(p_val_t)}).")
    else:
        print("   Data tidak mencukupi untuk uji H2.")

    print("\n" + "="*70)


# ──────────────────────────────────────────────────────────────
# 3. APA-STYLE SUMMARY TABLE (Plain Text)
# ──────────────────────────────────────────────────────────────
def print_apa_summary_table(df):
    terminal_df = df[df['isTerminal'] == True].copy()
    if terminal_df.empty:
        terminal_df = df.sort_values('turn').groupby('sessionId').tail(1).copy()
    terminal_df['AbsMargin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime'])

    # H1
    paired_margin = terminal_df.groupby(['participant', 'isDda'])['AbsMargin'].mean().unstack().dropna()
    if len(paired_margin) >= 2:
        dda_m = paired_margin[True]
        nodda_m = paired_margin[False]
        diff_m = dda_m - nodda_m
        mean_diff_m = diff_m.mean()
        if len(diff_m) >= 3:
            _, p_norm_m = stats.shapiro(diff_m)
        else:
            p_norm_m = 1.0
        if p_norm_m > APA_ALPHA or len(paired_margin) < 5:
            t_h1, p_h1_two = stats.ttest_rel(dda_m, nodda_m)
            p_h1 = p_h1_two / 2 if mean_diff_m < 0 else 1 - p_h1_two / 2
            stat_h1 = f"t({len(paired_margin)-1}) = {t_h1:.3f}"
            d_h1 = _apa_effect_size_cohens_d(mean_diff_m, diff_m.std())
            es_h1 = f"d = {d_h1:.2f}"
        else:
            w_h1, p_h1_two = stats.wilcoxon(dda_m, nodda_m)
            p_h1 = p_h1_two / 2 if mean_diff_m < 0 else 1 - p_h1_two / 2
            n_p = len(diff_m)
            z_h1 = (w_h1 - n_p * (n_p + 1) / 4) / np.sqrt(n_p * (n_p + 1) * (2 * n_p + 1) / 24)
            stat_h1 = f"Z = {z_h1:.2f}"
            r_h1 = _apa_effect_size_r(z_h1, n_p)
            es_h1 = f"r = {r_h1:.2f}"
    else:
        p_h1 = 1.0
        stat_h1 = "N/A"
        es_h1 = "N/A"

    # H2
    paired_think = human_moves.groupby(['participant', 'isDda'])['thinkTime'].mean().unstack().dropna()
    if len(paired_think) >= 2:
        dda_t = paired_think[True]
        nodda_t = paired_think[False]
        diff_t = dda_t - nodda_t
        mean_diff_t = diff_t.mean()
        if len(diff_t) >= 3:
            _, p_norm_t = stats.shapiro(diff_t)
        else:
            p_norm_t = 1.0
        if p_norm_t > APA_ALPHA or len(paired_think) < 5:
            t_h2, p_h2_two = stats.ttest_rel(dda_t, nodda_t)
            p_h2 = p_h2_two / 2 if mean_diff_t < 0 else 1 - p_h2_two / 2
            stat_h2 = f"t({len(paired_think)-1}) = {t_h2:.3f}"
            d_h2 = _apa_effect_size_cohens_d(mean_diff_t, diff_t.std())
            es_h2 = f"d = {d_h2:.2f}"
        else:
            w_h2, p_h2_two = stats.wilcoxon(dda_t, nodda_t)
            p_h2 = p_h2_two / 2 if mean_diff_t < 0 else 1 - p_h2_two / 2
            n_p2 = len(diff_t)
            z_h2 = (w_h2 - n_p2 * (n_p2 + 1) / 4) / np.sqrt(n_p2 * (n_p2 + 1) * (2 * n_p2 + 1) / 24)
            stat_h2 = f"Z = {z_h2:.2f}"
            r_h2 = _apa_effect_size_r(z_h2, n_p2)
            es_h2 = f"r = {r_h2:.2f}"
    else:
        p_h2 = 1.0
        stat_h2 = "N/A"
        es_h2 = "N/A"

    print("\n" + "="*70)
    print("TABEL RINGKASAN HASIL UJI HIPOTESIS (APA STYLE)")
    print("="*70)
    print(f"{'Hipotesis':<40} {'Uji Statistik':<25} {'p':<12} {'Effect Size':<15} {'Keputusan':<15}")
    print("-"*107)
    print(f"{'H1: Margin DDA < Non-DDA':<40} {stat_h1:<25} {_apa_interpret_p(p_h1):<12} {es_h1:<15} {_apa_decision(p_h1):<15}")
    print(f"{'H2: ThinkTime DDA vs Non-DDA':<40} {stat_h2:<25} {_apa_interpret_p(p_h2):<12} {es_h2:<15} {_apa_decision(p_h2):<15}")
    print("="*107)
    print("Catatan: alpha = .05. Effect size: d = Cohen's d, r = rank-biserial correlation.")
    print()


# ──────────────────────────────────────────────────────────────
# VISUALIZATION — MAINSTREAM GRAPHS ONLY
# ──────────────────────────────────────────────────────────────
def plot_thinktime_boxplot(df, timestamp_str):
    """Boxplot distribusi think time per kondisi dengan stripplot."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime']).copy()
    if human_moves.empty:
        print("[WARN] Tidak ada data think time.")
        return
    
    plt.figure(figsize=(8, 6))
    # Boxplot dengan stripplot (data point individual dengan warna yang sama)
    sns.boxplot(x='isDda', y='thinkTime', data=human_moves, palette=['#e74c3c', '#2ecc71'])
    # Stripplot dengan warna sesuai kelompok
    for dda_label, color in [(False, '#e74c3c'), (True, '#2ecc71')]:
        subset = human_moves[human_moves['isDda'] == dda_label]
        sns.stripplot(x='isDda', y='thinkTime', data=subset, color=color, 
                      alpha=0.6, size=4, jitter=0.2)
    plt.title('Boxplot Waktu Berpikir\nDDA vs Non-DDA', fontsize=13, fontweight='bold')
    plt.xlabel('Kondisi', fontsize=11)
    plt.ylabel('thinkTime (s)', fontsize=11)
    plt.xticks([0, 1], ['Non-DDA\n(AI Statis)', 'DDA\n(AI Adaptif)'])
    plt.grid(True, linestyle='--', alpha=0.5)
    # Tambahkan legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Non-DDA (AI Statis)'),
                       Patch(facecolor='#2ecc71', label='DDA (AI Adaptif)')]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'thinktime_boxplot_{timestamp_str}.png'), dpi=300)
    plt.close()
    print(f"[INFO] Grafik think time (boxplot + stripplot) disimpan ke: thinktime_boxplot_{timestamp_str}.png")

def plot_thinktime_bar(df, timestamp_str):
    """Bar chart rerata think time per kondisi dengan error bar (SD) - participant-level."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime']).copy()
    if human_moves.empty:
        print("[WARN] Tidak ada data think time.")
        return
    
    # Agregasi per participant (sesuai dengan uji hipotesis)
    part_think = human_moves.groupby(['participant', 'isDda'])['thinkTime'].mean().unstack().dropna()
    n_participants = len(part_think)
    
    # Hitung statistik per kondisi (mean dan SD across participants)
    stats_data = []
    for dda_label, dda_name in [(False, 'Non-DDA'), (True, 'DDA')]:
        if dda_label in part_think.columns:
            values = part_think[dda_label]
            mean_val = values.mean()
            sd_val = values.std()
            stats_data.append({'isDda': dda_label, 'mean': mean_val, 'sd': sd_val})
            print(f"   [{dda_name}] N = {len(values)}, Mean = {mean_val:.2f}, SD = {sd_val:.2f}")
    
    # Buat DataFrame untuk barplot
    bar_data = pd.DataFrame({
        'isDda': [False, True],
        'thinkTime': [stats_data[0]['mean'], stats_data[1]['mean']],
        'sd': [stats_data[0]['sd'], stats_data[1]['sd']]
    })
    
    plt.figure(figsize=(8, 6))
    # Bar chart dengan SD error bar (konsisten dengan uji-t N=7)
    ax = sns.barplot(x='isDda', y='thinkTime', data=bar_data, palette=['#e74c3c', '#2ecc71'], 
                     errorbar=('sd', 0), capsize=0.1, errwidth=1.5)
    
    # Tambahkan error bar manual dengan SD yang benar
    for i, row in bar_data.iterrows():
        ax.errorbar(i, row['thinkTime'], yerr=row['sd'], 
                    fmt='none', color='black', capsize=5, capthick=1.5, elinewidth=1.5)
    
    plt.title(f'Rata-Rata Waktu Berpikir (N = {n_participants} partisipan)\nDDA vs Non-DDA', fontsize=13, fontweight='bold')
    plt.xlabel('Kondisi', fontsize=11)
    plt.ylabel('thinkTime (s)', fontsize=11)
    plt.xticks([0, 1], ['Non-DDA\n(AI Statis)', 'DDA\n(AI Adaptif)'])
    plt.grid(True, linestyle='--', alpha=0.5)
    # Tambahkan legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Non-DDA (AI Statis)'),
                       Patch(facecolor='#2ecc71', label='DDA (AI Adaptif)')]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'thinktime_bar_{timestamp_str}.png'), dpi=300)
    plt.close()
    print(f"[INFO] Grafik think time (bar, participant-level) disimpan ke: thinktime_bar_{timestamp_str}.png")

def plot_thinktime_distribution(df, timestamp_str):
    """Distribusi waktu berpikir per kondisi (histogram + KDE)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime']).copy()
    if human_moves.empty:
        print("[WARN] Tidak ada data think time untuk distribusi.")
        return
    
    plt.figure(figsize=(10, 6))
    for dda_label, dda_name, color in [(True, 'DDA', '#2ecc71'), (False, 'Non-DDA', '#e74c3c')]:
        data = human_moves[human_moves['isDda'] == dda_label]['thinkTime']
        if len(data) > 1:
            sns.kdeplot(data, label=dda_name, color=color, fill=True, alpha=0.3, linewidth=2)
            plt.hist(data, bins=30, alpha=0.4, color=color, density=True)
    
    plt.title('Distribusi Waktu Berpikir\nDDA vs Non-DDA', fontsize=13, fontweight='bold')
    plt.xlabel('thinkTime (s)', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.legend(title='Kondisi')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'thinktime_dist_{timestamp_str}.png'), dpi=300)
    plt.close()
    print(f"[INFO] Grafik distribusi think time disimpan ke: thinktime_dist_{timestamp_str}.png")

def plot_thinktime_by_turn(df, timestamp_str):
    """Waktu berpikir per giliran (turn) untuk kedua kondisi."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime']).copy()
    if human_moves.empty:
        print("[WARN] Tidak ada data think time per turn.")
        return
    
    turn_agg = human_moves.groupby(['turn', 'isDda'])['thinkTime'].mean().unstack()
    
    plt.figure(figsize=(12, 6))
    for dda_label, dda_name, color in [(True, 'DDA', '#2ecc71'), (False, 'Non-DDA', '#e74c3c')]:
        if dda_label in turn_agg.columns:
            plt.plot(turn_agg.index, turn_agg[dda_label], marker='o', label=dda_name, color=color, alpha=0.8)
    
    plt.title('Rata-Rata Waktu Berpikir per Giliran (Turn)', fontsize=13, fontweight='bold')
    plt.xlabel('Giliran (Turn)', fontsize=11)
    plt.ylabel('thinkTime (s)', fontsize=11)
    plt.legend(title='Kondisi')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'thinktime_by_turn_{timestamp_str}.png'), dpi=300)
    plt.close()
    print(f"[INFO] Grafik think time per turn disimpan ke: thinktime_by_turn_{timestamp_str}.png")

def plot_scoremargin_boxplot(df, timestamp_str):
    """Boxplot distribusi selisih skor per kondisi dengan stripplot."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    terminal_df = df[df['isTerminal'] == True].copy()
    if terminal_df.empty:
        terminal_df = df.sort_values('turn').groupby('sessionId').tail(1).copy()
    terminal_df['AbsMargin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    
    plt.figure(figsize=(8, 6))
    # Boxplot dengan stripplot (data point individual dengan warna yang sama)
    sns.boxplot(x='isDda', y='AbsMargin', data=terminal_df, palette=['#e74c3c', '#2ecc71'])
    # Stripplot dengan warna sesuai kelompok
    for dda_label, color in [(False, '#e74c3c'), (True, '#2ecc71')]:
        subset = terminal_df[terminal_df['isDda'] == dda_label]
        sns.stripplot(x='isDda', y='AbsMargin', data=subset, color=color, 
                      alpha=0.6, size=4, jitter=0.2)
    plt.title('Boxplot Selisih Skor\nDDA vs Non-DDA', fontsize=13, fontweight='bold')
    plt.xlabel('Kondisi', fontsize=11)
    plt.ylabel('Δscore', fontsize=11)
    plt.xticks([0, 1], ['Non-DDA\n(AI Statis)', 'DDA\n(AI Adaptif)'])
    plt.grid(True, linestyle='--', alpha=0.5)
    # Tambahkan legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Non-DDA (AI Statis)'),
                       Patch(facecolor='#2ecc71', label='DDA (AI Adaptif)')]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'scoremargin_boxplot_{timestamp_str}.png'), dpi=300)
    plt.close()
    print(f"[INFO] Grafik score margin (boxplot + stripplot) disimpan ke: scoremargin_boxplot_{timestamp_str}.png")

def plot_scoremargin_bar(df, timestamp_str):
    """Bar chart rerata selisih skor per kondisi dengan error bar (SD) - participant-level."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    terminal_df = df[df['isTerminal'] == True].copy()
    if terminal_df.empty:
        terminal_df = df.sort_values('turn').groupby('sessionId').tail(1).copy()
    terminal_df['AbsMargin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    
    # Agregasi per participant (sesuai dengan uji hipotesis)
    part_margin = terminal_df.groupby(['participant', 'isDda'])['AbsMargin'].mean().unstack().dropna()
    n_participants = len(part_margin)
    
    # Hitung statistik per kondisi (mean dan SD across participants)
    stats_data = []
    for dda_label, dda_name in [(False, 'Non-DDA'), (True, 'DDA')]:
        if dda_label in part_margin.columns:
            values = part_margin[dda_label]
            mean_val = values.mean()
            sd_val = values.std()
            stats_data.append({'isDda': dda_label, 'mean': mean_val, 'sd': sd_val})
            print(f"   [{dda_name}] N = {len(values)}, Mean = {mean_val:.2f}, SD = {sd_val:.2f}")
    
    # Buat DataFrame untuk barplot
    bar_data = pd.DataFrame({
        'isDda': [False, True],
        'AbsMargin': [stats_data[0]['mean'], stats_data[1]['mean']],
        'sd': [stats_data[0]['sd'], stats_data[1]['sd']]
    })
    
    plt.figure(figsize=(8, 6))
    # Bar chart dengan SD error bar (konsisten dengan uji-t N=7)
    ax = sns.barplot(x='isDda', y='AbsMargin', data=bar_data, palette=['#e74c3c', '#2ecc71'], 
                     errorbar=('sd', 0), capsize=0.1, errwidth=1.5)
    
    # Tambahkan error bar manual dengan SD yang benar
    for i, row in bar_data.iterrows():
        ax.errorbar(i, row['AbsMargin'], yerr=row['sd'], 
                    fmt='none', color='black', capsize=5, capthick=1.5, elinewidth=1.5)
    
    plt.title(f'Rata-Rata Selisih Skor (N = {n_participants} partisipan)\nDDA vs Non-DDA', fontsize=13, fontweight='bold')
    plt.xlabel('Kondisi', fontsize=11)
    plt.ylabel('Δscore', fontsize=11)
    plt.xticks([0, 1], ['Non-DDA\n(AI Statis)', 'DDA\n(AI Adaptif)'])
    plt.grid(True, linestyle='--', alpha=0.5)
    # Tambahkan legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Non-DDA (AI Statis)'),
                       Patch(facecolor='#2ecc71', label='DDA (AI Adaptif)')]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'scoremargin_bar_{timestamp_str}.png'), dpi=300)
    plt.close()
    print(f"[INFO] Grafik score margin (bar, participant-level) disimpan ke: scoremargin_bar_{timestamp_str}.png")

def plot_score_progression(df, timestamp_str):
    """
    Menghasilkan dua plot progresi skor: satu untuk sesi DDA, satu untuk sesi Non-DDA.
    Tanpa menampilkan nama participant (anonim).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Pilih sesi terakhir untuk masing-masing kondisi jika tidak disediakan
    dda_sessions = df[df['isDda'] == True]['sessionId'].unique()
    session_id_dda = dda_sessions[1] if len(dda_sessions) > 0 else None
    nodda_sessions = df[df['isDda'] == False]['sessionId'].unique()
    session_id_nodda = nodda_sessions[1] if len(nodda_sessions) > 0 else None
    
    for session_id, label, is_dda in [
        (session_id_dda, 'DDA', True),
        (session_id_nodda, 'Non-DDA', False)
    ]:
        if session_id is None:
            continue
        session_data = df[df['sessionId'] == session_id].copy()
        if session_data.empty:
            continue
        plt.figure(figsize=(12, 6))
        plt.plot(session_data['turn'], session_data['scoreP1'], label='P1 (Human)', marker='o', color='#ff0000')
        plt.plot(session_data['turn'], session_data['scoreP2'], label='P2 (AI)', marker='s', color="#0000ff")
        plt.title(f'Progresi Skor — {label}', fontsize=12)
        plt.xlabel('Giliran (Turn)')
        plt.ylabel('Skor')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        fname = f'score_progression_{label.lower()}_{timestamp_str}.png'
        plt.savefig(os.path.join(script_dir, fname), dpi=300)
        plt.close()
        print(f"[INFO] Grafik progresi skor ({label}) disimpan ke: {fname}")

def plot_score_diff_by_turn(df, timestamp_str):
    """Score difference per turn untuk kedua kondisi."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    terminal_df = df[df['isTerminal'] == True].copy()
    if terminal_df.empty:
        terminal_df = df.sort_values('turn').groupby('sessionId').tail(1).copy()
    
    # Hitung score diff per turn
    df_with_margin = df.copy()
    df_with_margin['scoreDiff'] = df_with_margin['scoreP1'] - df_with_margin['scoreP2']
    
    # Agregasi per turn dan kondisi
    turn_diff = df_with_margin.groupby(['turn', 'isDda'])['scoreDiff'].mean().unstack()
    
    plt.figure(figsize=(12, 6))
    for dda_label, dda_name, color in [(True, 'DDA', '#2ecc71'), (False, 'Non-DDA', '#e74c3c')]:
        if dda_label in turn_diff.columns:
            plt.plot(turn_diff.index, turn_diff[dda_label], marker='o', label=dda_name, color=color, alpha=0.8, linewidth=2)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.title('Rata-Rata Selisih Skor per Giliran (Turn)\n(Positif = Human unggul, Negatif = AI unggul)', fontsize=13, fontweight='bold')
    plt.xlabel('Giliran (Turn)', fontsize=11)
    plt.ylabel('Selisih Skor (P1 - P2)', fontsize=11)
    plt.legend(title='Kondisi')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'scorediff_by_turn_{timestamp_str}.png'), dpi=300)
    plt.close()
    print(f"[INFO] Grafik score diff per turn disimpan ke: scorediff_by_turn_{timestamp_str}.png")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_FILE = 'game_logs.csv'
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data = load_and_preprocess(DATA_FILE)
    if data is not None:
        print(f"Loaded {len(data)} rows from {len(data['sessionId'].unique())} sessions.\n")

        # 1. Descriptive Statistics (APA)
        print_descriptive_statistics(data)

        # 2. Hypothesis Testing (APA)
        print_hypothesis_testing(data)

        # 3. Summary Table (APA)
        print_apa_summary_table(data)

        # 4. Mainstream Visualizations only
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS...")
        print("="*70 + "\n")
        
        plot_thinktime_boxplot(data, current_timestamp)
        plot_thinktime_bar(data, current_timestamp)
        plot_thinktime_distribution(data, current_timestamp)
        plot_thinktime_by_turn(data, current_timestamp)
        plot_scoremargin_boxplot(data, current_timestamp)
        plot_scoremargin_bar(data, current_timestamp)
        plot_score_progression(data, current_timestamp)
        plot_score_diff_by_turn(data, current_timestamp)
        
        print("\n[INFO] Semua grafik berhasil di-generate!")
        print(f"[INFO] Total 9 grafik PNG (300 dpi) telah disimpan di direktori yang sama.")
