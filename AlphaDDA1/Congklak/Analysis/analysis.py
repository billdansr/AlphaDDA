import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import numpy as np
import base64
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
HTML_CONTENT = ""

# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
def add_to_html(title, content):
    global HTML_CONTENT
    HTML_CONTENT += f"<h3>{title}</h3>\n{content}<br><br>\n"

def _apa_interpret_p(p):
    if p < 0.001:
        return "p < .001"
    else:
        p_str = f"{p:.3f}"
        if p_str.startswith("0."):
            p_str = p_str[1:]
        return f"p = {p_str}"

def _apa_effect_size_cohens_d(mean_diff, std_diff):
    return mean_diff / std_diff if std_diff != 0 else 0.0

def _apa_effect_size_r(z_stat, n):
    return abs(z_stat) / np.sqrt(n) if n != 0 else 0.0

def load_and_preprocess(file_name='game_logs.csv'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    if not os.path.exists(file_path):
        excel_path = file_path.replace('.csv', '.xlsx')
        if os.path.exists(excel_path):
            file_path = excel_path
        else:
            print(f"Error: File {file_name} tidak ditemukan.")
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
    
    # Filter out local multiplayer sessions (Human vs Human)
    if 'isP2Human' in df.columns:
        df['isP2Human'] = df['isP2Human'].astype(str).str.upper().str.strip() == 'TRUE'
        df = df[df['isP2Human'] == False]
    
    numeric_cols = ['thinkTime', 'v', 'simulations', 'scoreP1', 'scoreP2', 'turn']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# ──────────────────────────────────────────────────────────────
# 1. ANALISIS DESKRIPTIF & WIN RATE AGREGAT
# ──────────────────────────────────────────────────────────────
def analyze_descriptive(df):
    terminal_df = df.sort_values('turn').groupby('sessionId').last().reset_index()
    
    # 1. Demografi Partisipan (APA Table 1 format)
    player_metadata = {
        'Rizka': {'age': 22, 'gender': 'Male', 'exp': 'Inexperienced'},
        'Dies Natalis': {'age': 22, 'gender': 'Male', 'exp': 'Experienced'},
        'Budi A': {'age': 22, 'gender': 'Male', 'exp': 'Experienced'},
        'Vanka': {'age': 22, 'gender': 'Male', 'exp': 'Inexperienced'},
        'Budi2 A': {'age': 22, 'gender': 'Male', 'exp': 'Inexperienced'},
        'Budiono Siregar': {'age': 21, 'gender': 'Male', 'exp': 'Experienced'},
        'SAUQI A': {'age': 22, 'gender': 'Male', 'exp': 'Inexperienced'},
        'rifal casmana': {'age': 21, 'gender': 'Male', 'exp': 'Inexperienced'},
        'NHD B': {'age': 21, 'gender': 'Male', 'exp': 'Experienced'},
        'mumtaz': {'age': 22, 'gender': 'Male', 'exp': 'Experienced'}
    }
    
    males = sum(1 for p, m in player_metadata.items() if m['gender'] == 'Male')
    females = sum(1 for p, m in player_metadata.items() if m['gender'] == 'Female')
    age_21 = sum(1 for p, m in player_metadata.items() if m['age'] == 21)
    age_22 = sum(1 for p, m in player_metadata.items() if m['age'] == 22)
    exp_y = sum(1 for p, m in player_metadata.items() if m['exp'] == 'Experienced')
    exp_n = sum(1 for p, m in player_metadata.items() if m['exp'] == 'Inexperienced')
    
    demo_html = f"""
    <table class="table-apa" style="width: 60%; margin: 15px auto 35px auto;">
      <thead>
        <tr>
          <th style="text-align: left;">Baseline Characteristic</th>
          <th>n</th>
          <th>%</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="text-align: left; font-weight: bold;" colspan="3">Gender</td>
        </tr>
        <tr>
          <td style="text-align: left; padding-left: 20px;">Female</td>
          <td>{females}</td>
          <td>{females*10:.0f}%</td>
        </tr>
        <tr>
          <td style="text-align: left; padding-left: 20px;">Male</td>
          <td>{males}</td>
          <td>{males*10:.0f}%</td>
        </tr>
        <tr>
          <td style="text-align: left; font-weight: bold;" colspan="3">Age (Years)</td>
        </tr>
        <tr>
          <td style="text-align: left; padding-left: 20px;">21</td>
          <td>{age_21}</td>
          <td>{age_21*10:.0f}%</td>
        </tr>
        <tr>
          <td style="text-align: left; padding-left: 20px;">22</td>
          <td>{age_22}</td>
          <td>{age_22*10:.0f}%</td>
        </tr>
        <tr>
          <td style="text-align: left; font-weight: bold;" colspan="3">Previous Congklak Experience</td>
        </tr>
        <tr>
          <td style="text-align: left; padding-left: 20px;">Experienced (Yes)</td>
          <td>{exp_y}</td>
          <td>{exp_y*10:.0f}%</td>
        </tr>
        <tr>
          <td style="text-align: left; padding-left: 20px;">Inexperienced (No)</td>
          <td>{exp_n}</td>
          <td>{exp_n*10:.0f}%</td>
        </tr>
      </tbody>
    </table>
    <div style="font-size: 11px; text-align: center; font-family: 'Times New Roman'; margin-top: -25px; margin-bottom: 30px;">
      <i>Note.</i> N = 10. Participants were on average 21.70 years old (SD = 0.48).
    </div>
    """
    add_to_html("1. Sociodemographic Characteristics of Participants at Baseline (Table 1)", demo_html)

    # ── HITUNG WINNER UNTUK ANALISIS LANJUTAN ──
    def get_winner(row):
        if row['scoreP1'] > row['scoreP2']: return 'Human'
        if row['scoreP2'] > row['scoreP1']: return 'AI'
        return 'Draw'
    terminal_df['Winner'] = terminal_df.apply(get_winner, axis=1)
    
    # ── ANALISIS STRATEGI LANGKAH PEMBUKA (TURN 1 & TURN 2) ──
    turn1_human = df[(df['turn'] == 1) & (df['player'] == 1) & (df['move'].isin(range(7)))]
    t1_counts = turn1_human['move'].value_counts().reindex(range(7), fill_value=0)
    t1_pct = (t1_counts / t1_counts.sum() * 100) if t1_counts.sum() > 0 else t1_counts
    
    t1_df = pd.DataFrame({
        'Lubang Pilihan': [f"Lubang {i}" for i in range(7)],
        'Frekuensi': t1_counts.values,
        'Persentase (%)': [f"{v:.1f}%" for v in t1_pct.values]
    })
    
    turn2_ai = df[(df['turn'] == 2) & (df['player'] == -1) & (df['move'].isin(range(7)))]
    if not turn2_ai.empty:
        t2_counts = turn2_ai.groupby(['isDda', 'move']).size().unstack(fill_value=0).reindex(columns=range(7), fill_value=0)
        t2_counts.index = t2_counts.index.map({True: 'DDA (Adaptif)', False: 'Non-DDA (Statis)'})
        t2_counts.columns = [f"Lubang {c}" for c in t2_counts.columns]
        t2_html = t2_counts.to_html(classes='table table-apa', border=0)
    else:
        t2_html = "<p>Data AI Turn 2 tidak ditemukan.</p>"
        
    add_to_html("3a. Preferensi Langkah Pembuka Manusia (Turn 1)", t1_df.to_html(index=False, classes='table table-apa', border=0))
    add_to_html("3b. Preferensi Langkah Balasan AI (Turn 2) per Kondisi", t2_html)
    
    return terminal_df

# ──────────────────────────────────────────────────────────────
# 2. PENGUJIAN HIPOTESIS (H1 & H2)
# ──────────────────────────────────────────────────────────────
def test_hypotheses(df, terminal_df):
    terminal_df['AbsMargin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime'])
    playtime_df = human_moves.groupby(['sessionId', 'participant', 'isDda'])['thinkTime'].sum().reset_index()
    playtime_df.rename(columns={'thinkTime': 'TotalPlaytime'}, inplace=True)
    
    paired_margin = terminal_df.groupby(['participant', 'isDda'])['AbsMargin'].mean().unstack().dropna()
    paired_playtime = playtime_df.groupby(['participant', 'isDda'])['TotalPlaytime'].mean().unstack().dropna()
    
    # Uji Normalitas (Shapiro-Wilk) pada selisih data
    dda_margin = paired_margin[True]
    nodda_margin = paired_margin[False]
    diff_margin = dda_margin - nodda_margin
    shapiro_w_margin, p_norm_margin = stats.shapiro(diff_margin)
    
    if p_norm_margin > 0.05:
        t_margin, p_margin = stats.ttest_rel(dda_margin, nodda_margin)
        test_type_margin = "Paired t-Test"
    else:
        # Wilcoxon returns statistic W
        t_margin, p_margin = stats.wilcoxon(dda_margin, nodda_margin)
        test_type_margin = "Wilcoxon W"
        
    cohen_margin = _apa_effect_size_cohens_d(diff_margin.mean(), diff_margin.std())
    
    dda_playtime = paired_playtime[True]
    nodda_playtime = paired_playtime[False]
    diff_playtime = dda_playtime - nodda_playtime
    shapiro_w_playtime, p_norm_playtime = stats.shapiro(diff_playtime)
    
    if p_norm_playtime > 0.05:
        t_playtime, p_playtime = stats.ttest_rel(dda_playtime, nodda_playtime)
        test_type_playtime = "Paired t-Test"
    else:
        t_playtime, p_playtime = stats.wilcoxon(dda_playtime, nodda_playtime)
        test_type_playtime = "Wilcoxon W"
        
    cohen_playtime = _apa_effect_size_cohens_d(diff_playtime.mean(), diff_playtime.std())
    
    # Generate Shapiro-Wilk Table
    dec_margin = "Normal (p > .05)" if p_norm_margin > 0.05 else "Tidak Normal (p &le; .05)"
    dec_playtime = "Normal (p > .05)" if p_norm_playtime > 0.05 else "Tidak Normal (p &le; .05)"
    
    normality_html = f"""
    <table class="table-apa" style="width: 70%; margin: 15px auto 35px auto;">
      <thead>
        <tr style="border-top: 1.5px solid black; border-bottom: 1px solid black;">
          <th style="text-align: left; padding: 8px;">Variable (Paired Differences)</th>
          <th style="padding: 8px;">Shapiro-Wilk W</th>
          <th style="padding: 8px;">p-value</th>
          <th style="padding: 8px;">Decision</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="text-align: left; padding: 8px;">&Delta;score (&Delta;score_DDA - &Delta;score_NonDDA)</td>
          <td style="padding: 8px;">{shapiro_w_margin:.3f}</td>
          <td style="padding: 8px;">{_apa_interpret_p(p_norm_margin)}</td>
          <td style="padding: 8px;">{dec_margin}</td>
        </tr>
        <tr style="border-bottom: 1.5px solid black;">
          <td style="text-align: left; padding: 8px;">playtime (playtime_DDA - playtime_NonDDA)</td>
          <td style="padding: 8px;">{shapiro_w_playtime:.3f}</td>
          <td style="padding: 8px;">{_apa_interpret_p(p_norm_playtime)}</td>
          <td style="padding: 8px;">{dec_playtime}</td>
        </tr>
      </tbody>
    </table>
    <div style="font-size: 11px; text-align: center; font-family: 'Times New Roman'; margin-top: -25px; margin-bottom: 30px;">
      <i>Note.</i> N = 10 pairs. If p > .05, data is considered normally distributed (parametric test is appropriate).
    </div>
    """
    add_to_html("3c. Shapiro-Wilk Normality Test Results (Table 1b)", normality_html)

    # Hitung p-value one-tailed
    p_margin_val = p_margin / 2 if diff_margin.mean() < 0 else 1 - p_margin / 2
    p_playtime_val = p_playtime / 2 if diff_playtime.mean() > 0 else 1 - p_playtime / 2
    
    p_margin_str = _apa_interpret_p(p_margin_val).replace("p = ", "")
    p_playtime_str = _apa_interpret_p(p_playtime_val).replace("p = ", "")

    table2_html = f"""
    <table class="table-apa" style="width: 100%; border-collapse: collapse; font-family: 'Times New Roman'; margin-top: 15px; margin-bottom: 35px;">
      <thead>
        <tr style="border-top: 1.5px solid black; border-bottom: 1px solid black;">
          <th style="text-align: left; padding: 8px;" rowspan="2">Metric</th>
          <th colspan="2" style="border-bottom: 1px solid black; padding: 8px;">DDA Condition (n = 10)</th>
          <th colspan="2" style="border-bottom: 1px solid black; padding: 8px;">Non-DDA Condition (n = 10)</th>
          <th rowspan="2" style="padding: 8px;">Test Type</th>
          <th rowspan="2" style="padding: 8px;">Statistic</th>
          <th rowspan="2" style="padding: 8px;">p (One-Tailed)</th>
          <th rowspan="2" style="padding: 8px;">Cohen's d</th>
        </tr>
        <tr style="border-bottom: 1.5px solid black;">
          <th style="padding: 8px;">M</th>
          <th style="padding: 8px;">SD</th>
          <th style="padding: 8px;">M</th>
          <th style="padding: 8px;">SD</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="text-align: left; padding: 8px;">&Delta;score</td>
          <td style="padding: 8px;">{dda_margin.mean():.2f}</td>
          <td style="padding: 8px;">{dda_margin.std():.2f}</td>
          <td style="padding: 8px;">{nodda_margin.mean():.2f}</td>
          <td style="padding: 8px;">{nodda_margin.std():.2f}</td>
          <td style="padding: 8px;">{test_type_margin}</td>
          <td style="padding: 8px;">{t_margin:.3f}</td>
          <td style="padding: 8px;">{p_margin_str}</td>
          <td style="padding: 8px;">{cohen_margin:.2f}</td>
        </tr>
        <tr style="border-bottom: 1.5px solid black;">
          <td style="text-align: left; padding: 8px;">playtime (seconds)</td>
          <td style="padding: 8px;">{dda_playtime.mean():.2f}</td>
          <td style="padding: 8px;">{dda_playtime.std():.2f}</td>
          <td style="padding: 8px;">{nodda_playtime.mean():.2f}</td>
          <td style="padding: 8px;">{nodda_playtime.std():.2f}</td>
          <td style="padding: 8px;">{test_type_playtime}</td>
          <td style="padding: 8px;">{t_playtime:.3f}</td>
          <td style="padding: 8px;">{p_playtime_str}</td>
          <td style="padding: 8px;">{cohen_playtime:.2f}</td>
        </tr>
      </tbody>
    </table>
    <div style="font-size: 11px; text-align: left; font-family: 'Times New Roman'; margin-top: -25px; margin-bottom: 30px;">
      <i>Note.</i> N = 10 pairs. Test Type changes dynamically based on Shapiro-Wilk normality test results.
    </div>
    """
    
    add_to_html("4. Results of Paired t-Tests Comparing DDA and Non-DDA (Table 2)", table2_html)
    return paired_margin, paired_playtime, terminal_df



# ──────────────────────────────────────────────────────────────
# 3b. ANALISIS PENGARUH PENGALAMAN BERMAIN (EXPERIENCED VS INEXPERIENCED)
# ──────────────────────────────────────────────────────────────
def analyze_experience_influence(df, terminal_df):
    experience_map = {
        'Rizka': 'N', 'Dies Natalis': 'Y', 'Budi A': 'Y', 'Vanka': 'N', 'Budi2 A': 'N',
        'Budiono Siregar': 'Y', 'SAUQI A': 'N', 'rifal casmana': 'N', 'NHD B': 'Y', 'mumtaz': 'Y'
    }
    
    terminal_df = terminal_df.copy()
    terminal_df['AbsMargin'] = (terminal_df['scoreP1'] - terminal_df['scoreP2']).abs()
    
    human_moves = df[df['player'] == 1].dropna(subset=['thinkTime'])
    playtime_df = human_moves.groupby(['sessionId', 'participant', 'isDda'])['thinkTime'].sum().reset_index()
    playtime_df.rename(columns={'thinkTime': 'TotalPlaytime'}, inplace=True)
    
    terminal_df['Experience'] = terminal_df['participant'].map(experience_map)
    playtime_df['Experience'] = playtime_df['participant'].map(experience_map)
    
    part_margin = terminal_df.groupby(['participant', 'Experience', 'isDda'])['AbsMargin'].mean().unstack().reset_index()
    part_playtime = playtime_df.groupby(['participant', 'Experience', 'isDda'])['TotalPlaytime'].mean().unstack().reset_index()
    
    t3_rows = []
    for metric_name, data_df in [('&Delta;score', part_margin), ('playtime (seconds)', part_playtime)]:
        for dda_cond in [True, False]:
            cond_str = "DDA" if dda_cond else "Non-DDA"
            y_group = data_df[data_df['Experience'] == 'Y'][dda_cond].dropna()
            n_group = data_df[data_df['Experience'] == 'N'][dda_cond].dropna()
            
            _, p_shapiro_y = stats.shapiro(y_group)
            _, p_shapiro_n = stats.shapiro(n_group)
            
            if p_shapiro_y > 0.05 and p_shapiro_n > 0.05:
                _, p_levene = stats.levene(y_group, n_group)
                equal_var = p_levene > 0.05
                test_stat, p_t = stats.ttest_ind(y_group, n_group, equal_var=equal_var)
                test_type = "Ind. t-Test"
            else:
                test_stat, p_t = stats.mannwhitneyu(y_group, n_group, alternative='two-sided')
                test_type = "Mann-Whitney U"
            
            # Pooled SD for Cohen's d
            pooled_sd = np.sqrt(((len(y_group)-1)*y_group.var() + (len(n_group)-1)*n_group.var()) / (len(y_group)+len(n_group)-2))
            cohen_d = (y_group.mean() - n_group.mean()) / pooled_sd if pooled_sd != 0 else 0.0
            
            p_t_str = _apa_interpret_p(p_t).replace("p = ", "")
            
            t3_rows.append(f"""
            <tr>
              <td style="text-align: left; padding: 8px;">{metric_name} in {cond_str}</td>
              <td style="padding: 8px;">{y_group.mean():.2f}</td>
              <td style="padding: 8px;">{y_group.std():.2f}</td>
              <td style="padding: 8px;">{n_group.mean():.2f}</td>
              <td style="padding: 8px;">{n_group.std():.2f}</td>
              <td style="padding: 8px;">{test_type}</td>
              <td style="padding: 8px;">{test_stat:.3f}</td>
              <td style="padding: 8px;">{p_t_str}</td>
              <td style="padding: 8px;">{cohen_d:.2f}</td>
            </tr>
            """)
            
    table3_html = f"""
    <table class="table-apa" style="width: 100%; border-collapse: collapse; font-family: 'Times New Roman'; margin-top: 15px; margin-bottom: 35px;">
      <thead>
        <tr style="border-top: 1.5px solid black; border-bottom: 1px solid black;">
          <th style="text-align: left; padding: 8px;" rowspan="2">Metric and Condition</th>
          <th colspan="2" style="border-bottom: 1px solid black; padding: 8px;">Experienced (n = 5)</th>
          <th colspan="2" style="border-bottom: 1px solid black; padding: 8px;">Inexperienced (n = 5)</th>
          <th rowspan="2" style="padding: 8px;">Test Type</th>
          <th rowspan="2" style="padding: 8px;">Statistic</th>
          <th rowspan="2" style="padding: 8px;">p (Two-Tailed)</th>
          <th rowspan="2" style="padding: 8px;">Cohen's d</th>
        </tr>
        <tr style="border-bottom: 1.5px solid black;">
          <th style="padding: 8px;">M</th>
          <th style="padding: 8px;">SD</th>
          <th style="padding: 8px;">M</th>
          <th style="padding: 8px;">SD</th>
        </tr>
      </thead>
      <tbody>
        {"".join(t3_rows[:-1])}
        <tr style="border-bottom: 1.5px solid black;">
          {t3_rows[-1].replace('<tr>', '').replace('</tr>', '')}
        </tr>
      </tbody>
    </table>
    <div style="font-size: 11px; text-align: left; font-family: 'Times New Roman'; margin-top: -25px; margin-bottom: 30px;">
      <i>Note.</i> Cohen's d is calculated based on pooled standard deviation. Test Type changes dynamically based on Shapiro-Wilk normality test results.

    </div>
    """
    add_to_html("5a. Independent Samples t-Tests Comparing Experienced and Inexperienced Players (Table 3)", table3_html)

# ──────────────────────────────────────────────────────────────
# 4. VISUALIZATION & HTML EMBEDDING
# ──────────────────────────────────────────────────────────────
def generate_plots_and_html(paired_margin, paired_playtime, terminal_df, df):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_paths = {}

    # 1a. Bar Chart H1 (Margin / Δscore)
    if not paired_margin.empty:
        plot_df = paired_margin.reset_index().melt(id_vars='participant', value_name='Margin', var_name='isDda')
        plt.figure(figsize=(6, 4))
        sns.barplot(data=plot_df, x='isDda', y='Margin', errorbar='se', hue='isDda', palette=['#e74c3c', '#2ecc71'], legend=False, capsize=0.1)
        plt.title('Mean Δscore')
        plt.xticks([0, 1], ['Non-DDA', 'DDA'])
        plt.ylabel('Δscore')
        plt.xlabel('Kondisi')
        plt.tight_layout()
        margin_bar_path = f'chart_H1_margin_bar_{ts}.png'
        plt.savefig(os.path.join(script_dir, margin_bar_path))
        plt.close()
        plot_paths['mean_deltascore'] = margin_bar_path

    # 1b. Boxplot H1 (Margin / Δscore)
    if not paired_margin.empty:
        plot_df = paired_margin.reset_index().melt(id_vars='participant', value_name='Margin', var_name='isDda')
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=plot_df, x='isDda', y='Margin', hue='isDda', palette=['#e74c3c', '#2ecc71'], legend=False, width=0.5, showfliers=False)
        sns.stripplot(data=plot_df, x='isDda', y='Margin', color='black', alpha=0.6, jitter=True, size=5)
        # plt.title('Distribusi Δscore (Game Balance)')
        plt.title('Distribusi Δscore')
        plt.xticks([0, 1], ['Non-DDA', 'DDA'])
        plt.ylabel('Δscore')
        plt.xlabel('Kondisi')
        plt.tight_layout()
        margin_path = f'chart_H1_margin_box_{ts}.png'
        plt.savefig(os.path.join(script_dir, margin_path))
        plt.close()
        plot_paths['deltascore_distribution'] = margin_path
        
    # 2a. Bar Chart H2 (playtime)
    if not paired_playtime.empty:
        plot_df2 = paired_playtime.reset_index().melt(id_vars='participant', value_name='Playtime', var_name='isDda')
        plt.figure(figsize=(6, 4))
        sns.barplot(data=plot_df2, x='isDda', y='Playtime', errorbar='se', hue='isDda', palette=['#e74c3c', '#2ecc71'], legend=False, capsize=0.1)
        plt.title('Mean playtime')
        plt.xticks([0, 1], ['Non-DDA', 'DDA'])
        plt.ylabel('Total playtime (detik)')
        plt.xlabel('Kondisi')
        plt.tight_layout()
        playtime_bar_path = f'chart_H2_playtime_bar_{ts}.png'
        plt.savefig(os.path.join(script_dir, playtime_bar_path))
        plt.close()
        plot_paths['mean_playtime'] = playtime_bar_path

    # 2b. Boxplot H2 (playtime)
    if not paired_playtime.empty:
        plot_df2 = paired_playtime.reset_index().melt(id_vars='participant', value_name='Playtime', var_name='isDda')
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=plot_df2, x='isDda', y='Playtime', hue='isDda', palette=['#e74c3c', '#2ecc71'], legend=False, width=0.5, showfliers=False)
        sns.stripplot(data=plot_df2, x='isDda', y='Playtime', color='black', alpha=0.6, jitter=True, size=5)
        # plt.title('Distribusi playtime')
        plt.title('Distribusi playtime')
        plt.xticks([0, 1], ['Non-DDA', 'DDA'])
        plt.ylabel('playtime (detik)')
        plt.xlabel('Kondisi')
        plt.tight_layout()
        playtime_path = f'chart_H2_playtime_box_{ts}.png'
        plt.savefig(os.path.join(script_dir, playtime_path))
        plt.close()
        plot_paths['playtime_distribution'] = playtime_path

    # 4. Grafik Garis Trajektori V-Value (Turn-by-Turn)
    v_moves = df[(df['player'] == -1) & (df['v'].notnull())]
    if not v_moves.empty:
        plt.figure(figsize=(7, 4))
        # Filter turn <= 100 agar grafiknya tidak terlalu panjang/gepeng ke kanan
        plot_v_moves = v_moves[v_moves['turn'] <= 100].copy()
        plot_v_moves['isDda_str'] = plot_v_moves['isDda'].map({True: 'DDA', False: 'Non-DDA'})
        sns.lineplot(data=plot_v_moves, x='turn', y='v', hue='isDda_str', palette={'DDA': '#2ecc71', 'Non-DDA': '#e74c3c'}, errorbar=None, linewidth=2)
        plt.title('Trajektori Rata-Rata Evaluasi Posisi AI (v) Sepanjang Giliran')
        plt.xlabel('Giliran (Turn)')
        plt.ylabel('Nilai Evaluasi AI (v)')
        plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.legend(title='Kondisi')
        plt.tight_layout()
        traj_path = f'chart_v_trajectory_{ts}.png'
        plt.savefig(os.path.join(script_dir, traj_path))
        plt.close()
        plot_paths['v_trajectory'] = traj_path

    # 4b. Grafik Trajektori Score Difference per Turn (P1 - P2)
    score_traj = df[df['scoreP1'].notnull() & df['scoreP2'].notnull()].copy()
    if not score_traj.empty:
        score_traj['ScoreDiff'] = score_traj['scoreP1'] - score_traj['scoreP2']
        plt.figure(figsize=(7, 4))
        plot_score_traj = score_traj[score_traj['turn'] <= 100].copy()
        plot_score_traj['isDda_str'] = plot_score_traj['isDda'].map({True: 'DDA', False: 'Non-DDA'})
        sns.lineplot(data=plot_score_traj, x='turn', y='ScoreDiff', hue='isDda_str', palette={'DDA': '#2ecc71', 'Non-DDA': '#e74c3c'}, errorbar=None, linewidth=2)
        plt.title('Trajektori Selisih Skor (Human - AI) Sepanjang Permainan')
        plt.xlabel('Giliran (Turn)')
        plt.ylabel('scoreP1 - scoreP2')
        plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.legend(title='Kondisi')
        plt.tight_layout()
        scorediff_path = f'chart_scorediff_trajectory_{ts}.png'
        plt.savefig(os.path.join(script_dir, scorediff_path))
        plt.close()
        plot_paths['scorediff_trajectory'] = scorediff_path

    # 4c. Boxplot thinkTime per Langkah (Distribusi thinkTime per Langkah)
    human_moves_all = df[df['player'] == 1].dropna(subset=['thinkTime']).copy()
    if not human_moves_all.empty:
        human_moves_all['isDda_str'] = human_moves_all['isDda'].map({True: 'DDA', False: 'Non-DDA'})
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=human_moves_all, x='isDda_str', y='thinkTime', hue='isDda_str', palette={'DDA': '#2ecc71', 'Non-DDA': '#e74c3c'}, legend=False, width=0.5, showfliers=False)
        sample_moves = human_moves_all.sample(min(500, len(human_moves_all)), random_state=42)
        sns.stripplot(data=sample_moves, x='isDda_str', y='thinkTime', color='black', alpha=0.3, jitter=True, size=3)
        plt.title('Distribusi thinkTime per Langkah\n(thinkTime per turn)')
        plt.xlabel('Kondisi')
        plt.ylabel('thinkTime (detik)')
        plt.tight_layout()
        tt_path = f'chart_thinktime_per_turn_box_{ts}.png'
        plt.savefig(os.path.join(script_dir, tt_path))
        plt.close()
        plot_paths['thinktime_per_turn'] = tt_path
    # 4d. Grafik Trajektori thinkTime per Turn (P1)
    if not human_moves_all.empty:
        plt.figure(figsize=(7, 4))
        plot_tt_traj = human_moves_all[human_moves_all['turn'] <= 100].copy()
        sns.lineplot(data=plot_tt_traj, x='turn', y='thinkTime', hue='isDda_str', palette={'DDA': '#2ecc71', 'Non-DDA': '#e74c3c'}, errorbar='se', linewidth=2)
        plt.title('Trajektori Rata-Rata thinkTime Sepanjang Permainan')
        plt.xlabel('Giliran (Turn)')
        plt.ylabel('thinkTime (detik)')
        plt.legend(title='Kondisi')
        plt.tight_layout()
        tt_traj_path = f'chart_thinktime_trajectory_{ts}.png'
        plt.savefig(os.path.join(script_dir, tt_traj_path))
        plt.close()
        plot_paths['thinktime_trajectory'] = tt_traj_path

    # 5. Heatmap Frekuensi Move Pilihan (0-6)
    moves_df = df[df['move'].isin(range(7))].copy()
    if not moves_df.empty:
        # Group and calculate percentages per (isDda, player)
        move_counts = moves_df.groupby(['isDda', 'player', 'move']).size().unstack(fill_value=0)
        move_pcts = move_counts.div(move_counts.sum(axis=1), axis=0) * 100
        
        # Rename index for better display
        move_pcts.index = move_pcts.index.map(lambda x: (
            f"DDA | AI (P2)" if x[0] and x[1] == -1 else
            f"DDA | Human (P1)" if x[0] and x[1] == 1 else
            f"Non-DDA | AI (P2)" if not x[0] and x[1] == -1 else
            f"Non-DDA | Human (P1)"
        ))
        
        plt.figure(figsize=(8, 5))
        sns.heatmap(move_pcts, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Persentase Pilihan (%)'}, linewidths=0.5)
        plt.title('Heatmap Preferensi Pemilihan Lubang (Move 0-6)\n(Persentase per Kondisi & Pemain)')
        plt.ylabel('Kondisi & Pemain')
        plt.xlabel('Nomor Lubang (0-6)')
        plt.tight_layout()
        heatmap_path = f'chart_moves_heatmap_{ts}.png'
        plt.savefig(os.path.join(script_dir, heatmap_path))
        plt.close()
        plot_paths['preferensi_lubang'] = heatmap_path

    # Embed images directly inside HTML as Base64 to make it a standalone portable file
    img_html = "<h3>6. Visualisasi Hasil Penelitian</h3><div style='display: flex; flex-wrap: wrap; gap: 20px;'>"
    for name, path in plot_paths.items():
        abs_path = os.path.join(script_dir, path)
        if os.path.exists(abs_path):
            with open(abs_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                img_html += f"""
                <div style='flex: 1; min-width: 300px; max-width: 450px; text-align: center; border: 1px solid #ddd; padding: 10px; border-radius: 4px; margin-bottom: 20px;'>
                    <img src='data:image/png;base64,{encoded_string}' style='width: 100%; height: auto;' />
                    <p style='font-size: 12px; color: #666; font-style: italic; margin-top: 8px;'>Gambar: {name.replace('_', ' ').capitalize()}</p>
                </div>
                """
    img_html += "</div>"
    add_to_html("Visualisasi", img_html)

    # HTML assembly with APA 7th Edition style override
    css = """
    <style>
        body { font-family: "Times New Roman", Times, serif; margin: 40px; color: #222; line-height: 1.6; }
        h1 { font-family: Arial, sans-serif; color: #2c3e50; border-bottom: 3px solid #2c3e50; padding-bottom: 10px; text-align: center; }
        h3 { font-family: Arial, sans-serif; color: #34495e; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 40px; }
        ul { padding-left: 20px; }
        
        /* APA Style Tables Overrides */
        table { border-collapse: collapse; width: 100%; margin-top: 15px; margin-bottom: 35px; font-size: 13px; text-align: center; }
        th, td { padding: 10px; border-bottom: 0.5px solid #ccc; }
        th { font-weight: bold; border-top: 1.5px solid black; border-bottom: 1.5px solid black; background-color: transparent !important; }
        tr:last-child td { border-bottom: 1.5px solid black; }
        td, th { border-left: none !important; border-right: none !important; }
        .table { border: none !important; }
    </style>
    """
    html_out = f"<html><head>{css}</head><body><h1>Hasil Analisis DDA (APA Style & SPSS Style Output)</h1>{HTML_CONTENT}</body></html>"
    
    html_path = os.path.join(script_dir, 'spss_style_output.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_out)
    print(f"\n[INFO] Laporan HTML dengan chart tersemat berhasil diekspor ke: {html_path}")

if __name__ == "__main__":
    data = load_and_preprocess()
    if data is not None:
        term_df = analyze_descriptive(data)
        marg, play, term = test_hypotheses(data, term_df)
        analyze_experience_influence(data, term_df)
        generate_plots_and_html(marg, play, term, data)

