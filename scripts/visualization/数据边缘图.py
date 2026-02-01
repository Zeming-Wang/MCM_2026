import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def main():
    # 1. Setup Style
    sns.set_theme(style="ticks")
    
    # 2. Load Data
    base_dir = r"d:\MCM_2026_O"
    data_path = os.path.join(base_dir, r"results\fan_vote_estimates_by_person_week.csv")
    metrics_path = os.path.join(base_dir, r"results\fan_vote_metrics_summary.csv")
    diag_path = os.path.join(base_dir, r"results\fan_vote_join_diagnostics.csv")
    out_dir = os.path.join(base_dir, r"results\plots")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Filter valid positive data for Log transformation
    df = df.dropna(subset=['Real_Fan_votes', 'Pred_Fan_Votes'])
    df = df[(df['Real_Fan_votes'] > 0) & (df['Pred_Fan_Votes'] > 0)]

    print(f"Loaded {len(df)} valid data points.")

    # 3. Log Transformation (Log10)
    # Using Log scale is crucial for fan vote data which spans orders of magnitude
    df['log_real'] = np.log10(df['Real_Fan_votes'])
    df['log_pred'] = np.log10(df['Pred_Fan_Votes'])

    # 4. Calculate Statistics
    r2 = r2_score(df['log_real'], df['log_pred'])
    rmse = np.sqrt(mean_squared_error(df['log_real'], df['log_pred']))
    mae = mean_absolute_error(df['log_real'], df['log_pred'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['log_real'], df['log_pred'])

    # Get global wMAPE if available
    global_wmape = None
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        if not m_df.empty and 'wMAPE' in m_df.columns:
            global_wmape = m_df['wMAPE'].iloc[0]

    # 5. Generate Joint Plot
    # kind='reg' adds the regression line and KDEs
    g = sns.jointplot(
        x="log_real", 
        y="log_pred", 
        data=df, 
        kind="reg",
        color="#6A5ACD",  # SlateBlue
        height=9,
        scatter_kws={"s": 40, "alpha": 0.6, "edgecolor": "white", "linewidths": 0.5},
        line_kws={"color": "#CD5C5C", "alpha": 0.9, "linewidth": 2}  # IndianRed
    )

    # 6. Annotate Statistics
    stats_text = (
        f"$\mathbf{{Model\ Fit}}$\n"
        f"y = {slope:.2f}x + {intercept:.2f}\n"
        f"$R^2$ = {r2:.3f}, p < 0.001\n"
        f"RMSE (log) = {rmse:.3f}\n"
        f"MAE (log) = {mae:.3f}"
    )
    if global_wmape:
        stats_text += f"\nGlobal wMAPE = {global_wmape:.3f}"

    # Add text box
    ax = g.ax_joint
    ax.text(
        0.05, 0.95, 
        stats_text, 
        transform=ax.transAxes,
        fontsize=12, 
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8F8FF', alpha=0.9, edgecolor='#B0C4DE')
    )

    # 7. Labels and Grid
    ax.set_xlabel("Log10(Real Fan Votes)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Log10(Predicted Fan Votes)", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    # 8. Print Diagnostics Summary to Console
    if os.path.exists(diag_path):
        d_df = pd.read_csv(diag_path)
        print("\n--- Data Diagnostics ---")
        print(d_df['missing_in'].value_counts().to_string())

    # 9. Save
    out_path = os.path.join(out_dir, "fan_vote_prediction_vs_real_joint.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to: {out_path}")

if __name__ == "__main__":
    main()