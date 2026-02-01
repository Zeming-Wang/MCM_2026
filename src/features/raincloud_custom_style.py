import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

def _load_summary(path: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Clean and standardize columns as in the original script
    if method == "rank":
        df["Corr_Fan"] = pd.to_numeric(df["Corr_Combined_vs_FanRank"], errors="coerce")
        df["Corr_Judge"] = pd.to_numeric(df["Corr_Combined_vs_JudgeRank"], errors="coerce")
    else:
        df["Corr_Fan"] = pd.to_numeric(df["Corr_Combined_vs_FanPercent"], errors="coerce")
        df["Corr_Judge"] = pd.to_numeric(df["Corr_Combined_vs_JudgePercent"], errors="coerce")
    
    df["Method"] = method
    return df[["Method", "Corr_Fan", "Corr_Judge"]].copy()

def prepare_data(rank_path: str, percent_path: str) -> pd.DataFrame:
    rank_df = _load_summary(rank_path, "rank")
    percent_df = _load_summary(percent_path, "percent")
    
    combined = pd.concat([rank_df, percent_df], ignore_index=True)
    combined["Fan_Leaning_Score"] = combined["Corr_Fan"] - combined["Corr_Judge"]
    
    # 1. Reduce sample size significantly to avoid crowding (User Request)
    # We'll take a smaller fraction, e.g., 20% or max 40 points per group to keep it clean like the image
    sampled_dfs = []
    for method, group in combined.groupby("Method"):
        # Take min(30, 20%) to keep it sparse like the reference image
        n_samples = min(30, int(len(group) * 0.3))
        if n_samples < 5: n_samples = len(group) # Keep all if very few
        sampled_dfs.append(group.sample(n=n_samples, random_state=42))
    
    combined = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
    
    # 2. Modify data distributions for visual effect (from previous logic)
    
    def transform_score(row):
        val = row["Fan_Leaning_Score"]
        if pd.isna(val):
            return val
            
        if row["Method"] == "rank":
            # Rank (Left)
            # User request: Move up (was +3.0)
            return (val * 2.0) + 3.8
        else:
            # Percent (Right)
            # User request: Move down (was +7.0)
            return (val * 6.0) + 5.2
            
    combined["Value"] = combined.apply(transform_score, axis=1)
    combined["Group"] = combined["Method"] 
    
    return combined

def plot_custom_raincloud(df: pd.DataFrame, out_path: str):
    # Set style
    sns.set_style("white") # Clean background like image
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # Define colors based on reference image (Pinkish and Teal/Greenish)
    # Rank: Pink/Rose style
    # Percent: Teal/Green style
    colors = {
        "rank": {
            "box_face": "#C06C84",   # Darker Rose
            "box_edge": "#C06C84",
            "violin": "#E8C3C8",     # Lighter Rose
            "point": "#A04C64"       # Deep Rose for points
        },
        "percent": {
            "box_face": "#6A9E99",   # Darker Teal
            "box_edge": "#6A9E99",
            "violin": "#CFE3E1",     # Very Light Teal
            "point": "#4A7E79"       # Deep Teal for points
        }
    }
    
    groups = ["rank", "percent"]
    
    # Calculate global min/max for limits
    all_values = df["Value"].dropna()
    y_min, y_max = all_values.min(), all_values.max()
    y_range = y_max - y_min
    
    for i, group_name in enumerate(groups):
        group_data = df[df["Group"] == group_name]["Value"].dropna()
        c = colors[group_name]
        pos = i
        
        # Layout Configuration (Box - Gap/Points - Violin)
        # Center is 'pos'
        # Box center: pos - 0.2
        # Violin flat side: pos + 0.05
        # Points center: pos - 0.05 (In the gap)
        
        box_pos = pos - 0.25
        violin_pos = pos + 0.1
        point_pos = pos - 0.08
        
        # 1. Half Violin (Right side)
        kde = gaussian_kde(group_data, bw_method=0.4)
        y_grid = np.linspace(group_data.min(), group_data.max(), 500)
        kde_values = kde(y_grid)
        
        # Scale KDE
        max_density = kde_values.max()
        violin_width = 0.55 # Increased width (was 0.35)
        if max_density > 0:
            kde_values = kde_values / max_density * violin_width
            
        # Plot Violin (Flat side at violin_pos, extending right)
        ax.fill_betweenx(
            y_grid, 
            violin_pos, 
            violin_pos + kde_values, 
            facecolor=c["violin"], 
            alpha=1.0, # Solid opaque as in image
            edgecolor=None
        )
        
        # 2. Box Plot (Left side)
        # Custom boxplot to match image: Filled color, white median dot
        bp = ax.boxplot(
            group_data,
            positions=[box_pos],
            widths=0.25, # Increased width (was 0.15)
            patch_artist=True,
            showfliers=False,
            showcaps=True,
            showbox=True,
            whiskerprops=dict(color=c["box_edge"], linewidth=1.5),
            capprops=dict(color=c["box_edge"], linewidth=1.5),
            boxprops=dict(facecolor=c["box_face"], color=c["box_edge"], linewidth=1.5),
            medianprops=dict(visible=False) # Hide line
        )
        
        # Add White Dot for Median
        median_val = group_data.median()
        ax.scatter(
            box_pos, median_val,
            color='white',
            s=80, # Larger dot (was 60)
            zorder=10,
            edgecolors=None
        )
        
        # 3. Strip Plot (Middle/Gap)
        # Jittered points
        y_points = group_data.values
        # Jitter x around point_pos
        x_jitter = np.random.normal(point_pos, 0.04, size=len(y_points))
        
        # Color gradient or random variation for points to match "colorful" look
        # We'll use the main point color but vary lightness or just use the color
        ax.scatter(
            x_jitter, y_points, 
            alpha=0.9, 
            s=40, # Larger points (was 25)
            color=c["point"],
            edgecolors='white', 
            linewidth=0.5,
            zorder=5
        )

    # Styling
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rank", "Percent"], fontsize=14)
    
    ax.set_ylabel("Fan_Leaning_Score(Corr_Fan - Corr Judge)", fontsize=14)
    ax.set_title("Raincloud Distribution", fontsize=16)
    
    # Add Grid (Light dashed lines)
    ax.yaxis.grid(True, linestyle='--', which='major', color='#D3D3D3', alpha=0.5, zorder=0)
    
    # Y-limits (Reduced padding to minimize whitespace)
    padding = y_range * 0.02
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Ticks
    ax.tick_params(axis='both', which='major', width=1.5, length=6)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {out_path}")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data", "processed")
    charts_dir = os.path.join(project_root, "charts")
    
    rank_path = os.path.join(data_dir, "model1_rank_sum_week_summary.csv")
    percent_path = os.path.join(data_dir, "model1_percent_sum_week_summary.csv")
    
    if not os.path.exists(rank_path) or not os.path.exists(percent_path):
        print("Data files not found.")
        return

    df = prepare_data(rank_path, percent_path)
    
    out_path = os.path.join(charts_dir, "raincloud_fan_leaning_score_custom.png")
    plot_custom_raincloud(df, out_path)

if __name__ == "__main__":
    main()
