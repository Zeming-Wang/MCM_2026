import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import cm

# --- 1. Core Logic Copied from Elimination Risk.py ---

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    exp_x = np.exp(x - float(np.max(x)))
    s = float(np.sum(exp_x))
    return exp_x / (s if s > 0 else 1.0)


def _prepare_predictions(df_pred: pd.DataFrame) -> pd.DataFrame:
    df = df_pred.copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype(int)
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype(int)
    df["Name"] = df["Name"].astype(str)
    df["Fan_Vote_Percent"] = pd.to_numeric(df.get("Fan_Vote_Percent"), errors="coerce").fillna(0.0)
    df["V_Base"] = pd.to_numeric(df.get("V_Base"), errors="coerce").fillna(0.0)
    df["Delta_V"] = pd.to_numeric(df.get("Delta_V"), errors="coerce").fillna(0.0)
    return df


def _safe_sigma(v_base: np.ndarray, delta_v: np.ndarray) -> float:
    sigma = float(np.std(delta_v))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(v_base))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1e-6
    return sigma


def compute_elimination_risk(
    df_pred: pd.DataFrame,
    alpha: float = 0.5,
    draws: int = 800,
    seed: int = 0,
) -> pd.DataFrame:
    df = _prepare_predictions(df_pred)
    rng = np.random.default_rng(int(seed))
    out_rows = []

    for (season, week), g in df.groupby(["Season", "Week"], sort=True):
        if len(g) <= 1:
            continue
        names = g["Name"].astype(str).to_numpy()
        v_base = g["V_Base"].to_numpy(dtype=float)
        delta_v = g["Delta_V"].to_numpy(dtype=float)

        point = g["Fan_Vote_Percent"].to_numpy(dtype=float)
        if float(np.sum(point)) <= 0:
            point = _softmax(v_base + alpha * delta_v)
        else:
            point = point / float(np.sum(point))

        sigma = _safe_sigma(v_base, delta_v)
        n = len(g)
        fan_samples = np.zeros((int(draws), n), dtype=float)

        for i in range(int(draws)):
            eps = rng.normal(0.0, sigma, size=n)
            raw = v_base + alpha * (delta_v + eps)
            fan_samples[i] = _softmax(raw)

        elim_idx = np.argmin(fan_samples, axis=1)
        elim_counts = np.bincount(elim_idx, minlength=n).astype(float) / float(draws)

        for i in range(n):
            out_rows.append(
                {
                    "Season": int(season),
                    "Week": int(week),
                    "Name": str(names[i]),
                    "Elimination_Prob": float(elim_counts[i]),
                }
            )

    return pd.DataFrame(out_rows)

# --- 2. Visualization Logic ---

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.0), *zip(x, y), (x[-1], 0.0)]

def plot_3d_waterfall(df_risk: pd.DataFrame, season: int, out_dir: str, top_k: int = 8):
    sub = df_risk[df_risk["Season"] == int(season)].copy()
    if sub.empty:
        print(f"No data for Season {season}")
        return

    # Pivot to get matrix: index=Week, columns=Name, values=Prob
    pivot = (
        sub.pivot_table(index="Week", columns="Name", values="Elimination_Prob", aggfunc="mean")
        .fillna(0.0)
    )
    
    # Filter top K riskiest on average
    avg_risk = pivot.mean(axis=0).sort_values(ascending=False)
    names = avg_risk.head(int(top_k)).index.tolist()
    if not names:
        return
    
    # Re-order names reverse so the highest risk is at the front or back as preferred
    # In the reference image, it seems sorted. Let's sort by risk.
    # We want the "front" (lowest y-coord in 3D) to be visible.
    # Usually, we plot from back to front to avoid occlusion issues if not using transparency well,
    # but Matplotlib handles this reasonably well.
    # Let's keep them sorted.
    names = names[::-1] # Reverse so highest risk is at the "front" (last plotted usually means on top in 2D, but in 3D z-order matters)
    
    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    weeks = pivot.index.to_numpy()
    
    # Prepare data for PolyCollection
    verts = []
    # We need to map names to y-coordinates (0, 1, 2, ...)
    ys = range(len(names))
    
    for name in names:
        probs = pivot[name].to_numpy()
        # Add 0 at the start and end for the polygon base if needed, but polygon_under_graph handles it
        verts.append(polygon_under_graph(weeks, probs))

    # Color map - replicate the purple to yellow gradient
    # The image goes from purple (front/low) to yellow (back/high)
    # or vice versa. The provided image shows Group 1 (front) is purple, Group 7 (back) is yellow.
    # So we want the first items in our loop (which correspond to y=0) to be purple.
    # We are iterating names. Let's map 0..N to the colormap.
    
    # Use 'plasma' or 'viridis' - plasma goes purple to yellow.
    cmap = plt.get_cmap('plasma') 
    # Create colors for each ribbon
    colors = [cmap(i / (len(names) - 1)) if len(names) > 1 else cmap(0.5) for i in range(len(names))]
    
    poly = PolyCollection(verts, facecolors=colors, edgecolors='gray', linewidths=0.5, alpha=0.7)
    
    # Set the z-limit of the polygons (which is actually the Y axis in 3D plot terms of Matplotlib)
    # Matplotlib 3D:
    # zs argument in add_collection3d sets the position on the Z axis (which we will map to our "Y" category axis)
    # zdir='y' means the polygons are in the X-Z plane, stacked along Y.
    
    # Wait, standard convention in PolyCollection 3d demo:
    # ax.add_collection3d(poly, zs=zs, zdir='y')
    # Here 'zs' are the positions along the 'zdir' axis.
    
    ax.add_collection3d(poly, zs=ys, zdir='y')

    # Set limits
    ax.set_xlim(weeks.min(), weeks.max())
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_zlim(0, 1.1) # Probabilities are 0-1.0

    # Labels
    ax.set_xlabel('Week', fontsize=12, labelpad=10)
    # ax.set_ylabel('Contestant', fontsize=12, labelpad=10) # Removed per user request
    ax.set_zlabel('Elimination Probability', fontsize=12, labelpad=10)
    
    # Set Y ticks to names
    ax.set_yticks(ys)
    ax.set_yticklabels(names, rotation=-15, va='center', ha='left', fontsize=9)
    
    # Title
    ax.set_title(f'3D Waterfall Chart of Elimination Risk - Season {season}', fontsize=16, pad=20)

    # View angle - try to match the image (isometric-ish)
    ax.view_init(elev=30, azim=-60)
    
    # Add text labels for peak values like in the image
    # The image has labels on the peaks.
    for i, name in enumerate(names):
        probs = pivot[name]
        # Label local maxima or just some points?
        # Image shows labels on almost every point. That might be too crowded.
        # Let's label the max point and maybe start/end.
        # Or label points > 0.1 to avoid clutter
        
        for w_idx, w in enumerate(weeks):
            val = probs.loc[w]
            if val > 0.05: # Threshold to avoid labeling zero-risk
                # Coordinates: x=w, y=i, z=val
                # Text in 3D
                label_text = f"{val:.1%}" if val < 100 else f"{val:.0f}" # Format as percentage
                label_text = f"{val*100:.1f}" # Matches image style "94.2", "106.5" etc. (assuming those are counts or values)
                # Our values are probabilities 0-1. So 0.942 -> 94.2
                
                # Check if this point is a local peak or significant
                # For now, let's just label values > 0.2 to avoid clutter, or every Nth point
                ax.text(w, i, val, label_text, fontsize=7, ha='center', va='bottom', color='black')

    # Grid styling
    ax.grid(True, linestyle="--", alpha=0.3)
    # White background pane
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"elimination_risk_3d_waterfall_season_{season}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Adjust paths to match project structure
    # This file is in src/features, so .. -> src, .. -> project_root
    
    pred_path = os.path.join(project_root, "data", "processed", "model1_fan_vote_predictions.csv")
    pred_path_fallback = os.path.join(project_root, "results", "model1_fan_vote_predictions_subset.csv")
    out_dir = os.path.join(project_root, "charts")

    if os.path.exists(pred_path):
        print(f"Loading {pred_path}...")
        df_pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    elif os.path.exists(pred_path_fallback):
        print(f"Loading {pred_path_fallback}...")
        df_pred = pd.read_csv(pred_path_fallback, encoding="utf-8-sig")
    else:
        # Fallback to absolute path hardcoded if relative fails (for safety)
        hardcoded_path = r"d:\MCM_2026_O\data\processed\model1_fan_vote_predictions.csv"
        if os.path.exists(hardcoded_path):
             df_pred = pd.read_csv(hardcoded_path, encoding="utf-8-sig")
        else:
            raise FileNotFoundError("Could not find predictions file.")

    os.makedirs(out_dir, exist_ok=True)

    print("Computing elimination risks...")
    df_risk = compute_elimination_risk(df_pred, alpha=0.5, draws=800, seed=0)

    target_seasons = [2, 4, 11, 27]
    
    for season in target_seasons:
        print(f"Plotting Season {season}...")
        plot_3d_waterfall(df_risk, season, out_dir, top_k=7) # Top 7 like the image shows Group1-7

if __name__ == "__main__":
    main()
