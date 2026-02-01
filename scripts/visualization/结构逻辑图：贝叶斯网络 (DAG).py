import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _node(ax, xy, text, fc="#F0F4F8", ec="#1A86F2"):
    x, y = xy
    w, h = 1.7, 0.55
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.4,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


def _arrow(ax, a, b, color="#2DC0CB"):
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, shrinkA=10, shrinkB=12),
    )


def plot_bayesian_network(output_path: str) -> str:
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(12.0, 6.6), dpi=170)

    pos = {
        "Industry": (0.0, 2.0),
        "Age": (0.0, 1.0),
        "FanBase": (0.0, 0.0),
        "Pref": (3.0, 1.0),
        "Vol": (3.0, 0.0),
        "DeltaV": (6.0, 0.6),
    }
    labels = {
        "Industry": "行业\n(Industry)",
        "Age": "年龄\n(Age)",
        "FanBase": "初始粉丝基数\n(Fan Base)",
        "Pref": "潜在偏好\n(Latent Pref)",
        "Vol": "投票波动\n(Volatility)",
        "DeltaV": "残差校准\n(Residual ΔV)",
    }

    for k, xy in pos.items():
        _node(ax, xy, labels[k])

    _arrow(ax, pos["Industry"], pos["Pref"])
    _arrow(ax, pos["Age"], pos["Pref"])
    _arrow(ax, pos["FanBase"], pos["Pref"])
    _arrow(ax, pos["Pref"], pos["DeltaV"])
    _arrow(ax, pos["Vol"], pos["DeltaV"])

    ax.text(1.55, 1.7, r"$\mu_{\mathrm{pref}}$", fontsize=13, color="#1D4ED8")
    ax.text(4.35, 1.15, r"$\sigma_{\mathrm{residual}}$", fontsize=13, color="#B91C1C")

    ax.set_title("Bayesian Residual Architecture (DAG)")
    ax.set_xlim(-1.4, 7.4)
    ax.set_ylim(-0.8, 2.6)
    ax.axis("off")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "scripts", "visualization", "outputs")
    out_path = os.path.join(out_dir, "bayesian_network_dag.png")
    plot_bayesian_network(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
