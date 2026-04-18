"""
Generate training dynamics plot: loss/metric curves over epochs.
Appendix figure.

Reads from: runs/*/run.log (parsed training metrics)
Output: paper/figures/fig_training.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# TODO: Parse run.log files to extract epoch-by-epoch metrics
# Suggested layout: 2x3 grid (one per task), each showing:
# - Left y-axis: loss components (task, fidelity, range)
# - Right y-axis: validation AUROC

STYLE_PATH = os.path.join(os.path.dirname(__file__), 'neurips.mplstyle')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'fig_training.pdf')

plt.style.use(STYLE_PATH)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0))

    for ax, task in zip(axes, ['Mortality', 'AKI', 'Sepsis']):
        ax.set_title(task, fontsize=8)
        ax.set_xlabel('Epoch')
        ax.text(0.5, 0.5, 'TODO', ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')

    axes[0].set_ylabel('Loss / AUROC')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
