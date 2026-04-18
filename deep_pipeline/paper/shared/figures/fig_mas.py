"""
Generate Multi-Architecture Stability (MAS) chart.
Shows LSTM-trained translator performance on frozen GRU and TCN.
Figure 3 or 4 in paper.

Reads from: experiments/results/mas_*.json
Output: paper/figures/fig_mas.pdf
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')
STYLE_PATH = os.path.join(os.path.dirname(__file__), 'neurips.mplstyle')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'fig_mas.pdf')

plt.style.use(STYLE_PATH)

# --- Data ---
# AUROC deltas: translator trained on LSTM, evaluated on each architecture
TASKS = ['Mortality', 'AKI', 'Sepsis']
ARCHITECTURES = {
    'LSTM (trained)': {'Mortality': 0.0476, 'AKI': 0.0556, 'Sepsis': 0.0512},
    'GRU (zero-shot)': {'Mortality': 0.0404, 'AKI': 0.0311, 'Sepsis': 0.0240},
    'TCN (zero-shot)': {'Mortality': 0.0342, 'AKI': 0.0316, 'Sepsis': 0.0442},
}


def main():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    x = np.arange(len(TASKS))
    n = len(ARCHITECTURES)
    width = 0.25

    for i, (arch, deltas) in enumerate(ARCHITECTURES.items()):
        values = [deltas[t] for t in TASKS]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=arch, zorder=3)

    ax.set_ylabel(r'$\Delta$ AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Add percentage annotations
    for i, task in enumerate(TASKS):
        lstm_val = ARCHITECTURES['LSTM (trained)'][task]
        for j, (arch, deltas) in enumerate(ARCHITECTURES.items()):
            if arch == 'LSTM (trained)':
                continue
            pct = deltas[task] / lstm_val * 100
            offset = (j - n / 2 + 0.5) * width
            ax.annotate(f'{pct:.0f}%', (i + offset, deltas[task]),
                        ha='center', va='bottom', fontsize=6, color='gray')

    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
