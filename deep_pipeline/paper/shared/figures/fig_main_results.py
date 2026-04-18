"""
Generate main results bar chart: all tasks x all methods.
Figure 2 in paper (experiments section).

Reads from: experiments/results/*.json
Output: paper/figures/fig_main_results.pdf
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')
STYLE_PATH = os.path.join(os.path.dirname(__file__), 'neurips.mplstyle')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'fig_main_results.pdf')

plt.style.use(STYLE_PATH)

# --- Data: AUROC deltas from frozen baseline ---
# TODO: Replace with actual values loaded from result JSONs + bootstrap CIs

TASKS = ['Mortality', 'AKI', 'Sepsis']
BASELINES = {
    'Mortality': 0.8079,
    'AKI': 0.8558,
    'Sepsis': 0.7159,
}

# Method name -> {task: delta_auroc}
METHODS = {
    'Statistics-only': {'Mortality': 0.0, 'AKI': 0.0, 'Sepsis': 0.0},   # placeholder
    'Fine-tuned LSTM': {'Mortality': 0.0, 'AKI': 0.0, 'Sepsis': 0.0},   # placeholder
    'DANN':            {'Mortality': 0.0359, 'AKI': 0.0316, 'Sepsis': 0.0164},
    'Deep CORAL':      {'Mortality': 0.0374, 'AKI': 0.0308, 'Sepsis': 0.0167},
    'CoDATS':          {'Mortality': 0.0352, 'AKI': 0.0126, 'Sepsis': -0.0037},
    'CDAN':            {'Mortality': 0.0, 'AKI': 0.0, 'Sepsis': 0.0},    # placeholder
    'Ours (Retrieval)':{'Mortality': 0.0476, 'AKI': 0.0556, 'Sepsis': 0.0512},
}


def load_result(filename):
    """Load a result JSON and return AUROC delta."""
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data.get('difference', {}).get('AUCROC', 0.0)
    return None


def main():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    x = np.arange(len(TASKS))
    n_methods = len(METHODS)
    width = 0.8 / n_methods

    for i, (method, deltas) in enumerate(METHODS.items()):
        values = [deltas[t] for t in TASKS]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, zorder=3)

    ax.set_ylabel(r'$\Delta$ AUROC vs frozen baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(loc='upper left', ncol=2, fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
