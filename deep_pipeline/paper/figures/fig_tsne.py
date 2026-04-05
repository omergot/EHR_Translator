"""
Generate t-SNE/UMAP visualization: source vs translated vs target distributions.
Figure for experiments section or appendix.

Requires: hidden state dumps from analyze_hidden_states.py
Output: paper/figures/fig_tsne.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# TODO: Implement once hidden state dumps are available
# Steps:
# 1. Run src/core/eval.py to get hidden states for source, translated, target
# 2. Apply t-SNE or UMAP (from sklearn.manifold or umap-learn)
# 3. Plot 3 panels: source vs target, translated vs target, overlay

STYLE_PATH = os.path.join(os.path.dirname(__file__), 'neurips.mplstyle')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'fig_tsne.pdf')

plt.style.use(STYLE_PATH)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0))

    for ax, title in zip(axes, ['Source vs Target', 'Translated vs Target', 'All Three']):
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, 'TODO', ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
