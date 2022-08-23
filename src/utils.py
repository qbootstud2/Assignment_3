import numpy as np
import matplotlib.pyplot as plt


def shuffle_2d(a: np.array, b: np.array) -> (np.array, np.array):
    n_elem = a.shape[0]
    indeces = np.random.choice(n_elem, size=n_elem, replace=False)
    return a[indeces], b[indeces]


def plot_prc(precision: float, recall: float, no_skill: float, area: float) -> None:

    plt.figure(figsize=(5, 4))

    # Plot data
    plt.plot(recall, precision, color='C0', linewidth=2, alpha=1, label='Area = %.2f' % area)
    plt.plot([0, 1], [no_skill, no_skill], color='C0', linewidth=2, alpha=1, linestyle='--', label='Random')

    # Options
    plt.title(r'Precision-recall curve', fontsize=15)
    plt.xlabel(r'Recall', fontsize=15, labelpad=8)
    plt.ylabel(r'Precision', fontsize=15, labelpad=8)
    plt.legend(loc='upper right', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tick_params(which='both', direction='out', bottom=True, left=True, right=True, labelsize=14)
    plt.show()
