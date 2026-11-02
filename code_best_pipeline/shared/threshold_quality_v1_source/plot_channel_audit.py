"""Plot paired validation reconstruction probes (not final student detection)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source, output):
    data = json.loads(Path(source).read_text())['summary']
    channels = ['rgb', 'student_vae_sample0', 'unadapted_x0_t100',
                'unadapted_x0_t300', 'unadapted_x0_t500']
    labels = ['Original RGB', 'Student VAE', 'x0 at t=100', 'x0 at t=300', 'x0 at t=500']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for name, label, color in [('scale100', 'Scale 1.0', '#185abc'),
                               ('scale085', 'Scale 0.85', '#d45b17'),
                               ('conv_restore', 'Conv restore', '#218351')]:
        rows = [data[name][c] for c in channels]
        values = np.array([r['paired_signal'] for r in rows])
        ci = np.array([r['paired_signal_ci95'] for r in rows])
        axes[0].errorbar(range(5), values, yerr=np.maximum(0, np.vstack((values-ci[:, 0], ci[:, 1]-values))),
                         color=color, marker='o', capsize=3, label=label)
        axes[1].plot(range(5), [100*r['paired_signal_retention_vs_rgb'] for r in rows],
                     color=color, marker='o', label=label)
    axes[0].set_ylabel('Mean score(marked) - score(clean)')
    axes[0].set_title('Paired signal; 95% image-bootstrap intervals')
    axes[1].set_ylabel('Signal remaining relative to original RGB (%)')
    axes[1].set_title('Most attenuation occurs after denoising in this probe')
    for ax in axes:
        ax.set_xticks(range(5), labels, rotation=18)
        ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.suptitle('64 validation pairs / unadapted SD1.4 / frozen semantic verifier', fontsize=13)
    fig.supxlabel('Independent reconstruction probes, not a sequential trajectory or a student-training experiment', fontsize=9)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    plot(args.source, args.output)
