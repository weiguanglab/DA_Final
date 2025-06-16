import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
import os

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class PerformanceVisualizer:

    @staticmethod
    def load_detailed_results(file_path: str) -> List[Dict]:
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                results.append(json.loads(line.strip()))
        return results

    @staticmethod
    def plot_confidence_lines(detailed_results: List[Dict], save_path: str):
        print("Starting confidence lines plotting...")

        fig, ax = plt.subplots(figsize=(15, 10))

        total_samples = len(detailed_results)
        correct_samples = 0

        for i, result in enumerate(detailed_results):
            scores = result['top_200_scores']
            if not scores:
                continue

            x_values = range(1, len(scores) + 1)
            ax.plot(x_values, scores, color='lightblue', alpha=0.3, linewidth=0.8)

            if result['correct_item_rank'] > 0:
                correct_samples += 1
                rank = result['correct_item_rank']
                if rank <= len(scores):
                    confidence_score = scores[rank - 1]
                    ax.scatter(rank, confidence_score, color='red', s=30, alpha=0.8, zorder=5)

        ax.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top-200 Confidence Distribution\nTotal Samples: {total_samples}, Correct Retrieval: {correct_samples}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 200)

        blue_line = mpatches.Patch(color='lightblue', label='All Sample Confidence Lines')
        red_dot = mpatches.Patch(color='red', label='Correct Sample Position')
        ax.legend(handles=[blue_line, red_dot], loc='upper right')

        recall_rate = correct_samples / total_samples if total_samples > 0 else 0
        textstr = f'Retrieval Success Rate: {recall_rate:.2%}\nCorrect Samples: {correct_samples}/{total_samples}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confidence lines plot saved to: {save_path}")
        print(f"Statistics - Total samples: {total_samples}, Correct retrieval: {correct_samples}, Success rate: {recall_rate:.2%}")

    @staticmethod
    def plot_rank_distribution(detailed_results: List[Dict], save_path: str):
        print("Starting rank distribution plotting...")

        correct_ranks = []
        for result in detailed_results:
            if result['correct_item_rank'] > 0:
                correct_ranks.append(result['correct_item_rank'])

        if not correct_ranks:
            print("Warning: No correct retrieval samples found, cannot plot rank distribution")
            return

        correct_ranks = np.array(correct_ranks)

        fig, ax1 = plt.subplots(figsize=(15, 10))
        ax2 = ax1.twinx()

        bins = np.arange(1, 202, 10)
        n, bins_edges, patches = ax1.hist(correct_ranks, bins=bins, alpha=0.7,
                                         color='skyblue', edgecolor='black', linewidth=0.8)

        ax1.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        from scipy import stats

        density = stats.gaussian_kde(correct_ranks)
        xs = np.linspace(1, 200, 200)
        density_values = density(xs)

        line1 = ax2.plot(xs, density_values, color='red', linewidth=2,
                        label='Probability Density')
        ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        sorted_ranks = np.sort(correct_ranks)
        cumulative_prob = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)

        density_max = np.max(density_values)
        scaled_cumulative = cumulative_prob * density_max

        line2 = ax2.plot(sorted_ranks, scaled_cumulative, color='green', linewidth=2,
                        linestyle='--', label='Cumulative Probability (Scaled)')

        ax1.set_title(f'Correct Sample Rank Distribution Analysis\nTotal Correct Samples: {len(correct_ranks)} / {len(detailed_results)}',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 200)

        mean_rank = np.mean(correct_ranks)
        median_rank = np.median(correct_ranks)

        top10_ratio = np.sum(correct_ranks <= 10) / len(correct_ranks)
        top50_ratio = np.sum(correct_ranks <= 50) / len(correct_ranks)
        top100_ratio = np.sum(correct_ranks <= 100) / len(correct_ranks)

        hist_patch = mpatches.Patch(color='skyblue', label='Frequency Distribution')
        all_lines = [hist_patch] + line1 + line2
        all_labels = ['Frequency Distribution', 'Probability Density', 'Cumulative Probability (Scaled)']

        ax1.legend(all_lines, all_labels, loc='upper right')

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Rank distribution plot saved to: {save_path}")
        print(f"Statistics:")
        print(f"  - Correct retrieval samples: {len(correct_ranks)} / {len(detailed_results)}")
        print(f"  - Mean rank: {mean_rank:.1f}")
        print(f"  - Median rank: {median_rank:.1f}")
        print(f"  - Top-10 ratio: {top10_ratio:.2%}")
        print(f"  - Top-50 ratio: {top50_ratio:.2%}")
        print(f"  - Top-100 ratio: {top100_ratio:.2%}")

    @staticmethod
    def generate_all_visualizations(detailed_results_path: str, output_dir: str):
        print("Starting to generate all visualization charts...")

        detailed_results = PerformanceVisualizer.load_detailed_results(detailed_results_path)
        print(f"Loaded {len(detailed_results)} detailed results")

        os.makedirs(output_dir, exist_ok=True)

        confidence_plot_path = os.path.join(output_dir, "confidence_lines.png")
        PerformanceVisualizer.plot_confidence_lines(detailed_results, confidence_plot_path)

        rank_dist_path = os.path.join(output_dir, "rank_distribution.png")
        PerformanceVisualizer.plot_rank_distribution(detailed_results, rank_dist_path)

        print(f"All visualization charts saved to directory: {output_dir}")