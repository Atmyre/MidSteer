"""
Pareto frontier plots for abstract concept steering (toxicity → helpfulness).
Toxicity measured via RealToxicityPrompts + detoxify (BERT classifier).
Helpfulness measured via LLM-as-judge on template prompts.
"""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


METHOD_COLORS = {
    'casteer': '#e41a1c',
    'leace': '#377eb8',
    'midsteer': '#4daf4a',
    'orig': '#999999',
}
METHOD_LABELS = {
    'casteer': 'CASteer',
    'leace': 'LEACE',
    'midsteer': 'MidSteer',
    'orig': 'Baseline',
}
METHOD_MARKERS = {
    'casteer': 'o',
    'leace': 's',
    'midsteer': 'D',
    'orig': '*',
}


def split_last(s, sep):
    idx = s.rfind(sep)
    if idx == -1:
        return s, ''
    return s[:idx], s[idx+1:]


def calculate_pareto_frontier(points, x_better="lower", y_better="lower"):
    sorted_points = sorted(points, key=lambda p: p[0])
    pareto_frontier = []
    if y_better == "lower":
        best_y = float('inf')
        for x, y in sorted_points if x_better == "lower" else reversed(sorted_points):
            if y <= best_y:
                pareto_frontier.append((x, y))
                best_y = y
    else:
        best_y = float('-inf')
        for x, y in sorted_points if x_better == "lower" else reversed(sorted_points):
            if y >= best_y:
                pareto_frontier.append((x, y))
                best_y = y
    return pareto_frontier


def plot_pareto(ax, data_by_method, xlabel, ylabel, title,
                x_better='lower', y_better='higher', show_strengths=True):
    all_points = []
    for method, points in data_by_method.items():
        if method == 'orig':
            ax.scatter(points['x'], points['y'],
                      c=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                      s=200, zorder=10, label=METHOD_LABELS[method], edgecolors='black')
        else:
            ax.scatter(points['x'], points['y'],
                      c=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                      s=80, zorder=5, label=METHOD_LABELS[method], alpha=0.8)
            if show_strengths and 'strengths' in points:
                for x, y, s in zip(points['x'], points['y'], points['strengths']):
                    ax.annotate(f'{s:.0f}', (x, y), textcoords="offset points",
                               xytext=(5, 5), fontsize=7, alpha=0.7)

        for x, y in zip(points['x'], points['y']):
            all_points.append((x, y))

    if all_points:
        pareto = calculate_pareto_frontier(all_points, x_better=x_better, y_better=y_better)
        if pareto:
            px, py = zip(*pareto)
            ax.plot(px, py, 'k--', linewidth=1.5, alpha=0.5, label='Pareto Frontier', zorder=-1)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def load_data(results_dir):
    results_dir = Path(results_dir)
    consistency_scores_list = []
    concept_scores_list = []

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # LLM judge concept scores
        eval_concept = subdir / 'eval' / 'concept_scores.tsv'
        if eval_concept.exists():
            df = pd.read_csv(eval_concept, sep='\t')
            df['experiment'] = subdir.name
            concept_scores_list.append(df)

        # ArmoRM helpfulness scores
        armo_scores = subdir / 'eval' / 'concept_scores_armo.tsv'
        if armo_scores.exists():
            df = pd.read_csv(armo_scores, sep='\t')
            df['experiment'] = subdir.name
            concept_scores_list.append(df)

        # Consistency scores (eval, alpaca, mmlu)
        for ds_name, ds_path in [('eval', subdir / 'eval' / 'consistency_scores.tsv'),
                                  ('alpaca', subdir / 'alpaca' / 'consistency_scores.tsv'),
                                  ('mmlu', subdir / 'mmlu' / 'consistency_scores.tsv')]:
            if ds_path.exists():
                df = pd.read_csv(ds_path, sep='\t')
                df['experiment'] = subdir.name
                df['dataset'] = ds_name
                consistency_scores_list.append(df)

    consistency_scores = pd.concat(consistency_scores_list, axis=0) if consistency_scores_list else pd.DataFrame()
    concept_scores = pd.concat(concept_scores_list, axis=0) if concept_scores_list else pd.DataFrame()

    for df in [consistency_scores, concept_scores]:
        if not df.empty:
            df['method_name'] = df['file'].apply(lambda x: split_last(x[:-5], '_')[0])
            df['strength'] = df['file'].apply(lambda x: float(split_last(x[:-5], '_')[1]))
            df.loc[df['method_name'] == 'None', 'strength'] = None
            df.loc[df['method_name'] == 'None', 'method_name'] = 'orig'

    return consistency_scores, concept_scores


def make_plots(results_dir, output_dir, exclude=None):
    os.makedirs(output_dir, exist_ok=True)
    consistency_scores, concept_scores = load_data(results_dir)

    if exclude:
        for exc in exclude:
            method, strength = exc.split('_', 1)
            strength = float(strength)
            mask_cs = ~((consistency_scores['method_name'] == method) & (consistency_scores['strength'] == strength))
            mask_cp = ~((concept_scores['method_name'] == method) & (concept_scores['strength'] == strength))
            consistency_scores = consistency_scores[mask_cs]
            concept_scores = concept_scores[mask_cp]

    # Load RTP scores
    rtp_path = Path(results_dir) / 'rtp' / 'rtp_scores.tsv'
    rtp = None
    if rtp_path.exists():
        rtp = pd.read_csv(rtp_path, sep='\t')
        rtp['method'] = rtp['label'].apply(lambda x: x.rsplit('_', 1)[0] if '_' in x else x)
        rtp['strength'] = rtp['label'].apply(lambda x: float(x.rsplit('_', 1)[1]) if '_' in x else None)
        if exclude:
            for exc in exclude:
                method, strength = exc.split('_', 1)
                strength = float(strength)
                rtp = rtp[~((rtp['method'] == method) & (rtp['strength'] == strength))]

    alpaca_cons = consistency_scores[consistency_scores['dataset'] == 'alpaca'] if not consistency_scores.empty else pd.DataFrame()

    # ================================================================
    # PLOT 1: RTP Toxicity vs Steering Strength
    # ================================================================
    if rtp is not None:
        baseline_tox = rtp[rtp['method'] == 'baseline']['avg_toxicity'].values[0]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.axhline(baseline_tox, color='gray', linestyle=':', alpha=0.6, label='Baseline', linewidth=2)

        for method in ['casteer', 'leace', 'midsteer']:
            m = rtp[rtp['method'] == method].sort_values('strength')
            if not m.empty:
                ax.plot(m['strength'], m['avg_toxicity'],
                       color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                       label=METHOD_LABELS[method], markersize=8, linewidth=2)
                ax.fill_between(m['strength'],
                               m['avg_toxicity'] - m['std_toxicity'] / np.sqrt(m['num_samples']) * 1.96,
                               m['avg_toxicity'] + m['std_toxicity'] / np.sqrt(m['num_samples']) * 1.96,
                               color=METHOD_COLORS[method], alpha=0.15)

        ax.set_xlabel('Steering Strength', fontsize=12)
        ax.set_ylabel('Avg Toxicity (Detoxify, lower = better)', fontsize=12)
        ax.set_title('RealToxicityPrompts: Toxicity vs Steering Strength', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{output_dir}/rtp_toxicity_vs_strength.pdf', dpi=150, bbox_inches='tight')
        fig.savefig(f'{output_dir}/rtp_toxicity_vs_strength.png', dpi=150, bbox_inches='tight')
        print("Saved rtp_toxicity_vs_strength")
        plt.close()

    # ================================================================
    # PLOT 2: RTP Pareto — Toxicity vs Alpaca Consistency
    # ================================================================
    if rtp is not None and not alpaca_cons.empty:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        data = {}
        for method in ['casteer', 'leace', 'midsteer']:
            rtp_m = rtp[rtp['method'] == method].sort_values('strength')
            cons_m = alpaca_cons[alpaca_cons['method_name'] == method].sort_values('strength')
            if rtp_m.empty or cons_m.empty:
                continue
            merged = pd.merge(rtp_m[['strength', 'avg_toxicity']],
                              cons_m[['strength', 'bert_f1']],
                              on='strength')
            data[method] = {
                'x': (1 - merged['bert_f1'].values),
                'y': merged['avg_toxicity'].values,
                'strengths': merged['strength'].values,
            }

        orig_cons = alpaca_cons[alpaca_cons['method_name'] == 'orig']
        if not orig_cons.empty:
            data['orig'] = {
                'x': np.array([1 - orig_cons['bert_f1'].values[0]]),
                'y': np.array([baseline_tox]),
            }

        plot_pareto(ax, data,
                   xlabel='Inconsistency (1 - Alpaca BERT-F1, lower = better)',
                   ylabel='RTP Toxicity (Detoxify, lower = better)',
                   title='RealToxicityPrompts: Toxicity vs Consistency Pareto',
                   x_better='lower', y_better='lower')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/rtp_pareto_toxicity_vs_consistency.pdf', dpi=150, bbox_inches='tight')
        fig.savefig(f'{output_dir}/rtp_pareto_toxicity_vs_consistency.png', dpi=150, bbox_inches='tight')
        print("Saved rtp_pareto_toxicity_vs_consistency")
        plt.close()

    # ================================================================
    # PLOT 2b: RTP Toxicity vs ArmoRM Helpfulness (both classifiers, no LLM judge)
    # ================================================================
    armo_scores = concept_scores[concept_scores['concept'] == 'helpfulness_armo'] if not concept_scores.empty else pd.DataFrame()
    if rtp is not None and not armo_scores.empty:
        # Use helpfulness prompts for ArmoRM
        armo_help = armo_scores[armo_scores['experiment'].str.contains('__helpfulness')]
        if not armo_help.empty:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            data = {}
            for method in ['casteer', 'leace', 'midsteer']:
                rtp_m = rtp[rtp['method'] == method].sort_values('strength')
                arm_m = armo_help[armo_help['method_name'] == method].sort_values('strength')
                if rtp_m.empty or arm_m.empty:
                    continue
                merged = pd.merge(rtp_m[['strength', 'avg_toxicity']],
                                  arm_m[['strength', 'avg_score']],
                                  on='strength')
                data[method] = {
                    'x': merged['avg_toxicity'].values,
                    'y': merged['avg_score'].values,
                    'strengths': merged['strength'].values,
                }

            orig_armo = armo_help[armo_help['method_name'] == 'orig']
            if not orig_armo.empty:
                data['orig'] = {
                    'x': np.array([baseline_tox]),
                    'y': orig_armo['avg_score'].values[:1],
                }

            plot_pareto(ax, data,
                       xlabel='RTP Toxicity (Detoxify, lower = better)',
                       ylabel='ArmoRM Helpfulness (higher = better)',
                       title='Toxicity (Detoxify) vs Helpfulness (ArmoRM) — Classifier-Only',
                       x_better='lower', y_better='higher')
            fig.tight_layout()
            fig.savefig(f'{output_dir}/pareto_rtp_vs_armo.pdf', dpi=150, bbox_inches='tight')
            fig.savefig(f'{output_dir}/pareto_rtp_vs_armo.png', dpi=150, bbox_inches='tight')
            print("Saved pareto_rtp_vs_armo")
            plt.close()

    # ================================================================
    # PLOT 3: Helpfulness on helpfulness prompts vs Alpaca consistency
    # ================================================================
    help_scores = concept_scores[concept_scores['experiment'].str.contains('__helpfulness')]
    if not help_scores.empty and not alpaca_cons.empty:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        data = {}
        for method in ['orig', 'casteer', 'leace', 'midsteer']:
            hs = help_scores[(help_scores['method_name'] == method) & (help_scores['concept'] == 'helpfulness')]
            cons = alpaca_cons[alpaca_cons['method_name'] == method]
            if hs.empty or cons.empty:
                continue
            hs = hs.sort_values('strength')
            cons = cons.sort_values('strength')
            merged = pd.merge(hs[['strength', 'avg_score']],
                              cons[['strength', 'bert_f1']],
                              on='strength')
            data[method] = {
                'x': (1 - merged['bert_f1'].values),
                'y': merged['avg_score'].values,
                'strengths': merged['strength'].values if method != 'orig' else None,
            }

        plot_pareto(ax, data,
                   xlabel='Inconsistency (1 - Alpaca BERT-F1, lower = better)',
                   ylabel='Helpfulness Score (LLM judge, higher = better)',
                   title='Helpfulness vs Consistency Pareto',
                   x_better='lower', y_better='higher')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/pareto_helpfulness_vs_consistency.pdf', dpi=150, bbox_inches='tight')
        fig.savefig(f'{output_dir}/pareto_helpfulness_vs_consistency.png', dpi=150, bbox_inches='tight')
        print("Saved pareto_helpfulness_vs_consistency")
        plt.close()

    # ================================================================
    # PLOT 4: RTP Toxicity vs Helpfulness (the key trade-off plot)
    # ================================================================
    if rtp is not None and not help_scores.empty:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        data = {}
        for method in ['casteer', 'leace', 'midsteer']:
            rtp_m = rtp[rtp['method'] == method].sort_values('strength')
            hs = help_scores[(help_scores['method_name'] == method) & (help_scores['concept'] == 'helpfulness')]
            hs = hs.sort_values('strength')
            if rtp_m.empty or hs.empty:
                continue
            merged = pd.merge(rtp_m[['strength', 'avg_toxicity']],
                              hs[['strength', 'avg_score']],
                              on='strength')
            data[method] = {
                'x': merged['avg_toxicity'].values,
                'y': merged['avg_score'].values,
                'strengths': merged['strength'].values,
            }

        orig_hs = help_scores[(help_scores['method_name'] == 'orig') & (help_scores['concept'] == 'helpfulness')]
        if not orig_hs.empty:
            data['orig'] = {
                'x': np.array([baseline_tox]),
                'y': orig_hs['avg_score'].values[:1],
            }

        plot_pareto(ax, data,
                   xlabel='RTP Toxicity (Detoxify, lower = better)',
                   ylabel='Helpfulness Score (LLM judge, higher = better)',
                   title='Toxicity–Helpfulness Trade-off',
                   x_better='lower', y_better='higher')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/rtp_toxicity_vs_helpfulness.pdf', dpi=150, bbox_inches='tight')
        fig.savefig(f'{output_dir}/rtp_toxicity_vs_helpfulness.png', dpi=150, bbox_inches='tight')
        print("Saved rtp_toxicity_vs_helpfulness")
        plt.close()

    # ================================================================
    # PLOT 5: Unrelated concept preservation (own concept score)
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    unrelated = ['sarcasm', 'politeness', 'creativity', 'mathematics']
    for idx, concept_name in enumerate(unrelated):
        ax = axes[idx // 2][idx % 2]
        concept_exp = concept_scores[concept_scores['experiment'].str.contains(f'__{concept_name}')]
        if concept_exp.empty:
            continue

        own_scores = concept_exp[concept_exp['concept'] == concept_name]

        for method in ['casteer', 'leace', 'midsteer']:
            ms = own_scores[own_scores['method_name'] == method].sort_values('strength')
            if not ms.empty:
                ax.plot(ms['strength'], ms['avg_score'],
                       color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                       label=METHOD_LABELS[method], markersize=6, linewidth=2)

        orig = own_scores[own_scores['method_name'] == 'orig']
        if not orig.empty:
            ax.axhline(orig['avg_score'].values[0], color='gray', linestyle=':', alpha=0.5, label='Baseline', linewidth=2)

        ax.set_xlabel('Steering Strength', fontsize=10)
        ax.set_ylabel(f'{concept_name.capitalize()} Score', fontsize=10)
        ax.set_title(f'Unrelated: "{concept_name}"', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Preservation of Unrelated Concepts (toxicity → helpfulness steering)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/unrelated_concepts_preservation.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(f'{output_dir}/unrelated_concepts_preservation.png', dpi=150, bbox_inches='tight')
    print("Saved unrelated_concepts_preservation")
    plt.close()

    # ================================================================
    # PLOT 6: Alpaca & MMLU consistency vs strength
    # ================================================================
    if not consistency_scores.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for idx, dataset in enumerate(['alpaca', 'mmlu']):
            ax = axes[idx]
            ds = consistency_scores[consistency_scores['dataset'] == dataset]
            if ds.empty:
                continue
            for method in ['casteer', 'leace', 'midsteer']:
                m = ds[ds['method_name'] == method].sort_values('strength')
                if not m.empty:
                    ax.plot(m['strength'], m['bert_f1'],
                           color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                           label=METHOD_LABELS[method], markersize=6, linewidth=2)

            orig = ds[ds['method_name'] == 'orig']
            if not orig.empty:
                ax.axhline(orig['bert_f1'].values[0], color='gray', linestyle=':', alpha=0.5, label='Baseline', linewidth=2)

            ax.set_xlabel('Steering Strength', fontsize=11)
            ax.set_ylabel('BERT-F1 Score', fontsize=11)
            ax.set_title(f'{dataset.upper()} Consistency', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Consistency Preservation (toxicity → helpfulness)', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/consistency_vs_strength.pdf', dpi=150, bbox_inches='tight')
        fig.savefig(f'{output_dir}/consistency_vs_strength.png', dpi=150, bbox_inches='tight')
        print("Saved consistency_vs_strength")
        plt.close()

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='exp/plots/abstract_concepts')
    parser.add_argument('--exclude', type=str, nargs='*', default=None)
    args = parser.parse_args()
    make_plots(args.results_dir, args.output_dir, exclude=args.exclude)
