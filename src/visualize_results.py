#!/usr/bin/env python3
"""
Create visualizations for fairness evaluation results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_results(csv_path):
    """Load fairness report"""
    df = pd.read_csv(csv_path)
    return df

def plot_fairness_gaps(df, output_path):
    """Plot fairness gaps across models"""
    
    # Calculate gaps
    gaps_data = []
    
    for idx, row in df.iterrows():
        model = row['model']
        
        # Race gap
        race_cols = [c for c in df.columns if (c.startswith('acc_') or c.startswith('jaccard_')) 
                     and c not in ['acc_Female', 'acc_Male', 'jaccard_Female', 'jaccard_Male']]
        race_vals = [row[c] for c in race_cols if pd.notna(row[c])]
        if race_vals:
            race_gap = max(race_vals) - min(race_vals)
            gaps_data.append({'model': model, 'dimension': 'Race', 'gap': race_gap})
        
        # Gender gap
        gender_cols = [c for c in df.columns if c in ['acc_Female', 'acc_Male', 'jaccard_Female', 'jaccard_Male']]
        gender_vals = [row[c] for c in gender_cols if pd.notna(row[c])]
        if gender_vals:
            gender_gap = max(gender_vals) - min(gender_vals)
            gaps_data.append({'model': model, 'dimension': 'Gender', 'gap': gender_gap})
    
    gaps_df = pd.DataFrame(gaps_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(gaps_df['model'].unique()))
    width = 0.35
    
    race_gaps = gaps_df[gaps_df['dimension'] == 'Race']['gap'].values
    gender_gaps = gaps_df[gaps_df['dimension'] == 'Gender']['gap'].values
    
    ax.bar(x - width/2, race_gaps, width, label='Race Gap', color='#e74c3c')
    ax.bar(x + width/2, gender_gaps, width, label='Gender Gap', color='#3498db')
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fairness Gap (Max - Min Accuracy)', fontsize=14, fontweight='bold')
    ax.set_title('Fairness Gaps Across Models\n(Lower is Better)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gaps_df['model'].unique())
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (r, g) in enumerate(zip(race_gaps, gender_gaps)):
        ax.text(i - width/2, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, g + 0.01, f'{g:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_accuracy_by_race(df, output_path):
    """Plot accuracy breakdown by race for each model"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract race accuracies
    race_cols = [c for c in df.columns if c.startswith('acc_') 
                 and c not in ['acc_Female', 'acc_Male']]
    
    # v1 and v2 models (single-label)
    single_label_models = df[df['overall_accuracy'].notna()]
    
    for idx, (model_idx, row) in enumerate(single_label_models.iterrows()):
        ax = axes[idx] if idx < 2 else None
        if ax is None:
            continue
            
        model = row['model']
        races = []
        accuracies = []
        
        for col in race_cols:
            if pd.notna(row[col]):
                race = col.replace('acc_', '')
                races.append(race)
                accuracies.append(row[col])
        
        # Create bar plot
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(races)]
        bars = ax.bar(races, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1%}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_title(f'{model.upper()}: Accuracy by Race', fontsize=14, fontweight='bold')
        ax.set_xlabel('Race Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add overall accuracy line
        overall = row['overall_accuracy']
        ax.axhline(y=overall, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall:.1%}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_model_comparison(df, output_path):
    """Compare overall performance across models"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].values
    
    # Get overall metrics
    overall_scores = []
    labels = []
    colors = []
    
    for idx, row in df.iterrows():
        model = row['model']
        if pd.notna(row.get('overall_accuracy')):
            overall_scores.append(row['overall_accuracy'])
            labels.append(f"{model}\n(Single-Label)")
            colors.append('#95a5a6')
        elif pd.notna(row.get('overall_jaccard')):
            overall_scores.append(row['overall_jaccard'])
            labels.append(f"{model}\n(Multi-Label)")
            colors.append('#27ae60')
    
    bars = ax.bar(range(len(labels)), overall_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, overall_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.1%}' if score > 0.01 else f'{score:.3f}',
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('Overall Model Performance Comparison\n(Accuracy for v1/v2, Jaccard for v3)', 
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(overall_scores) * 1.2 if overall_scores else 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_table(df, output_path):
    """Create a summary table of key metrics"""
    
    summary_data = []
    
    for idx, row in df.iterrows():
        model = row['model']
        
        summary = {
            'Model': model,
            'Type': 'Single-Label' if pd.notna(row.get('overall_accuracy')) else 'Multi-Label',
        }
        
        # Overall performance
        if pd.notna(row.get('overall_accuracy')):
            summary['Overall Score'] = f"{row['overall_accuracy']:.1%}"
        elif pd.notna(row.get('overall_jaccard')):
            summary['Overall Score'] = f"{row['overall_jaccard']:.3f}"
        
        # Calculate gaps
        race_cols = [c for c in df.columns if (c.startswith('acc_') or c.startswith('jaccard_')) 
                     and c not in ['acc_Female', 'acc_Male', 'jaccard_Female', 'jaccard_Male']]
        race_vals = [row[c] for c in race_cols if pd.notna(row[c])]
        if race_vals:
            race_gap = max(race_vals) - min(race_vals)
            summary['Race Gap'] = f"{race_gap:.3f}"
        
        gender_cols = [c for c in df.columns if c in ['acc_Female', 'acc_Male', 'jaccard_Female', 'jaccard_Male']]
        gender_vals = [row[c] for c in gender_cols if pd.notna(row[c])]
        if gender_vals:
            gender_gap = max(gender_vals) - min(gender_vals)
            summary['Gender Gap'] = f"{gender_gap:.3f}"
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    # Print to console
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60 + "\n")


def main():
    # Paths
    results_path = Path('results/fairness_report.csv')
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Load results
    df = load_results(results_path)
    
    # Create plots
    plot_fairness_gaps(df, output_dir / 'fairness_gaps.png')
    plot_accuracy_by_race(df, output_dir / 'accuracy_by_race.png')
    plot_model_comparison(df, output_dir / 'model_comparison.png')
    
    # Create summary table
    create_summary_table(df, Path('results/summary_table.csv'))
    
    print("\n" + "="*60)
    print("VISUALIZATIONS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - fairness_gaps.png")
    print("  - accuracy_by_race.png")
    print("  - model_comparison.png")
    print("  - ../summary_table.csv")
    print("="*60)


if __name__ == '__main__':
    main()
