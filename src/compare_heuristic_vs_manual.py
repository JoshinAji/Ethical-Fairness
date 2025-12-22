#!/usr/bin/env python3
"""
Compare heuristic labels vs. manual human annotations.

Calculates agreement metrics:
- Jaccard similarity (for multi-label race_ml)
- Hamming loss (overall disagreement)
- MAE for skin tone
- Exact match accuracy
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if isinstance(set1, str):
        set1 = set(set1.split('|')) if set1 else set()
    if isinstance(set2, str):
        set2 = set(set2.split('|')) if set2 else set()
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0  # Both empty = perfect agreement
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def hamming_loss(row_heuristic, row_manual, binary_fields):
    """Calculate Hamming loss (proportion of mismatched binary/categorical fields)."""
    mismatches = 0
    total = 0
    
    for field in binary_fields:
        if field in row_heuristic and field in row_manual:
            if pd.notna(row_heuristic[field]) and pd.notna(row_manual[field]):
                total += 1
                if row_heuristic[field] != row_manual[field]:
                    mismatches += 1
    
    return mismatches / total if total > 0 else 0.0


def mae_skin_tone(heuristic_tone, manual_tone):
    """Calculate MAE for skin tone (ordinal scale 1-7)."""
    try:
        h = float(heuristic_tone) if pd.notna(heuristic_tone) and heuristic_tone != "" else None
        m = float(manual_tone) if pd.notna(manual_tone) and manual_tone != "" else None
        
        if h is None or m is None:
            return None
        
        return abs(h - m)
    except:
        return None


def compare_labels(heuristic_csv, manual_csv, output_report):
    """
    Compare heuristic vs manual labels and generate agreement report.
    
    Args:
        heuristic_csv: CSV with heuristic labels
        manual_csv: CSV with manual annotations
        output_report: Path to save comparison report
    """
    print("Loading datasets...")
    df_heuristic = pd.read_csv(heuristic_csv)
    df_manual = pd.read_csv(manual_csv)
    
    # Merge on image_id
    print("Merging datasets...")
    df_merged = df_heuristic.merge(
        df_manual,
        on='image_id',
        suffixes=('_heuristic', '_manual')
    )
    
    n_samples = len(df_merged)
    print(f"Comparing {n_samples} samples...\n")
    
    if n_samples == 0:
        print("ERROR: No matching samples found between heuristic and manual datasets!")
        return
    
    # === 1. Race Multi-Label (Jaccard Similarity) ===
    print("=" * 60)
    print("1. RACE MULTI-LABEL (race_ml)")
    print("=" * 60)
    
    jaccard_scores = []
    for idx, row in df_merged.iterrows():
        h_race = row.get('race_ml_heuristic', '')
        m_race = row.get('race_ml_manual', '')
        jaccard = jaccard_similarity(h_race, m_race)
        jaccard_scores.append(jaccard)
    
    mean_jaccard = np.mean(jaccard_scores)
    print(f"Mean Jaccard Similarity: {mean_jaccard:.3f}")
    print(f"Median Jaccard Similarity: {np.median(jaccard_scores):.3f}")
    print(f"Samples with perfect match (Jaccard=1.0): {sum(1 for j in jaccard_scores if j == 1.0)}/{n_samples}")
    
    # === 2. Skin Tone (MAE) ===
    print("\n" + "=" * 60)
    print("2. SKIN TONE (skin_tone_bin)")
    print("=" * 60)
    
    mae_values = []
    for idx, row in df_merged.iterrows():
        h_tone = row.get('skin_tone_bin_heuristic', '')
        m_tone = row.get('skin_tone_bin_manual', '')
        mae = mae_skin_tone(h_tone, m_tone)
        if mae is not None:
            mae_values.append(mae)
    
    if mae_values:
        mean_mae = np.mean(mae_values)
        print(f"Mean Absolute Error (MAE): {mean_mae:.3f} (on 1-7 scale)")
        print(f"Samples with exact match: {sum(1 for m in mae_values if m == 0)}/{len(mae_values)}")
        print(f"Samples within ±1: {sum(1 for m in mae_values if m <= 1)}/{len(mae_values)}")
    else:
        print("No comparable skin tone values found.")
        mean_mae = None
    
    # === 3. Binary Flags (Hamming Loss) ===
    print("\n" + "=" * 60)
    print("3. BINARY FLAGS (ambiguous_mixed, prefer_not_to_label, unknown_uncertain)")
    print("=" * 60)
    
    binary_fields = ['ambiguous_mixed', 'prefer_not_to_label', 'unknown_uncertain']
    hamming_losses = []
    
    for idx, row in df_merged.iterrows():
        h_loss = hamming_loss(
            {f'{f}_heuristic': row.get(f'{f}_heuristic') for f in binary_fields},
            {f'{f}_manual': row.get(f'{f}_manual') for f in binary_fields},
            [f'{f}_heuristic' for f in binary_fields]
        )
        hamming_losses.append(h_loss)
    
    mean_hamming = np.mean(hamming_losses)
    print(f"Mean Hamming Loss: {mean_hamming:.3f} (lower is better, 0.0 = perfect agreement)")
    
    # Individual flag agreement
    for field in binary_fields:
        h_col = f'{field}_heuristic'
        m_col = f'{field}_manual'
        
        if h_col in df_merged.columns and m_col in df_merged.columns:
            matches = (df_merged[h_col] == df_merged[m_col]).sum()
            total = df_merged[[h_col, m_col]].notna().all(axis=1).sum()
            accuracy = matches / total if total > 0 else 0
            print(f"  {field}: {accuracy:.3f} agreement ({matches}/{total})")
    
    # === 4. Confidence Scores (Correlation) ===
    print("\n" + "=" * 60)
    print("4. CONFIDENCE SCORES")
    print("=" * 60)
    
    # Mapping for string confidence values
    conf_mapping = {'low': 1, 'medium': 2, 'high': 3}
    
    conf_fields = ['conf_race', 'conf_gender', 'conf_skin']
    for field in conf_fields:
        h_col = f'{field}_heuristic'
        m_col = f'{field}_manual'
        
        if h_col in df_merged.columns and m_col in df_merged.columns:
            # Filter valid pairs
            valid_pairs = df_merged[[h_col, m_col]].dropna()
            
            if len(valid_pairs) > 0:
                # Convert string values to numeric if needed
                h_vals = valid_pairs[h_col].replace(conf_mapping)
                m_vals = valid_pairs[m_col].replace(conf_mapping)
                
                # Only compute if we have numeric values
                try:
                    h_numeric = pd.to_numeric(h_vals, errors='coerce')
                    m_numeric = pd.to_numeric(m_vals, errors='coerce')
                    
                    # Drop any remaining NaN after conversion
                    valid_mask = h_numeric.notna() & m_numeric.notna()
                    h_numeric = h_numeric[valid_mask]
                    m_numeric = m_numeric[valid_mask]
                    
                    if len(h_numeric) > 0:
                        correlation = h_numeric.corr(m_numeric)
                        mae_conf = (h_numeric - m_numeric).abs().mean()
                        agreement = (h_numeric == m_numeric).sum() / len(h_numeric)
                        print(f"  {field}:")
                        print(f"    Correlation: {correlation:.3f}")
                        print(f"    MAE: {mae_conf:.3f}")
                        print(f"    Exact agreement: {agreement:.3f} ({(h_numeric == m_numeric).sum()}/{len(h_numeric)})")
                    else:
                        print(f"  {field}: No valid numeric pairs to compare")
                except Exception as e:
                    print(f"  {field}: Error comparing - {str(e)}")
            else:
                print(f"  {field}: No valid pairs to compare")
    
    # === 5. Overall Summary ===
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    print(f"✓ Samples compared: {n_samples}")
    print(f"✓ Race multi-label Jaccard: {mean_jaccard:.3f}")
    if mean_mae is not None:
        print(f"✓ Skin tone MAE: {mean_mae:.3f}")
    print(f"✓ Binary flags Hamming loss: {mean_hamming:.3f}")
    
    # Interpretation
    print("\nINTERPRETATION:")
    if mean_jaccard >= 0.70:
        print("  ✓ Race labeling: HIGH agreement (Jaccard ≥ 0.70)")
        race_quality = "HIGH"
    elif mean_jaccard >= 0.55:
        print("  ⚠ Race labeling: MODERATE agreement (0.55 ≤ Jaccard < 0.70)")
        race_quality = "MODERATE"
    else:
        print("  ✗ Race labeling: LOW agreement (Jaccard < 0.55)")
        race_quality = "LOW"
    
    if mean_mae is not None:
        if mean_mae <= 1.0:
            print("  ✓ Skin tone: HIGH agreement (MAE ≤ 1.0)")
            skin_quality = "HIGH"
        elif mean_mae <= 1.5:
            print("  ⚠ Skin tone: MODERATE agreement (1.0 < MAE ≤ 1.5)")
            skin_quality = "MODERATE"
        else:
            print("  ✗ Skin tone: LOW agreement (MAE > 1.5)")
            skin_quality = "LOW"
    else:
        skin_quality = "N/A"
    
    if mean_hamming <= 0.20:
        print("  ✓ Binary flags: HIGH agreement (Hamming ≤ 0.20)")
        flag_quality = "HIGH"
    elif mean_hamming <= 0.35:
        print("  ⚠ Binary flags: MODERATE agreement (0.20 < Hamming ≤ 0.35)")
        flag_quality = "MODERATE"
    else:
        print("  ✗ Binary flags: LOW agreement (Hamming > 0.35)")
        flag_quality = "LOW"
    
    # Recommendation
    print("\nRECOMMENDATION:")
    if race_quality == "HIGH" and (skin_quality in ["HIGH", "N/A"]) and flag_quality in ["HIGH", "MODERATE"]:
        print("  ✓ Heuristic labels are acceptable. Proceed with all 900 samples.")
    elif race_quality == "MODERATE":
        print("  ⚠ Consider refining heuristics or manually annotating more samples (200-300).")
    else:
        print("  ✗ Heuristics need significant refinement or expand manual annotation.")
    
    # Save detailed report
    print(f"\nSaving detailed comparison to {output_report}...")
    df_merged['jaccard_race'] = jaccard_scores
    df_merged['hamming_loss'] = hamming_losses
    
    output_path = Path(output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_report, index=False)
    
    print(f"✓ Report saved: {output_report}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare heuristic vs manual labels"
    )
    parser.add_argument(
        '--heuristic',
        required=True,
        help='CSV with heuristic labels'
    )
    parser.add_argument(
        '--manual',
        required=True,
        help='CSV with manual annotations'
    )
    parser.add_argument(
        '--output',
        default='data/comparison_report.csv',
        help='Output path for comparison report'
    )
    
    args = parser.parse_args()
    
    compare_labels(
        heuristic_csv=args.heuristic,
        manual_csv=args.manual,
        output_report=args.output
    )


if __name__ == '__main__':
    main()
