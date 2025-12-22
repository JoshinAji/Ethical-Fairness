#!/usr/bin/env python3
"""
Evaluate and compare fairness metrics across v1, v2, and v3 models.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_model(model_path):
    """Load trained model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def predict_single_label(model, encoder, X):
    """Make predictions for single-label model"""
    y_pred_encoded = model.predict(X)
    y_pred = encoder.inverse_transform(y_pred_encoded)
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X)
    
    return y_pred, y_pred_proba


def predict_multi_label(model, encoder, X):
    """Make predictions for multi-label model"""
    y_pred_encoded = model.predict(X)
    y_pred = encoder.inverse_transform(y_pred_encoded)
    
    # Get prediction probabilities for each label
    y_pred_proba = []
    for estimator in model.estimators_:
        y_pred_proba.append(estimator.predict_proba(X)[:, 1])
    y_pred_proba = np.array(y_pred_proba).T
    
    return y_pred, y_pred_proba


def compute_fairness_metrics(y_true, y_pred, df_test, model_name):
    """Compute fairness metrics stratified by demographic groups"""
    
    results = {
        'model': model_name,
        'overall_accuracy': accuracy_score(y_true, y_pred),
        'by_race': {},
        'by_gender': {},
        'by_race_gender': {}
    }
    
    # Overall metrics
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
    
    # Accuracy by race
    print(f"\n--- Accuracy by Race ---")
    for race in df_test['race_cat'].unique():
        if pd.isna(race):
            continue
        mask = df_test['race_cat'] == race
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            results['by_race'][race] = {
                'accuracy': acc,
                'count': mask.sum()
            }
            print(f"  {race:15s}: {acc:.3f} (n={mask.sum()})")
    
    # Accuracy by gender
    print(f"\n--- Accuracy by Gender ---")
    for gender in df_test['gender_cat'].unique():
        if pd.isna(gender):
            continue
        mask = df_test['gender_cat'] == gender
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            results['by_gender'][gender] = {
                'accuracy': acc,
                'count': mask.sum()
            }
            print(f"  {gender:15s}: {acc:.3f} (n={mask.sum()})")
    
    # Accuracy by race-gender intersection
    print(f"\n--- Accuracy by Race-Gender Intersection ---")
    for race in df_test['race_cat'].unique():
        if pd.isna(race):
            continue
        for gender in df_test['gender_cat'].unique():
            if pd.isna(gender):
                continue
            mask = (df_test['race_cat'] == race) & (df_test['gender_cat'] == gender)
            if mask.sum() > 5:  # Only show if enough samples
                acc = accuracy_score(y_true[mask], y_pred[mask])
                key = f"{race}_{gender}"
                results['by_race_gender'][key] = {
                    'accuracy': acc,
                    'count': mask.sum()
                }
                print(f"  {race:10s} × {gender:8s}: {acc:.3f} (n={mask.sum()})")
    
    return results


def compute_multi_label_fairness(y_true, y_pred, df_test, model_name):
    """Compute fairness metrics for multi-label classification"""
    
    results = {
        'model': model_name,
        'by_race': {},
        'by_gender': {},
        'by_race_gender': {}
    }
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name} (Multi-Label)")
    print(f"{'='*60}")
    
    # Compute metrics per sample
    jaccard_scores = []
    exact_matches = []
    
    for i in range(len(y_true)):
        true_set = set(y_true[i])
        pred_set = set(y_pred[i])
        
        # Jaccard similarity
        intersection = len(true_set & pred_set)
        union = len(true_set | pred_set)
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard)
        
        # Exact match
        exact_matches.append(true_set == pred_set)
    
    overall_jaccard = np.mean(jaccard_scores)
    overall_exact = np.mean(exact_matches)
    
    print(f"Overall Jaccard Similarity: {overall_jaccard:.3f}")
    print(f"Overall Exact Match Rate: {overall_exact:.3f}")
    
    results['overall_jaccard'] = overall_jaccard
    results['overall_exact_match'] = overall_exact
    
    # Metrics by race
    print(f"\n--- Jaccard by Race ---")
    for race in df_test['race_cat'].unique():
        if pd.isna(race):
            continue
        mask = df_test['race_cat'] == race
        if mask.sum() > 0:
            race_jaccard = np.mean([jaccard_scores[i] for i in range(len(mask)) if mask.iloc[i]])
            race_exact = np.mean([exact_matches[i] for i in range(len(mask)) if mask.iloc[i]])
            results['by_race'][race] = {
                'jaccard': race_jaccard,
                'exact_match': race_exact,
                'count': mask.sum()
            }
            print(f"  {race:15s}: Jaccard={race_jaccard:.3f}, Exact={race_exact:.3f} (n={mask.sum()})")
    
    # Metrics by gender
    print(f"\n--- Jaccard by Gender ---")
    for gender in df_test['gender_cat'].unique():
        if pd.isna(gender):
            continue
        mask = df_test['gender_cat'] == gender
        if mask.sum() > 0:
            gender_jaccard = np.mean([jaccard_scores[i] for i in range(len(mask)) if mask.iloc[i]])
            gender_exact = np.mean([exact_matches[i] for i in range(len(mask)) if mask.iloc[i]])
            results['by_gender'][gender] = {
                'jaccard': gender_jaccard,
                'exact_match': gender_exact,
                'count': mask.sum()
            }
            print(f"  {gender:15s}: Jaccard={gender_jaccard:.3f}, Exact={gender_exact:.3f} (n={mask.sum()})")
    
    return results


def compute_fairness_gaps(all_results):
    """Compute fairness gaps (max - min accuracy across groups)"""
    
    print(f"\n{'='*60}")
    print("FAIRNESS GAPS (Lower is Better)")
    print(f"{'='*60}")
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        
        # Race gap
        if 'by_race' in results and results['by_race']:
            race_accs = [v['accuracy'] if 'accuracy' in v else v.get('jaccard', 0) 
                        for v in results['by_race'].values()]
            race_gap = max(race_accs) - min(race_accs)
            print(f"  Race Gap: {race_gap:.3f} (max-min across racial groups)")
        
        # Gender gap
        if 'by_gender' in results and results['by_gender']:
            gender_accs = [v['accuracy'] if 'accuracy' in v else v.get('jaccard', 0) 
                          for v in results['by_gender'].values()]
            gender_gap = max(gender_accs) - min(gender_accs)
            print(f"  Gender Gap: {gender_gap:.3f} (max-min across gender groups)")
        
        # Intersectional gap
        if 'by_race_gender' in results and results['by_race_gender']:
            intersect_accs = [v['accuracy'] if 'accuracy' in v else v.get('jaccard', 0) 
                            for v in results['by_race_gender'].values()]
            if intersect_accs:
                intersect_gap = max(intersect_accs) - min(intersect_accs)
                print(f"  Intersectional Gap: {intersect_gap:.3f} (max-min across race×gender)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate fairness across models')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to trained model files (.pkl)')
    parser.add_argument('--output', default='results/fairness_report.csv',
                       help='Output path for fairness report')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FAIRNESS EVALUATION")
    print("="*60)
    
    all_results = {}
    
    for model_path in args.models:
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"Warning: Model not found: {model_path}")
            continue
        
        # Load model
        print(f"\nLoading {model_path.name}...")
        model_data = load_model(model_path)
        
        model = model_data['model']
        encoder = model_data['encoder']
        label_type = model_data['label_type']
        X_test = model_data['X_test']
        y_test = model_data['y_test']
        df_test = model_data['df_test']
        
        model_name = model_path.stem
        
        # Make predictions
        if label_type == 'single':
            y_pred, y_pred_proba = predict_single_label(model, encoder, X_test)
            results = compute_fairness_metrics(y_test, y_pred, df_test, model_name)
        else:
            y_pred, y_pred_proba = predict_multi_label(model, encoder, X_test)
            results = compute_multi_label_fairness(y_test, y_pred, df_test, model_name)
        
        all_results[model_name] = results
    
    # Compute fairness gaps
    if len(all_results) > 1:
        compute_fairness_gaps(all_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    rows = []
    for model_name, results in all_results.items():
        # Overall
        row = {'model': model_name}
        
        if 'overall_accuracy' in results:
            row['overall_accuracy'] = results['overall_accuracy']
        if 'overall_jaccard' in results:
            row['overall_jaccard'] = results['overall_jaccard']
            row['overall_exact_match'] = results['overall_exact_match']
        
        # By race
        for race, metrics in results.get('by_race', {}).items():
            if 'accuracy' in metrics:
                row[f'acc_{race}'] = metrics['accuracy']
            if 'jaccard' in metrics:
                row[f'jaccard_{race}'] = metrics['jaccard']
        
        # By gender
        for gender, metrics in results.get('by_gender', {}).items():
            if 'accuracy' in metrics:
                row[f'acc_{gender}'] = metrics['accuracy']
            if 'jaccard' in metrics:
                row[f'jaccard_{gender}'] = metrics['jaccard']
        
        rows.append(row)
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_path, index=False)
    
    print(f"\n✓ Fairness report saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
