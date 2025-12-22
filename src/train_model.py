#!/usr/bin/env python3
"""
Train demographic classification models on v1, v2, and v3 datasets.
Models predict race categories from face images.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Dummy feature extraction (replace with actual image features)
def extract_image_features(image_path, img_size=(224, 224)):
    """
    Extract features from image. For now, returns dummy features.
    In production, you would use:
    - Pre-trained CNN features (ResNet, VGG, etc.)
    - Face embeddings (FaceNet, ArcFace, etc.)
    """
    # Return dummy features for demonstration
    # Shape: (2048,) to simulate ResNet50 features
    np.random.seed(hash(str(image_path)) % 2**32)
    return np.random.randn(2048)


def load_dataset(csv_path, dataset_version, base_image_dir):
    """Load and prepare dataset based on version"""
    df = pd.read_csv(csv_path)
    
    print(f"Loading {dataset_version}...")
    print(f"  Total samples: {len(df)}")
    
    # Extract features (dummy for now)
    print("  Extracting image features...")
    features = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            # Construct image path
            rel_path = row['rel_path']
            img_path = Path(base_image_dir) / rel_path
            
            # Extract features
            feat = extract_image_features(img_path)
            features.append(feat)
            valid_indices.append(idx)
        except Exception as e:
            print(f"    Warning: Skipping {row.get('image_id', idx)}: {e}")
            continue
    
    df = df.iloc[valid_indices].reset_index(drop=True)
    X = np.array(features)
    
    print(f"  Valid samples: {len(df)}")
    
    # Prepare labels based on version
    if dataset_version == 'v1':
        # Single-label: race_cat
        y = df['race_cat'].values
        label_type = 'single'
        
    elif dataset_version == 'v2':
        # Single-label: race_cat (balanced version)
        y = df['race_cat'].values
        label_type = 'single'
        
    elif dataset_version == 'v3':
        # Multi-label: race_ml column
        # Format: "White,Asian" or single "White"
        race_ml = df['race_ml'].fillna(df['race_cat'])
        y = [str(r).split(',') if ',' in str(r) else [str(r)] for r in race_ml]
        label_type = 'multi'
    
    else:
        raise ValueError(f"Unknown dataset version: {dataset_version}")
    
    # Split by existing split column if available
    if 'split' in df.columns:
        train_mask = df['split'] == 'train'
        val_mask = df['split'] == 'val'
        test_mask = df['split'] == 'test'
        
        X_train = X[train_mask]
        X_val = X[val_mask] if val_mask.any() else None
        X_test = X[test_mask] if test_mask.any() else None
        
        y_train = y[train_mask] if label_type == 'single' else [y[i] for i in range(len(y)) if train_mask.iloc[i]]
        y_val = y[val_mask] if (label_type == 'single' and val_mask.any()) else ([y[i] for i in range(len(y)) if val_mask.iloc[i]] if val_mask.any() else None)
        y_test = y[test_mask] if (label_type == 'single' and test_mask.any()) else ([y[i] for i in range(len(y)) if test_mask.iloc[i]] if test_mask.any() else None)
        
        df_train = df[train_mask]
        df_val = df[val_mask] if val_mask.any() else None
        df_test = df[test_mask] if test_mask.any() else None
    else:
        # Create split
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx] if label_type == 'single' else [y[i] for i in train_idx]
        y_test = y[test_idx] if label_type == 'single' else [y[i] for i in test_idx]
        
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        df_val = None
        X_val, y_val = None, None
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test) if X_test is not None else 0}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'df_train': df_train,
        'df_val': df_val,
        'df_test': df_test,
        'label_type': label_type
    }


def train_single_label_model(X_train, y_train):
    """Train single-label classifier"""
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("  Training Random Forest...")
    model.fit(X_train, y_train_encoded)
    
    return model, le


def train_multi_label_model(X_train, y_train):
    """Train multi-label classifier"""
    # Encode labels
    mlb = MultiLabelBinarizer()
    y_train_encoded = mlb.fit_transform(y_train)
    
    # Train one-vs-rest classifiers
    from sklearn.multioutput import MultiOutputClassifier
    
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model = MultiOutputClassifier(base_model)
    
    print("  Training Multi-label Random Forest...")
    model.fit(X_train, y_train_encoded)
    
    return model, mlb


def main():
    parser = argparse.ArgumentParser(description='Train demographic classification models')
    parser.add_argument('--dataset', required=True, choices=['v1', 'v2', 'v3'],
                       help='Dataset version to train on')
    parser.add_argument('--output', required=True, help='Output path for trained model (.pkl)')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--image-dir', default='data', help='Base image directory')
    
    args = parser.parse_args()
    
    # Map dataset version to file
    dataset_files = {
        'v1': 'labels_v1.csv',
        'v2': 'labels_v2_balanced.csv',
        'v3': 'labels_v3_heuristic.csv'
    }
    
    csv_path = Path(args.data_dir) / dataset_files[args.dataset]
    
    if not csv_path.exists():
        print(f"Error: Dataset file not found: {csv_path}")
        return
    
    print("=" * 60)
    print(f"Training Model: {args.dataset.upper()}")
    print("=" * 60)
    
    # Load data
    data = load_dataset(csv_path, args.dataset, args.image_dir)
    
    # Train model
    if data['label_type'] == 'single':
        model, encoder = train_single_label_model(data['X_train'], data['y_train'])
    else:
        model, encoder = train_multi_label_model(data['X_train'], data['y_train'])
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'encoder': encoder,
        'label_type': data['label_type'],
        'dataset_version': args.dataset,
        'X_test': data['X_test'],
        'y_test': data['y_test'],
        'df_test': data['df_test']
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to: {output_path}")
    print(f"  Label type: {data['label_type']}")
    print(f"  Classes: {encoder.classes_ if hasattr(encoder, 'classes_') else list(encoder.classes_)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
