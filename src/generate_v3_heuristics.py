#!/usr/bin/env python3
"""
Generate heuristic multi-class labels for v3 dataset.

Strategy for high agreement:
1. Conservative confidence scores (lower when uncertain)
2. Flag ambiguity rather than forcing labels
3. Use image analysis for skin tone estimation
4. Preserve original single labels as baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def estimate_skin_tone(image_path, data_root):
    """
    Estimate skin tone on 1-7 scale using luminance.
    
    Conservative approach: use center region of image (likely face).
    Returns tuple: (skin_tone_bin, confidence)
    """
    try:
        full_path = Path(data_root) / image_path
        if not full_path.exists():
            return None, 0.0
        
        img = Image.open(full_path).convert('RGB')
        width, height = img.size
        
        # Extract center region (50% of image, likely face)
        left = width // 4
        top = height // 4
        right = 3 * width // 4
        bottom = 3 * height // 4
        face_region = img.crop((left, top, right, bottom))
        
        # Convert to numpy for analysis
        pixels = np.array(face_region)
        
        # Calculate perceived luminance (ITU-R BT.601)
        luminance = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]
        mean_luminance = np.mean(luminance)
        std_luminance = np.std(luminance)
        
        # Map luminance (0-255) to skin tone scale (1-7)
        # Lower luminance = darker skin
        # This is a coarse approximation
        if mean_luminance < 60:
            tone = 1
        elif mean_luminance < 90:
            tone = 2
        elif mean_luminance < 120:
            tone = 3
        elif mean_luminance < 150:
            tone = 4
        elif mean_luminance < 180:
            tone = 5
        elif mean_luminance < 210:
            tone = 6
        else:
            tone = 7
        
        # Confidence based on uniformity (low std = more confident)
        # High variance might indicate lighting issues or non-face content
        if std_luminance > 50:
            confidence = 0.5  # Low confidence
        elif std_luminance > 30:
            confidence = 0.7
        else:
            confidence = 0.8
        
        return int(tone), confidence
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, 0.0


def detect_cultural_markers(image_path, data_root):
    """
    Placeholder for cultural marker detection.
    
    In real implementation, this would use:
    - Object detection for headwear/attire
    - Pattern recognition for tattoos/piercings
    
    For now, returns conservative default (none detected).
    """
    # Conservative: don't guess, return "none" unless we're certain
    return "none", 0.9  # High confidence in "none" for most images


def assign_race_ml(race_cat):
    """
    Convert single-label race to multi-label format.
    
    Conservative approach: 1-to-1 mapping initially.
    Humans can add additional labels during manual annotation.
    """
    if pd.isna(race_cat) or race_cat == "":
        return "", 0.0
    
    # Direct mapping from single label
    race_ml = race_cat
    
    # Lower confidence for categories that are often mixed/ambiguous
    if race_cat in ["Latino", "MiddleEastern", "SoutheastAsian"]:
        confidence = 0.6  # These categories have high ambiguity
    else:
        confidence = 0.7  # Moderate confidence (never claim high confidence from heuristics)
    
    return race_ml, confidence


def assign_ambiguity_flags(race_cat, skin_tone_conf):
    """
    Flag samples that are likely ambiguous.
    
    Strategy: Be liberal with flagging (better to flag uncertainty than miss it).
    """
    # Flag as potentially ambiguous if:
    # - Latino/MiddleEastern (often multiracial)
    # - Low skin tone confidence (lighting/image quality issues)
    
    ambiguous_mixed = 0
    unknown_uncertain = 0
    
    if race_cat in ["Latino", "MiddleEastern"]:
        ambiguous_mixed = 1  # These categories often represent mixed heritage
    
    if skin_tone_conf < 0.6:
        unknown_uncertain = 1  # Low confidence = uncertain
    
    return ambiguous_mixed, unknown_uncertain


def assign_gender_confidence(gender_cat):
    """
    Assign confidence for gender labels.
    
    Conservative: moderate confidence since presentation != identity.
    """
    if pd.isna(gender_cat) or gender_cat == "":
        return 0.0
    
    # Moderate confidence (original labels are appearance-based)
    return 0.75


def generate_heuristic_labels(input_csv, output_csv, data_root, sample_images=True):
    """
    Generate heuristic v3 labels from v1 single labels.
    
    Args:
        input_csv: Path to labels_v3.csv (with empty multi-class columns)
        output_csv: Path to save heuristic-labeled version
        data_root: Root directory containing images
        sample_images: If True, process images for skin tone. If False, skip (faster for testing)
    """
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Processing {len(df)} samples...")
    
    # Initialize lists for new values
    race_ml_list = []
    skin_tone_list = []
    cultural_markers_list = []
    ambiguous_mixed_list = []
    prefer_not_to_label_list = []
    unknown_uncertain_list = []
    conf_race_list = []
    conf_gender_list = []
    conf_skin_list = []
    annotation_notes_list = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(df)}...")
        
        # 1. Race multi-label
        race_ml, conf_race = assign_race_ml(row['race_cat'])
        race_ml_list.append(race_ml)
        conf_race_list.append(conf_race)
        
        # 2. Skin tone (from image analysis if enabled)
        if sample_images and pd.notna(row['rel_path']):
            skin_tone, conf_skin = estimate_skin_tone(row['rel_path'], data_root)
        else:
            skin_tone, conf_skin = None, 0.0
        
        skin_tone_list.append(skin_tone if skin_tone else "")
        conf_skin_list.append(conf_skin)
        
        # 3. Cultural markers
        cultural_markers, _ = detect_cultural_markers(row.get('rel_path', ''), data_root)
        cultural_markers_list.append(cultural_markers)
        
        # 4. Ambiguity flags
        ambiguous, uncertain = assign_ambiguity_flags(row['race_cat'], conf_skin)
        ambiguous_mixed_list.append(ambiguous)
        unknown_uncertain_list.append(uncertain)
        prefer_not_to_label_list.append(0)  # Default: willing to label (heuristic choice)
        
        # 5. Gender confidence
        conf_gender = assign_gender_confidence(row['gender_cat'])
        conf_gender_list.append(conf_gender)
        
        # 6. Annotation notes
        notes = f"Heuristic label (conf_race={conf_race:.2f})"
        if ambiguous:
            notes += "; flagged_ambiguous"
        if uncertain:
            notes += "; flagged_uncertain"
        annotation_notes_list.append(notes)
    
    # Assign to dataframe
    df['race_ml'] = race_ml_list
    df['skin_tone_bin'] = skin_tone_list
    df['cultural_markers'] = cultural_markers_list
    df['ambiguous_mixed'] = ambiguous_mixed_list
    df['prefer_not_to_label'] = prefer_not_to_label_list
    df['unknown_uncertain'] = unknown_uncertain_list
    df['conf_race'] = conf_race_list
    df['conf_gender'] = conf_gender_list
    df['conf_skin'] = conf_skin_list
    df['annotation_notes'] = annotation_notes_list
    
    # Save
    print(f"Saving to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print("\nSummary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Samples flagged ambiguous_mixed: {df['ambiguous_mixed'].sum()}")
    print(f"  Samples flagged unknown_uncertain: {df['unknown_uncertain'].sum()}")
    print(f"  Mean conf_race: {df['conf_race'].mean():.3f}")
    print(f"  Mean conf_gender: {df['conf_gender'].mean():.3f}")
    print(f"  Mean conf_skin: {df['conf_skin'].mean():.3f}")
    print(f"\nHeuristic labels saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate heuristic v3 labels from single-label dataset"
    )
    parser.add_argument(
        '--input',
        default='data/labels_v3.csv',
        help='Input CSV with empty v3 columns (default: data/labels_v3.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/labels_v3_heuristic.csv',
        help='Output CSV with heuristic labels (default: data/labels_v3_heuristic.csv)'
    )
    parser.add_argument(
        '--data-root',
        default='data',
        help='Root directory containing images (default: data)'
    )
    parser.add_argument(
        '--skip-images',
        action='store_true',
        help='Skip image processing (faster, but no skin_tone_bin)'
    )
    
    args = parser.parse_args()
    
    generate_heuristic_labels(
        input_csv=args.input,
        output_csv=args.output,
        data_root=args.data_root,
        sample_images=not args.skip_images
    )


if __name__ == '__main__':
    main()
