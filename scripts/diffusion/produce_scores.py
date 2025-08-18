import argparse
import os

import pandas as pd

from core.eval.fid import compute_fid
from core.eval.clip import compute_clip

def main(
        dir: list,
        concepts: list[str],
):
    orig_path = os.path.join(dir, "orig")
    
    fids = []
    for subdir in os.listdir(dir):
        if subdir == "orig":
            continue
        subdir_path = os.path.join(dir, subdir)
        fid_score = compute_fid(subdir_path, '*.png', orig_path, '*.png')
        fids.append({
            'method': subdir,
            'fid': fid_score,
        })
    pd.DataFrame(fids).to_csv(f'{dir}/fid.tsv', index=False, sep='\t', encoding='utf-8')

    clip_scores = []
    for subdir in os.listdir(dir):
        for concept in concepts:
            clip_score, clip_accuracy = compute_clip(subdir_path, '*.png', concept)
            clip_scores.append({
                'method': subdir,
                'clip_score': clip_score,
                'clip_accuracy': clip_accuracy,
                'concept': concept,
            })


    pd.DataFrame(clip_scores).to_csv(f'{dir}/clip_score.tsv', index=False, sep='\t', encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, help='Subdirectory to process')
    parser.add_argument('--concept', type=str, nargs='+', help='Concept to score against')

    args = parser.parse_args()

    main(
        dir=args.dir,
        concepts=list(set(args.concept)),
    )