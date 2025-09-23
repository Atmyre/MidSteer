import argparse
import os

import pandas as pd

from core.eval.clip import compute_clip
from cleanfid import fid

def main(
        dir: list,
        concepts: list[str],
        num_workers: int,
        batch_size: int,
):
    orig_path = os.path.join(dir, "orig")
    method_dirs = [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]
    
    fids = []
    for subdir in method_dirs:
        if subdir == "orig":
            continue
        subdir_path = os.path.join(dir, subdir)
        print(f'Computing FID for {subdir_path}')
        fid_score = fid.compute_fid(
            subdir_path,
            orig_path,
            verbose=False,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        fids.append({
            'method': subdir,
            'fid': fid_score,
        })
        print(f'{subdir_path}, {fid_score}')
    pd.DataFrame(fids).to_csv(f'{dir}/fid.tsv', index=False, sep='\t', encoding='utf-8')

    clip_scores = []
    for subdir in method_dirs:
        subdir_path = os.path.join(dir, subdir)
        for concept in concepts:
            print(f'Computing CLIP for {subdir_path} and concept {concept}')
            clip_score, clip_accuracy = compute_clip(subdir_path, concept)
            clip_scores.append({
                'method': subdir,
                'clip_score': clip_score,
                'clip_accuracy': clip_accuracy,
                'concept': concept,
            })
            print(f'{subdir_path}, {clip_score}')


    pd.DataFrame(clip_scores).to_csv(f'{dir}/clip_score.tsv', index=False, sep='\t', encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, help='Subdirectory to process')
    parser.add_argument('--concept', type=str, nargs='*', help='Concept to score against')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of workers to use for FID and CLIP')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for FID and CLIP')

    args = parser.parse_args()

    if args.concept is None:
        concepts = []
    else:
        concepts = list(set(args.concept))

    main(
        dir=args.dir,
        concepts=concepts,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
