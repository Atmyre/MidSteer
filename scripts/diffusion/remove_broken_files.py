import argparse
import numpy as np
import os

from PIL import Image
from tqdm import tqdm
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, help='Subdirectory to process')

    args = parser.parse_args()

    removed = 0
    total = 0
    for file in tqdm(glob.glob(f'{args.dir}/**/*.png', recursive=True)):
        total += 1
        try:
            image_np = Image.open(file).convert('RGB')
            image_np = np.array(image_np)
        except OSError:
            print(f'Removing {file} because it is broken')
            removed += 1
            print('Removed:', removed)
            os.remove(file)

    print(f'Removed {removed} broken files out of {total} in {args.dir}')
