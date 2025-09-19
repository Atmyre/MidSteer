import argparse
import os

from PIL import Image

import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, help='Subdirectory to process')

    args = parser.parse_args()

    removed = 0
    for file in glob.glob(f'{args.dir}/**/*.png', recursive=True):
        try:
            Image.open(file)
        except OSError:
            print(f'Removing {file} because it is broken')
            removed += 1
            os.remove(file)

    print(f'Removed {removed} broken files in {args.dir}')
