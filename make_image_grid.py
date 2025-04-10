import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

base_dir = "images/interp_horse_motorcycle/forward/sdxl"

# Helper function to extract strength and method
def parse_file_info(filename):
    if filename.startswith("orig"):
        return -1, "orig"
    elif filename.startswith("casteer"):
        return float(filename.split("_")[1].replace(".png", "")), "casteer"
    elif filename.startswith("mmsteer"):
        return float(filename.split("_")[1].replace(".png", "")), "mmsteer"
    return None, None

def create_two_row_grid(orig_img, casteer_imgs, mmsteer_imgs, img_size=(512, 512), out_path="grid.png"):
    cols = max(len(casteer_imgs), len(mmsteer_imgs)) + 1  # one extra for orig
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    if isinstance(axes, np.ndarray):
        axes = axes.reshape(rows, cols)

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # First column = orig
    for row in range(2):
        axes[row][0].imshow(orig_img.resize(img_size))
        axes[row][0].set_title("orig")

    # Fill in casteer
    for i, (strength, img) in enumerate(casteer_imgs):
        axes[0][i + 1].imshow(img.resize(img_size))
        axes[0][i + 1].set_title(f"casteer_{strength}")

    # Fill in mmsteer
    for i, (strength, img) in enumerate(mmsteer_imgs):
        axes[1][i + 1].imshow(img.resize(img_size))
        axes[1][i + 1].set_title(f"mmsteer_{strength}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()

# Iterate through prompts
for prompt in os.listdir(base_dir):
    prompt_path = os.path.join(base_dir, prompt)
    if not os.path.isdir(prompt_path):
        continue

    for seed in os.listdir(prompt_path):
        seed_path = os.path.join(prompt_path, seed)
        if not os.path.isdir(seed_path):
            continue

        orig_img = None
        casteer_imgs = []
        mmsteer_imgs = []

        for file in os.listdir(seed_path):
            file_path = os.path.join(seed_path, file)
            strength, method = parse_file_info(file)
            if strength is None:
                continue
            img = Image.open(file_path)
            if method == "orig":
                orig_img = img
            elif method == "casteer":
                casteer_imgs.append((strength, img))
            elif method == "mmsteer":
                mmsteer_imgs.append((strength, img))

        # Sort by strength
        casteer_imgs.sort(key=lambda x: x[0])
        mmsteer_imgs.sort(key=lambda x: x[0])

        if orig_img is None:
            print(f"Skipping {prompt} seed {seed} due to missing orig image.")
            continue

        out_file = os.path.join(prompt_path, f"{seed}_comparison_grid.png")
        create_two_row_grid(orig_img, casteer_imgs, mmsteer_imgs, out_path=out_file)
        print(f"Saved grid for {prompt} seed {seed} at: {out_file}")