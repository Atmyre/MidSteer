#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/metrics/"



prompts=(
    'a bad photo of a Snoopy.'
    'a photo of many Snoopy.'
    'a sculpture of a Snoopy.'
    'a photo of the hard to see Snoopy.'
    'a low resolution photo of the Snoopy.'
    'a rendering of a Snoopy.'
    'graffiti of a Snoopy.'
    'a bad photo of the Snoopy.'
    'a cropped photo of the Snoopy.'
    'a tattoo of a Snoopy.'
    'the embroidered Snoopy.'
    'a photo of a hard to see Snoopy.'
    'a bright photo of a Snoopy.'
    'a photo of a clean Snoopy.'
    'a photo of a dirty Snoopy.'
    'a dark photo of the Snoopy.'
    'a drawing of a Snoopy.'
    'a photo of my Snoopy.'
    'the plastic Snoopy.'
    'a photo of the cool Snoopy.'
    'a close-up photo of a Snoopy.'
    'a black and white photo of the Snoopy.'
    'a painting of the Snoopy.'
    'a painting of a Snoopy.'
    'a pixelated photo of the Snoopy.'
    'a sculpture of the Snoopy.'
    'a bright photo of the Snoopy.'
    'a cropped photo of a Snoopy.'
    'a plastic Snoopy.'
    'a photo of the dirty Snoopy.'
    'a jpeg corrupted photo of a Snoopy.'
    'a blurry photo of the Snoopy.'
    'a photo of the Snoopy.'
    'a good photo of the Snoopy.'
    'a rendering of the Snoopy.'
    'a Snoopy in a video game.'
    'a photo of one Snoopy.'
    'a doodle of a Snoopy.'
    'a close-up photo of the Snoopy.'
    'a photo of a Snoopy.'
    'the origami Snoopy.'
    'the Snoopy in a video game.'
    'a sketch of a Snoopy.'
    'a doodle of the Snoopy.'
    'a origami Snoopy.'
    'a low resolution photo of a Snoopy.'
    'the toy Snoopy.'
    'a rendition of the Snoopy.'
    'a photo of the clean Snoopy.'
    'a photo of a large Snoopy.'
    'a rendition of a Snoopy.'
    'a photo of a nice Snoopy.'
    'a photo of a weird Snoopy.'
    'a blurry photo of a Snoopy.'
    'a cartoon Snoopy.'
    'art of a Snoopy.'
    'a sketch of the Snoopy.'
    'a embroidered Snoopy.'
    'a pixelated photo of a Snoopy.'
    'itap of the Snoopy.'
    'a jpeg corrupted photo of the Snoopy.'
    'a good photo of a Snoopy.'
    'a plushie Snoopy.'
    'a photo of the nice Snoopy.'
    'a photo of the small Snoopy.'
    'a photo of the weird Snoopy.'
    'the cartoon Snoopy.'
    'art of the Snoopy.'
    'a drawing of the Snoopy.'
    'a photo of the large Snoopy.'
    'a black and white photo of a Snoopy.'
    'the plushie Snoopy.'
    'a dark photo of a Snoopy.'
    'itap of a Snoopy.'
    'graffiti of the Snoopy.'
    'a toy Snoopy.'
    'itap of my Snoopy.'
    'a photo of a cool Snoopy.'
    'a photo of a small Snoopy.'
    'a tattoo of the Snoopy.'
)

for model in "sdxl"; do
    for prompt in "${prompts[@]}"; do
        for seed in 0 1 2 3 4; do
            echo "Inverse generating for $prompt $model $seed "
            path="$PREFIX/snoopy/inverse/$model/$prompt/$seed/"
            mkdir -p "$path"

            if [ ! -f "$path/orig.png" ]; then
                CUDA_VISIBLE_DEVICES=$((seed % 2)) python generate_casteer.py --model $model --prompt "$prompt"  --seed $seed --output "$path/orig.png" --not_steer &
            fi

            for beta in 2; do
                if [ ! -f "$path/casteer_${beta}.png" ]; then
                    CUDA_VISIBLE_DEVICES=$((seed % 2 + 1) python generate_casteer.py --model $model --prompt "$prompt"  --seed $seed --output "$path/casteer_${beta}.png" --steering_vectors mickey_laion_steering_vectors.pickle --beta "${beta}" --steer_type casteer --steer_back &
                fi
            done
            
            for alpha in 1; do 
                if [ ! -f "$path/mmsteer_${alpha}.png" ]; then
                    CUDA_VISIBLE_DEVICES=$((seed % 2 + 2)) python generate_casteer.py --model $model --prompt "$prompt"  --seed $seed --output "$path/mmsteer_${alpha}.png" --steering_vectors mickey_laion_mm_inverse_steering_vectors.pickle --alpha "${alpha}" --steer_type mmsteer &
                fi
            done

            if (( seed % 2 == 1 || seed == 4 )); then
                wait
            fi
        done
    done
done