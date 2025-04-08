#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/metrics/"



prompts=(
    'a bad photo of a Pikachu.'
    'a photo of many Pikachu.'
    'a sculpture of a Pikachu.'
    'a photo of the hard to see Pikachu.'
    'a low resolution photo of the Pikachu.'
    'a rendering of a Pikachu.'
    'graffiti of a Pikachu.'
    'a bad photo of the Pikachu.'
    'a cropped photo of the Pikachu.'
    'a tattoo of a Pikachu.'
    'the embroidered Pikachu.'
    'a photo of a hard to see Pikachu.'
    'a bright photo of a Pikachu.'
    'a photo of a clean Pikachu.'
    'a photo of a dirty Pikachu.'
    'a dark photo of the Pikachu.'
    'a drawing of a Pikachu.'
    'a photo of my Pikachu.'
    'the plastic Pikachu.'
    'a photo of the cool Pikachu.'
    'a close-up photo of a Pikachu.'
    'a black and white photo of the Pikachu.'
    'a painting of the Pikachu.'
    'a painting of a Pikachu.'
    'a pixelated photo of the Pikachu.'
    'a sculpture of the Pikachu.'
    'a bright photo of the Pikachu.'
    'a cropped photo of a Pikachu.'
    'a plastic Pikachu.'
    'a photo of the dirty Pikachu.'
    'a jpeg corrupted photo of a Pikachu.'
    'a blurry photo of the Pikachu.'
    'a photo of the Pikachu.'
    'a good photo of the Pikachu.'
    'a rendering of the Pikachu.'
    'a Pikachu in a video game.'
    'a photo of one Pikachu.'
    'a doodle of a Pikachu.'
    'a close-up photo of the Pikachu.'
    'a photo of a Pikachu.'
    'the origami Pikachu.'
    'the Pikachu in a video game.'
    'a sketch of a Pikachu.'
    'a doodle of the Pikachu.'
    'a origami Pikachu.'
    'a low resolution photo of a Pikachu.'
    'the toy Pikachu.'
    'a rendition of the Pikachu.'
    'a photo of the clean Pikachu.'
    'a photo of a large Pikachu.'
    'a rendition of a Pikachu.'
    'a photo of a nice Pikachu.'
    'a photo of a weird Pikachu.'
    'a blurry photo of a Pikachu.'
    'a cartoon Pikachu.'
    'art of a Pikachu.'
    'a sketch of the Pikachu.'
    'a embroidered Pikachu.'
    'a pixelated photo of a Pikachu.'
    'itap of the Pikachu.'
    'a jpeg corrupted photo of the Pikachu.'
    'a good photo of a Pikachu.'
    'a plushie Pikachu.'
    'a photo of the nice Pikachu.'
    'a photo of the small Pikachu.'
    'a photo of the weird Pikachu.'
    'the cartoon Pikachu.'
    'art of the Pikachu.'
    'a drawing of the Pikachu.'
    'a photo of the large Pikachu.'
    'a black and white photo of a Pikachu.'
    'the plushie Pikachu.'
    'a dark photo of a Pikachu.'
    'itap of a Pikachu.'
    'graffiti of the Pikachu.'
    'a toy Pikachu.'
    'itap of my Pikachu.'
    'a photo of a cool Pikachu.'
    'a photo of a small Pikachu.'
    'a tattoo of the Pikachu.'
)

for model in "sdxl"; do
    for prompt in "${prompts[@]}"; do
        for seed in 0 1 2 3 4; do
            if [[ $model = 'sdxl' ]]; then
                num_denoising_steps=30
            else
                num_denoising_steps=1
            fi
    
            echo "Inverse generating for $prompt $model $seed $num_denoising_steps"
            path="$PREFIX/snoopy/inverse/$model/$prompt/$seed/"
            mkdir -p "$path"

            if [ ! -f "$path/orig.png" ]; then
                CUDA_VISIBLE_DEVICES=$(( 3 * (seed % 2) )) python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/orig.png" --not_steer &
            fi

            for beta in 2; do
                if [ ! -f "$path/casteer_${beta}.png" ]; then
                    CUDA_VISIBLE_DEVICES=$(( 3 * (seed % 2) + 1 )) python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/casteer_${beta}.png" --steering_vectors steering_vectors/mickey_laion_steering_vectors.pickle --beta "${beta}" --steer_type casteer --steer_back &
                fi
            done
            
            for alpha in 1; do 
                if [ ! -f "$path/mmsteer_${alpha}.png" ]; then
                    CUDA_VISIBLE_DEVICES=$(( 3 * (seed % 2) + 2 )) python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/mmsteer_${alpha}.png" --steering_vectors steering_vectors/mickey_laion_mm_inverse_steering_vectors.pickle --alpha "${alpha}" --steer_type mmsteer &
                fi
            done

            if (( seed % 2 == 1 || seed == 4 )); then
                wait
            fi
        done
    done
done