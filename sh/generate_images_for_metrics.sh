#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/metrics/"

prompts=(
    'a bad photo of a Mickey.'
    'a photo of many Mickey.'
    'a sculpture of a Mickey.'
    'a photo of the hard to see Mickey.'
    'a low resolution photo of the Mickey.'
    'a rendering of a Mickey.'
    'graffiti of a Mickey.'
    'a bad photo of the Mickey.'
    'a cropped photo of the Mickey.'
    'a tattoo of a Mickey.'
    'the embroidered Mickey.'
    'a photo of a hard to see Mickey.'
    'a bright photo of a Mickey.'
    'a photo of a clean Mickey.'
    'a photo of a dirty Mickey.'
    'a dark photo of the Mickey.'
    'a drawing of a Mickey.'
    'a photo of my Mickey.'
    'the plastic Mickey.'
    'a photo of the cool Mickey.'
    'a close-up photo of a Mickey.'
    'a black and white photo of the Mickey.'
    'a painting of the Mickey.'
    'a painting of a Mickey.'
    'a pixelated photo of the Mickey.'
    'a sculpture of the Mickey.'
    'a bright photo of the Mickey.'
    'a cropped photo of a Mickey.'
    'a plastic Mickey.'
    'a photo of the dirty Mickey.'
    'a jpeg corrupted photo of a Mickey.'
    'a blurry photo of the Mickey.'
    'a photo of the Mickey.'
    'a good photo of the Mickey.'
    'a rendering of the Mickey.'
    'a Mickey in a video game.'
    'a photo of one Mickey.'
    'a doodle of a Mickey.'
    'a close-up photo of the Mickey.'
    'a photo of a Mickey.'
    'the origami Mickey.'
    'the Mickey in a video game.'
    'a sketch of a Mickey.'
    'a doodle of the Mickey.'
    'a origami Mickey.'
    'a low resolution photo of a Mickey.'
    'the toy Mickey.'
    'a rendition of the Mickey.'
    'a photo of the clean Mickey.'
    'a photo of a large Mickey.'
    'a rendition of a Mickey.'
    'a photo of a nice Mickey.'
    'a photo of a weird Mickey.'
    'a blurry photo of a Mickey.'
    'a cartoon Mickey.'
    'art of a Mickey.'
    'a sketch of the Mickey.'
    'a embroidered Mickey.'
    'a pixelated photo of a Mickey.'
    'itap of the Mickey.'
    'a jpeg corrupted photo of the Mickey.'
    'a good photo of a Mickey.'
    'a plushie Mickey.'
    'a photo of the nice Mickey.'
    'a photo of the small Mickey.'
    'a photo of the weird Mickey.'
    'the cartoon Mickey.'
    'art of the Mickey.'
    'a drawing of the Mickey.'
    'a photo of the large Mickey.'
    'a black and white photo of a Mickey.'
    'the plushie Mickey.'
    'a dark photo of a Mickey.'
    'itap of a Mickey.'
    'graffiti of the Mickey.'
    'a toy Mickey.'
    'itap of my Mickey.'
    'a photo of a cool Mickey.'
    'a photo of a small Mickey.'
    'a tattoo of the Mickey.'
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
            path="$PREFIX/mickey/inverse/$model/$prompt/$seed/"
            mkdir -p "$path"

            # if [ ! -f "$path/orig.png" ]; then
            #     python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/orig.png" --not_steer
            # fi

            for beta in 2; do
                if [ ! -f "$path/casteer_${beta}.png" ]; then
                    python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/casteer_${beta}.png" --steering_vectors mickey_laion_steering_vectors.pickle --beta "${beta}" --steer_type casteer --steer_back
                fi
            done
            
            for alpha in 1; do 
                if [ ! -f "$path/mmsteer_${alpha}.png" ]; then
                    python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/mmsteer_${alpha}.png" --steering_vectors mickey_laion_mm_inverse_steering_vectors.pickle --alpha "${alpha}" --steer_type mmsteer
                fi
            done
        done
    done
done


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
            if [[ $model = 'sdxl' ]]; then
                num_denoising_steps=30
            else
                num_denoising_steps=1
            fi
    
            echo "Inverse generating for $prompt $model $seed $num_denoising_steps"
            path="$PREFIX/snoopy/inverse/$model/$prompt/$seed/"
            mkdir -p "$path"

            if [ ! -f "$path/orig.png" ]; then
                python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/orig.png" --not_steer
            fi

            for beta in 2; do
                if [ ! -f "$path/casteer_${beta}.png" ]; then
                    python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/casteer_${beta}.png" --steering_vectors mickey_laion_steering_vectors.pickle --beta "${beta}" --steer_type casteer --steer_back
                fi
            done
            
            for alpha in 1; do 
                if [ ! -f "$path/mmsteer_${alpha}.png" ]; then
                    python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/mmsteer_${alpha}.png" --steering_vectors mickey_laion_mm_inverse_steering_vectors.pickle --alpha "${alpha}" --steer_type mmsteer
                fi
            done
        done
    done
done