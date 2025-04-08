#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/mickey_interp"

prompts=(
  'a cartoon Pikachu.'
#   'a plushie cat'
#   'the cartoon dog'
#   'a toy of a elephant'
#   'a photo of a cool dog'
#   'a tattoo of the dragon'
)

for model in "sdxl"; do
    for prompt in "${prompts[@]}"; do
        for seed in 0 42; do
            if [[ $model = 'sdxl' ]]; then
                num_denoising_steps=30
            else
                num_denoising_steps=1
            fi

            echo "Forward generating for $prompt $model $seed $num_denoising_steps"
            path="$PREFIX/forward/$model/$prompt/$seed/"
            mkdir -p "$path"

            if [ ! -f "$path/orig.png" ]; then
                python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/orig.png" --not_steer
            fi

            for beta in 2; do
                if [ ! -f "$path/casteer_${beta}.png" ]; then
                    python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/casteer_${beta}.png" --steering_vectors steering_vectors/mickey_laion_steering_vectors.pickle --beta "${beta}" --steer_type casteer --steer_back
                fi
            done

            for alpha in 1; do 
                if [ ! -f "$path/mmsteer_${alpha}.png" ]; then
                    python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/mmsteer_${alpha}.png" --steering_vectors steering_vectors/mickey_laion_mm_inverse_steering_vectors.pickle --alpha "${alpha}" --steer_type mmsteer
                fi
            done
        done
    done
done



# prompts=(
#   'a cartoon mickey'
#   'a plushie cat'
#   'a pixelated photo of mickey'
#   'a good photo of mickey mouse'
#   'a tattoo of the mickey mouse'
# )

# for model in "sdxl-turbo" "sdxl"; do
#     for prompt in "${prompts[@]}"; do

#         for seed in 0 42; do
    
#             if [[ $model = 'sdxl' ]]; then
#                 num_denoising_steps=30
#             else
#                 num_denoising_steps=1
#             fi
    
#             echo "Inverse generating for $prompt $model $seed $num_denoising_steps"
#             path="$PREFIX/inverse/$model/$prompt/$seed/"
#             mkdir -p "$path"
            
#             if [ ! -f "$path/orig.png" ]; then
#                 python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/orig.png" --not_steer
#             fi

#             if [ ! -f "$path/casteer.png" ]; then
#                 python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/casteer.png" --steering_vectors mickey_laion_steering_vectors.pickle --beta 2 --steer_type casteer --steer_back
#             fi

#             if [ ! -f "$path/mmsteer.png" ]; then
#                 python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/mmsteer.png" --steering_vectors mickey_laion_mm_inverse_steering_vectors.pickle --steer_type mmsteer
#             fi
#         done
    
#     done

# done