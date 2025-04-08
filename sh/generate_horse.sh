#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/horse_motorcycle"

prompts=(
  'Astronaut rides a horse in a jungle, cold color palette, muted colors, detailed, 8k'
  'Horse race in the snowy mountains, studio ghibli style'
  'A photograph of stables with beautiful horses, detailed, film grain'
  'A unicorn in the woods, my little pony style'
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

            for alpha in $(seq 3 3 21); do
                if [ ! -f "$path/casteer_${alpha}.png" ]; then
                    python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/casteer_${alpha}.png" --steering_vectors horse_to_motorcycle_laion_steering_vectors.pickle --alpha "${alpha}" --steer_type casteer
                fi
            done

            for alpha in $(seq 0.5 0.25 1.5); do 
                if [ ! -f "$path/mmsteer_${alpha}.png" ]; then
                    python generate_casteer.py --model $model --prompt "$prompt" --num_denoising_steps $num_denoising_steps --seed $seed --output "$path/mmsteer_${alpha}.png" --steering_vectors horse_to_motorcycle_laion_mm_steering_vectors.pickle --alpha "${alpha}" --steer_type mmsteer
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