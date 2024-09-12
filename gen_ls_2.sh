for fold in 0; do
    for seed in 2 3; do
        python gen_main0.py --dataset lifesnaps --seed $seed --fold $fold --id
    done
done