for fold in 2; do
    for seed in 0 1; do
        python gen_main0.py --dataset lifesnaps --seed $seed --fold $fold --id
    done
done