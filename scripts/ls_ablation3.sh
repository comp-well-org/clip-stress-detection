for fold in 2; do
    for seed in 0 1; do
        for ablation_idx in 0 1 2 3 4; do
            python ablations.py --dataset lifesnaps --seed $seed --fold $fold --ablation_idx $ablation_idx --gpu 0
        done
    done
done