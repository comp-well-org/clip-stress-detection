for fold in 0; do
    for seed in 0 1; do
        for ablation_idx in 0 1 2 3; do
            python ablations1.py --dataset pmdata --seed $seed --fold $fold --ablation_idx $ablation_idx --gpu 1
        done
    done
done