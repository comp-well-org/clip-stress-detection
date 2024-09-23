for fold in 4; do
    for seed in 0 1; do
        python gen_main1.py --dataset pmdata --seed $seed --fold $fold --id
    done
done