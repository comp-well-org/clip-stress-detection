for fold in 0 1 2 3 4; do
    for seed in 0 1 2 3; do
        for norm_config in quantile_global; do
            for seq_enc in transformer; do
                for tab_enc in resnet; do
                    for dataset in lifesnaps; do
                        for label_ratio in 1.0; do
                            python main.py --train --dataset $dataset --exp run --mode sup --linear --finetune --n_epochs 300 --batch_size 512 --seq_enc $seq_enc --tab_enc $tab_enc --fold $fold --norm_config $norm_config --seed $seed --exclude all --unlabel_ratio 0.0 --label_ratio $label_ratio
                        done
                    done
                done
            done
        done
    done
done