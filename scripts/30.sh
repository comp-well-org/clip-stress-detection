for fold in 3; do
    for seed in 0 1; do
        for norm_config in quantile_global; do
            for seq_enc in transformer; do
                for tab_enc in resnet; do
                    for exclude in label; do
                        for label_ratio in 0.3; do
                            for unlabel_ratio in 0.01 0.1 0.5 1.0; do
                                python main.py --train --dataset pmdata --exp review --mode clip --linear --finetune --n_epochs 300 --batch_size 512 --seq_enc $seq_enc --tab_enc $tab_enc --fold $fold --norm_config $norm_config --seed $seed --exclude $exclude --unlabel_ratio $unlabel_ratio --label_ratio $label_ratio
                            done
                        done
                    done
                done
            done
        done
    done
done