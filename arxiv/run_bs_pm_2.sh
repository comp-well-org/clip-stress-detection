for fold in 0 1 2 3 4; do
    for seed in 2 3; do
        for norm_config in quantile_global; do
            for seq_enc in transformer; do
                for tab_enc in resnet; do
                    for dataset in pmdata; do
                        for mode in simclr byol leaves; do
                            for label_ratio in 0.1 0.5 1.0; do
                                for unlabel_ratio in 1.0; do
                                    python main.py --train --dataset $dataset --exp run --mode $mode --linear --finetune --n_epochs 300 --batch_size 128 --seq_enc $seq_enc --tab_enc $tab_enc --fold $fold --norm_config $norm_config --seed $seed --exclude none --unlabel_ratio $unlabel_ratio --label_ratio $label_ratio
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done