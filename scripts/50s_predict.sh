for hd in 128 256 512
    for ne in 1 2 3 4 5
    do
        for de in 1 2 3 4 5
        do  
            for VAR in 1 2 3 4 5
            do
                for ck in {30..199}
                do
                    echo "split"$VAR
                    python main.py --hidden_dim $hd --n_encoder_layer $ne --n_decoder_layer $de\
                    --n_query 20 --seg --task long --pos_emb --anticipate \
                    --max_pos_len 3100 --sample_rate 6 --dataset 50salads --predict --mode=train --split=$VAR --checkpoint=$ck
                done
            done
        done
    done
done