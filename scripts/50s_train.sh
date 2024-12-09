for hd in 128 256 512
do   
    for ne in 1 2 3 4 5
    do
        for de in 1 2 3 4 5
        do  
            for VAR in 1 2 3 4 5
            do
                python main.py \
                --task long --seg --anticipate --pos_emb\
                --n_query 20 --n_encoder_layer $ne --n_decoder_layer $de --batch_size 8 --hidden_dim $hd \
                --dataset 50salads --max_pos_len 3100 --sample_rate 6 --epochs 200 --mode=train --input_type=i3d_transcript --split=$VAR
            done
        done
    done
done