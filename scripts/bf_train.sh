for hd in 128 256 512
do
    for ne in 1 2 3 4 5
    do
        for de in 1 2 3 4 5 
        do  
            for VAR in 1 2 3 4
            do
                python main.py \
                --task long \
                --seg --anticipate --pos_emb\
                --n_encoder_layer $ne --n_decoder_layer $de --batch_size 16 --hidden_dim $hd --max_pos_len 2000\
                --epochs 200 --mode=train --input_type=i3d_transcript --split=$VAR
            done
        done
    done
done