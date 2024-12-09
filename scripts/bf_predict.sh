for hd in 128 256 512
do
    for ne in 1 2 3 4 5 
    do
        for de in 1 2 3 4 5 
        do  
            for VAR in 1 2 3 4
            do
                for ck in {30..199}
                do
                    echo "split"$VAR
                    python main.py --hidden_dim $hd --n_encoder_layer $ne --n_decoder_layer $de \
                        --seg --task long --anticipate --pos_emb \
                        --predict --epochs 200 --mode=train --batch_size 16 --input_type=i3d_transcript --split=$VAR --checkpoint=$ck
                done
            done
        done
    done
done