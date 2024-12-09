import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import pdb
import random
from torch.backends import cudnn
from opts import parser
from utils import read_mapping_dict
from data.basedataset import BaseDataset
from model.lrts import LRTS
from train import train
from model.ExtraLib.LinearWarmupCALR import LinearWarmupCosineAnnealingLR
from predict import predict
import pickle

device = torch.device('cuda:0')

# Seed fix
seed = 13452
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benchmark, cudnn.deterministic = False, True


def main():
    args = parser.parse_args()

    if args.cpu:
        device = torch.device('cpu')
        print('using cpu')
    else:
        device = torch.device('cuda')
        print('using gpu')

    print('model type : ', args.model)
    print('input type : ', args.input_type)

    print("| encoder layers : ",args.n_encoder_layer)
    print("| decoder layers : ",args.n_decoder_layer)
    print("| nheads : ",args.n_head)
    print("| n_query : ",args.n_query)
    print("| hidden dim : ",args.hidden_dim)

    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)
    print("Split : ", args.split)

    dataset = args.dataset
    task = args.task
    split = args.split

    if dataset == 'breakfast':
        data_path = './datasets/breakfast'
    elif dataset == '50salads' :
        data_path = './datasets/50salads'

    mapping_file = os.path.join(data_path, 'mapping.txt')
    actions_dict = read_mapping_dict(mapping_file)
    video_file_path = os.path.join(data_path, 'splits', 'train.split'+args.split+'.bundle' )
    video_file_test_path = os.path.join(data_path, 'splits', 'test.split'+args.split+'.bundle' )

    video_file = open(video_file_path, 'r')
    video_file_test = open(video_file_test_path, 'r')

    video_list = video_file.read().split('\n')[:-1]
    video_test_list = video_file_test.read().split('\n')[:-1]

    features_path = os.path.join(data_path, 'features')
    gt_path = os.path.join(data_path, 'groundTruth')

    n_class = len(actions_dict) + 1
    pad_idx = n_class + 1


    # Model specification
    model = LRTS(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer).to(device)

    # model_save_path = os.path.join('./save_dir', args.dataset, args.task, 'model/transformer', split, args.input_type, \
    #                                 'runs'+str(args.runs))
    model_save_path = os.path.join(f'./save_dir/{args.n_encoder_layer}_{args.n_decoder_layer}_{args.n_head}_{args.n_query}_{args.hidden_dim}', split)
    results_save_path = os.path.join('./save_dir/'+args.dataset+'/'+args.task+'/results/transformer', 'split'+split,
                                    args.input_type )
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)


    model_save_file = os.path.join(model_save_path, 'checkpoint.ckpt')
    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(),args.lr,weight_decay=args.weight_decay) # 在使用BayesLinear时使用
    warmup_epochs = args.warmup_epochs
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs)
    criterion = nn.MSELoss(reduction = 'none')
    
    bestMoc_path = './bestMoc'
    bestMoc_file = os.path.join(bestMoc_path,f'{args.n_encoder_layer}_{args.n_decoder_layer}_{args.n_head}_{args.n_query}_{args.hidden_dim}_split{args.split}.pkl')
    if not os.path.exists(bestMoc_path):
        os.makedirs(bestMoc_path)
    if not os.path.exists(bestMoc_file) or args.checkpoint==50 :
        bestMoc = dict()
        bestMoc[f'{model_save_path}'] = {}
        bestMoc[f'{model_save_path}'][f'{args.split}']={}
        bestMoc[f'{model_save_path}'][f'{args.split}']['0.2']=[0,0,0,0]
        bestMoc[f'{model_save_path}'][f'{args.split}']['0.3']=[0,0,0,0]
        bestMoc[f'{model_save_path}'][f'{args.split}']['bestCheck']=[30]
        with open(bestMoc_file, 'wb') as file:
            pickle.dump(bestMoc, file)

    if args.predict :
        obs_perc = [0.2, 0.3]

        with open(bestMoc_file, 'rb') as file:
            LastbestMoC = pickle.load(file)

        results_save_path = results_save_path +'/runs'+ str(args.runs) +'.txt'
        if args.dataset == 'breakfast' :
            model_path = f'./save_dir1/{args.n_encoder_layer}_{args.n_decoder_layer}_{args.n_head}_{args.n_query}_{args.hidden_dim}/'+args.split+'/checkpoint'+args.checkpoint+'.ckpt'
        elif args.dataset == '50salads':
            model_path = f'./save_dir1/{args.n_encoder_layer}_{args.n_decoder_layer}_{args.n_head}_{args.n_query}_{args.hidden_dim}/'+args.split+'/checkpoint'+args.checkpoint+'.ckpt'

        print("Predict with ", model_path)

        flag = 0
        MOC = dict()
        for obs_p in obs_perc :
            total = 0
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            total,mocs,ckpoint = predict(LastbestMoC[f'{model_save_path}'][f'{args.split}'],total,model, video_test_list, args, obs_p, n_class, actions_dict, device)
            MOC[f'{obs_p}']=mocs
            if total >= 3:
                flag +=1
            print(total)
        if flag == 2:
            MOC['bestCheck']=ckpoint
            LastbestMoC[f'{model_save_path}'][f'{args.split}'] = MOC
            with open(bestMoc_file, 'wb') as file:
                pickle.dump(LastbestMoC, file)
            print("save best checkpoint")
    else :
        # Training
        trainset = BaseDataset(video_list, actions_dict, features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, \
                                                    shuffle=True, num_workers=args.workers,
                                                    collate_fn=trainset.my_collate)
        train(args, model, train_loader, optimizer, scheduler, criterion,
                    model_save_path, pad_idx, device )


if __name__ == '__main__':
    main()
