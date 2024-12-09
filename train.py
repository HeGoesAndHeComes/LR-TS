import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import numpy as np
from utils import cal_performance, normalize_duration
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import math
from einops import rearrange

def train(args, model, train_loader, optimizer, scheduler, criterion,  model_save_path, pad_idx, device):
    model.to(device)
    model.train()
    print("Training Start")
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>> [ model : {args.n_encoder_layer}_{args.n_decoder_layer}_{args.n_head}_{args.n_query}_{args.hidden_dim} : train start ] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    for epoch in range(args.epochs):
        print("==================================== Epoch [ ", (epoch+1), '/', args.epochs, ' split : ',args.split,' ] ====================================' )
        epoch_acc =0
        epoch_loss = 0
        epoch_loss_class = 0
        epoch_loss_dur = 0
        epoch_loss_seg = 0
        total_class = 0
        total_class_correct = 0
        total_seg = 0
        total_seg_correct = 0
        for i, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            features, past_label, trans_dur_future, trans_future_target = data
            features = features.to(device) #[B, S, C]
            past_label = past_label.to(device) #[B, S]
            trans_dur_future = trans_dur_future.to(device)
            trans_future_target = trans_future_target.to(device)
            trans_dur_future_mask = (trans_dur_future != pad_idx).long().to(device)

            B = trans_dur_future.size(0)
            target_dur = trans_dur_future*trans_dur_future_mask
            target = trans_future_target
            if args.input_type == 'i3d_transcript':
                inputs = (features, past_label)
            elif args.input_type == 'gt':
                gt_features = past_label.int()
                inputs = (gt_features, past_label)

            outputs = model(inputs)
            losses = 0
            if args.seg :
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                #output_seg = output_seg.view(-1, C).to(device)
                output_seg = rearrange(output_seg," B T C -> (B T) C").to(device)
                target_past_label = past_label.view(-1)
                loss_seg, n_seg_correct, n_seg_total = cal_performance(output_seg, target_past_label, pad_idx)
                losses += loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                epoch_loss_seg += loss_seg.item()
            if args.anticipate :
                output = outputs['action']
                B, T, C = output.size()
                # output = output.view(-1, C).to(device)
                output = rearrange(output," B T C -> (B T) C").to(device)
                target = target.contiguous().view(-1)
                out = output.max(1)[1] #oneshot
                out = out.view(B, -1)
                loss, n_correct, n_total = cal_performance(output, target, pad_idx)
                acc = n_correct / n_total
                loss_class = loss.item()
                losses += loss
                total_class += n_total
                total_class_correct += n_correct
                epoch_loss_class += loss_class

                output_dur = outputs['duration']
                output_dur = normalize_duration(output_dur, trans_dur_future_mask)
                target_dur = target_dur * trans_dur_future_mask
                loss_dur = torch.sum(criterion(output_dur, target_dur)) / \
                torch.sum(trans_dur_future_mask)

                losses += loss_dur
                epoch_loss_dur += loss_dur.item()


            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()


        epoch_loss = epoch_loss / (i+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss)
        if args.anticipate :
            accuracy = total_class_correct/total_class
            epoch_loss_class = epoch_loss_class / (i+1)
            print('Training Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )
            if args.task == 'long' :
                epoch_loss_dur = epoch_loss_dur / (i+1)
                print('dur loss: %.5f'%epoch_loss_dur)

        if args.seg :
            acc_seg = total_seg_correct / total_seg
            epoch_loss_seg = epoch_loss_seg / (i+1)
            print('seg loss :%.3f'%epoch_loss_seg, ', seg acc : %.5f'%acc_seg)

        scheduler.step()

        save_path = os.path.join(model_save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch >=20 :
            save_file = os.path.join(save_path, 'checkpoint'+str(epoch)+'.ckpt')
            torch.save(model.state_dict(), save_file)
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>> [ model : {args.n_encoder_layer}_{args.n_decoder_layer}_{args.n_head}_{args.n_query}_{args.hidden_dim} : train end ] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    return model

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            cos_val = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            return [(base_lr - self.eta_min) * cos_val + self.eta_min for base_lr in self.base_lrs]
