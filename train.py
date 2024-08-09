import copy
import os
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import math
import numpy as np
from config import get_train_config
from data import ModelNet40
from models import MeshNet
from utils.retrival import append_feature, calculate_map
from sklearn.metrics import f1_score

cfg = get_train_config()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']

# seed
seed = cfg['seed']
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

# dataset
data_set = {
    x: ModelNet40(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
}
data_loader = {
    x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'], shuffle=True)
    for x in ['train', 'test']
}


def train_model(model, criterion, optimizer, scheduler, cfg):

    best_acc = 0.0
    best_f1 = 0.0
    best_map = 0.0
    all_preds = []
    all_targets = []
    best_model_wts_acc = copy.deepcopy(model.state_dict())
    best_model_wts_f1 = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)

        # adjust_learning_rate(cfg, epoch, optimizer)
        for phrase in ['train', 'test']:

            if phrase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            ft_all, lbl_all = None, None

            for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader[phrase]):
                centers = centers.cuda()
                corners = corners.cuda()
                normals = normals.cuda()
                neighbor_index = neighbor_index.cuda()
                targets = targets.cuda()

                targets = targets.view(-1, 1)

                with torch.set_grad_enabled(phrase == 'train'):
                    outputs, feas = model(centers, corners, normals, neighbor_index)
                    _, preds = torch.max(outputs, 1)

                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    loss = criterion(outputs, targets)
                    
                    if phrase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if phrase == 'test' and cfg['retrieval_on']:
                        ft_all = append_feature(ft_all, feas.detach().cpu())
                        lbl_all = append_feature(lbl_all, targets.detach().cpu(), flaten=True)

                    running_loss += loss.item() * centers.size(0)
                    running_corrects += torch.sum(preds == targets.data)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            epoch_loss = running_loss / len(data_set[phrase])
            epoch_acc = running_corrects.double() / len(data_set[phrase])
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            epoch_f1 = f1_score(all_targets, all_preds)

            if phrase == 'train':
                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phrase, epoch_loss, epoch_acc, epoch_f1))
                scheduler.step()

            if phrase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts_acc = copy.deepcopy(model.state_dict())
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts_f1 = copy.deepcopy(model.state_dict())
                print_info = '{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} (best_acc {:.4f}) (best_f1 {:.4f})'.format(phrase, epoch_loss, epoch_acc, epoch_f1, best_acc, best_f1)

                if cfg['retrieval_on']:
                    epoch_map = calculate_map(ft_all, lbl_all)
                    if epoch_map > best_map:
                        best_map = epoch_map
                    print_info += ' mAP: {:.4f}'.format(epoch_map)
                
                if epoch % cfg['save_steps'] == 0:
                    torch.save(copy.deepcopy(model.state_dict()), os.path.join(cfg['ckpt_root'], '{}.pkl'.format(epoch)))

                print(print_info)

    print('Best val acc: {:.4f}'.format(best_acc))
    print('Config: {}'.format(cfg))

    return best_model_wts_acc,best_model_wts_f1


if __name__ == '__main__':

    # prepare model
    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    # model.cuda()
    model = nn.DataParallel(model)

    # criterion
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    # scheduler
    if cfg['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['max_epoch'])

    # start training
    if not os.path.exists(cfg['ckpt_root']):
        os.mkdir(cfg['ckpt_root'])
    best_model_wts_acc, best_model_wts_f1 = train_model(model, criterion, optimizer, scheduler, cfg)
    torch.save(best_model_wts_acc, os.path.join(cfg['ckpt_root'], 'MeshNet_best_acc.pkl'))
    torch.save(best_model_wts_f1, os.path.join(cfg['ckpt_root'], 'MeshNet_best_f1.pkl'))