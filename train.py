import copy
import os
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np
from config import get_train_config
from data import ModelNet40
from models import MeshNet
from utils.retrival import append_feature, calculate_map
from sklearn.model_selection import StratifiedKFold
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
dataset = ModelNet40(cfg=cfg['dataset'], part='train')
test_set = ModelNet40(cfg=cfg['dataset'], part='test')

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, cfg):

    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)

        for phase, loader in [('train', train_loader), ('val', val_loader)]:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_targets = []

            for centers, corners, normals, neighbor_index, targets in loader:
                centers = centers.cuda()
                corners = corners.cuda()
                normals = normals.cuda()
                neighbor_index = neighbor_index.cuda()
                targets = targets.cuda()

                targets = targets.view(-1, 1)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(centers, corners, normals, neighbor_index)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    loss = criterion(outputs, targets)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item() * centers.size(0)
                    running_corrects += torch.sum(preds == targets.data)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            epoch_f1 = f1_score(all_targets, all_preds)

            if phase == 'train':
                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1))
                scheduler.step()

            if phase == 'val':
                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} (best F1 {:.4f})'.format(phase, epoch_loss, epoch_acc, epoch_f1, best_f1))
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())

                if epoch % cfg['save_steps'] == 0:
                    torch.save(copy.deepcopy(model.state_dict()), os.path.join(cfg['ckpt_root'], '{}.pkl'.format(epoch)))

    print('Best val F1: {:.4f}'.format(best_f1))
    print('Config: {}'.format(cfg))

    return best_model_wts

if __name__ == '__main__':
    # prepare model
    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model = nn.DataParallel(model)

    # criterion
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

    # start training with Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    targets = np.array([dataset[i][4].item() for i in range(len(dataset))])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f'Fold {fold+1}')
        train_sampler = data.SubsetRandomSampler(train_idx)
        val_sampler = data.SubsetRandomSampler(val_idx)

        train_loader = data.DataLoader(dataset, batch_size=cfg['batch_size'], sampler=train_sampler)
        val_loader = data.DataLoader(dataset, batch_size=cfg['batch_size'], sampler=val_sampler)

        best_model_wts = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, cfg)
        torch.save(best_model_wts, os.path.join(cfg['ckpt_root'], f'MeshNet_best_fold{fold+1}.pkl'))

    # Evaluate the model on the test set
    test_loader = data.DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False)
    model.load_state_dict(best_model_wts)
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for centers, corners, normals, neighbor_index, targets in test_loader:
            centers = centers.cuda()
            corners = corners.cuda()
            normals = normals.cuda()
            neighbor_index = neighbor_index.cuda()
            targets = targets.cuda()

            targets = targets.view(-1, 1)
            outputs, _ = model(centers, corners, normals, neighbor_index)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_f1 = f1_score(all_targets, all_preds)
    print(f'Test F1 Score: {test_f1:.4f}')
