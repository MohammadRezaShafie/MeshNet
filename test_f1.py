import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import ModelNet40
from models import MeshNet
from utils.retrival import append_feature, calculate_map
from sklearn.metrics import f1_score  # Import the F1 score metric


cfg = get_test_config()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']

data_set = ModelNet40(cfg=cfg['dataset'], part='test')
data_loader = data.DataLoader(data_set, batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=False)

def test_model(model):
    correct_num = 0
    ft_all, lbl_all = None, None

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader):
            # centers = centers
            # corners = corners
            # normals = normals
            # neighbor_index = neighbor_index
            # targets = targets
            centers = centers.cuda()
            corners = corners.cuda()
            normals = normals.cuda()
            neighbor_index = neighbor_index.cuda()
            targets = targets.cuda()

            targets = targets.view(-1, 1)

            outputs, feas = model(centers, corners, normals, neighbor_index)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_num += (preds == targets).float().sum()

            # Collect all predictions and targets for F1 score calculation
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            if cfg['retrieval_on']:
                ft_all = append_feature(ft_all, feas.detach().cpu())
                lbl_all = append_feature(lbl_all, targets.detach().cpu(), flaten=True)

    # Calculate and print accuracy
    print('Accuracy: {:.4f}'.format(float(correct_num) / len(data_set)))

    # Concatenate all predictions and targets for F1 score calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate and print F1 score
    f1 = f1_score(all_targets, all_preds, average='binary')
    print('F1 Score: {:.4f}'.format(f1))

    if cfg['retrieval_on']:
        print('mAP: {:.4f}'.format(calculate_map(ft_all, lbl_all)))


if __name__ == '__main__':
    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load(cfg['load_model']))
    # model.load_state_dict(torch.load(cfg['load_model'], map_location=torch.device('cpu')))
    model.eval()

    test_model(model)
