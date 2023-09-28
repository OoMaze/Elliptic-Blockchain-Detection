import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from lib.Elliptic_data import *
from lib.util import *
from model.loss import FocalLoss
from model.model import Net, GCNNet, GATNet
from lib.util import get_configs

config = get_configs(config_dir="./configs/config.yaml")
data, edge_index, sup_idx, unsup_idx = load_dataset(config)
input_Data = get_dataset(data, edge_index)
trn_idx, valid_idx = split_idx(config, input_Data, sup_idx)
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get Train model
model = Net(dim_in=config['MODEL']['INPUT_DIM'], dim_hidden=config['MODEL']['HID_DIM'],
                slices=config['BOOSTER']['SLICES'], num_layer=config['BOOSTER']['NUM_LAYERCES'], f_att=config['BOOSTER']['F_ATT'])
model = model.to(config["device"])
model.float()

best_loss = np.Inf
input_Data = input_Data.to(config["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

config['criterion'] = FocalLoss(alpha=0.25) if config['MODEL']['FL_LOSS'] else torch.nn.BCELoss()

if config['TRAIN']['TRAIN_MODE']:
    
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, config['TRAIN']['MAX_EPOCH']+1):
        # -----------------------------------------------
        # TRAIN MODE
        # -----------------------------------------------
        model.train()
        optimizer.zero_grad()
        out = model(input_Data)
        loss = config['criterion'](out[train_idx].squeeze(), input_Data.y[train_idx])
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.detach().numpy())
        if epoch%5 == 0:
            y_label_cpu = input_Data.y.detach().cpu().numpy()
            y_pred_cpu = out.detach().cpu().numpy()
            auc = roc_auc_score(y_pred_cpu[train_idx], y_pred_cpu[train_idx])
            print("Epoch: {:2d} - Train loss: {:.6f} - roc: {:.6f}".format(epoch, loss.item(), auc))
        # -----------------------------------------------
        # EVAL MODE
        # -----------------------------------------------
        model.eval()
        with torch.no_grad():
            val_loss = config['criterion'](out[valid_idx].squeeze(), input_Data.y[valid_idx])
            val_loss_list.append(val_loss.detach().numpy())
            if epoch%5 == 0:
                auc = roc_auc_score(y_label_cpu[valid_idx], y_pred_cpu[valid_idx])
                print("Epoch: {:2d} - Valid loss: {:.6f} - roc: {:.6f}".format(epoch, val_loss.item(), auc))
        if val_loss <= best_loss:
            best_loss = val_loss
            best_epoch = epoch
            save_best_checkpoint(model, best_epoch, config['MODEL']['MODEL_NAME'])
try:
    best_model, best_epoch = load_best_result(model, config['MODEL']['MODEL_NAME'])
    print('Best Model Load At {}'.format(best_epoch))
except:
    print('Load Model Fail!')
    
preds = best_model(input_Data)
preds = preds.detach().cpu().numpy()

out_labels = preds > 0.5
y_label = input_Data.y.detach().cpu().numpy()
train_pred, valid_pred =  out_labels[train_idx], out_labels[valid_idx]
train_label, valid_label =  y_label[train_idx], y_label[valid_idx]
print('TRAIN SET RESULTS:')
print_result(train_pred,train_label)
print('--------------------------')
print('VALID SET RESULTS:')
print_result(valid_pred,valid_label)
print('--------------------------')
f1_total = f1_score(y_label[sup_idx], out_labels[sup_idx])
print('Total F1 score: {:.6f}'.format( f1_total))
#print(confusion_matrix(input_Data.y.detach().cpu().numpy()[valid_idx], out_labels[valid_idx]))