import os
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score,confusion_matrix

def get_configs(config_dir="./configs/config.yaml"):
    # Get hyperparameters from yaml
    with open(config_dir, "r") as f:
        configs = yaml.safe_load(f)
    return configs


def save_best_checkpoint(model, best_epoch, model_name):
    """
    Save best model
    Args:
        model (torch.nn.Module): 要保存的模型.
        best_epoch (int): 最佳检查点所在的训练轮次.
        model_name (str): 模型名称.
    """
    save_state = {
        'model': model.state_dict(),
        'best_epoch': best_epoch,
    }
    ckpt_dir = os.path.join('./save_model', model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, 'best_ckpt.pth')
    torch.save(save_state, save_path)

def load_best_result(model, model_name):
    """
    加载模型的最佳检查点和相关信息

    Args:
        model (torch.nn.Module): 要加载检查点的模型.
        model_name (str): 模型名称.

    Returns:
        torch.nn.Module: 加载的模型.
        int: 最佳检查点所在的训练轮次.
    """
    ckpt_dir = os.path.join('./save_model', model_name)
    best_ckpt_path = os.path.join(ckpt_dir, 'best_ckpt.pth')
    ckpt = torch.load(best_ckpt_path)
    
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model = ckpt['model']

    best_epoch = ckpt['best_epoch']
    return model, best_epoch

def print_result(y_pred, y_label):
    train_acc = accuracy_score(y_pred, y_label)
    train_auc = roc_auc_score(y_pred, y_label)
    f1_train = f1_score(y_pred, y_label)
    print("Accuracy: {:.6f}".format(train_acc))
    print("AUC     : {:.6f}".format(train_auc))
    print("F1      : {:.6f}".format(f1_train))