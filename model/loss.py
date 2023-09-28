import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        """
        初始化 Focal Loss

        Args:
            alpha (float): 平衡因子 (default: 0.25).
            gamma (float): 聚焦参数 (default: 2).
            logits (bool): 输入是否是 logits（default: False）.
            reduce (bool): 是否应用 reduce 操作（default: True）.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets, weight=None):
        """
        计算 Focal Loss

        Args:
            inputs (torch.Tensor): 模型的输出.
            targets (torch.Tensor): 真实标签.
            weight (torch.Tensor): 权重 (optional).

        Returns:
            torch.Tensor: Focal Loss.
        """
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=weight)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none', weight=weight)
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

if __name__ == '__main__':
    # 示例用法
    loss_fn = FocalLoss()
    inputs = torch.randn(4, requires_grad=True)
    targets = torch.empty(4).random_(2)
    loss = loss_fn(inputs, targets)
    print(loss)
