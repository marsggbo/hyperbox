import torch
import torch.nn as nn
import torch.nn.functional as F


class MutualLoss(nn.Module):
    def __init__(self, criterion=None):
        super(MutualLoss, self).__init__()
        self.criterion = criterion
        if self.criterion is None:
            self.loss_fn = torch.nn.MSELoss()
        elif self.criterion == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif self.criterion == 'kl':
            self.loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, feat1, feat2):
        dim = torch.tensor(feat1.shape[0]+1e-12).sqrt().item()
        if self.criterion == 'kl':
            loss1 = self.loss_fn(feat1.log_softmax(-1), feat2.softmax(-1))
            # loss2 = self.loss_fn(feat2.log_softmax(-1), feat1.softmax(-1))
            loss = loss1
            # loss = loss1 + loss2
            return loss
        elif self.criterion == 'mse':
            feat1 = (feat1-feat1.min()) / (feat1.max()-feat1.min())
            feat2 = (feat2-feat2.min()) / (feat2.max()-feat2.min())
            sim1 = torch.mm(feat1, feat1.T) / dim
            sim2 = torch.mm(feat2, feat2.T) / dim
            # sim1 = F.softmax(torch.mm(feat1, feat1.T) / dim, -1)
            # sim2 = F.softmax(torch.mm(feat2, feat2.T) / dim, -1)
            return self.loss_fn(sim1, sim2)
