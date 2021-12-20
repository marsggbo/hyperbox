import torch


class MixupLoss(torch.nn.Module):
    def __init__(self, criterion):
        super(MixupLoss, self).__init__()
        self.criterion = criterion
        self.training = False

    def forward(self, logits, y, *args, **kwargs):
        if self.training:
            loss_a = self.criterion(logits, y[:, 0].long(), *args, **kwargs)
            loss_b = self.criterion(logits, y[:, 1].long(), *args, **kwargs)
            return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()
        else:
            return self.criterion(logits, y, *args, **kwargs)

