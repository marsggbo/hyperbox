import torch


__all__ = [
    'CrossEntropyLabelSmooth',
]


class CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, label_smoothing):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        n_classes = pred.size(1)
        # convert to one-hot
        target = torch.unsqueeze(target, 1)
        soft_target = torch.zeros_like(pred)
        soft_target.scatter_(1, target, 1)
        # label smoothing
        soft_target = soft_target * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))