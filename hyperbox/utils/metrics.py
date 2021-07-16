import torch


class Accuracy:
    def __call__(self, output, target, topk=(1,)):
        return accuracy(output, target, topk=(1,))


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(torch.tensor(correct_k.mul_(100.0 / batch_size)))
    if len(res) == 1:
        return res[-1]
    return res
