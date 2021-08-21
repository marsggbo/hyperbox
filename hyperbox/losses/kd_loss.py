import torch.nn.functional as F


def KDLoss(outputs, teacher_outputs, alpha=0.4, temperature=2):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temperature
    student = F.log_softmax(outputs/T, dim=-1)
    teacher = F.softmax(teacher_outputs/T, dim=-1)
    KD_loss = F.kl_div(student, teacher)* (alpha * T * T)

    return KD_loss