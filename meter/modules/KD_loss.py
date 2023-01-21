import torch
import math
from torch import nn
import torch.nn.functional as F

def distillation_loss(y, labels, teacher_scores, T, alpha, hard_negative,step,flag=False,reduction_kd='mean', reduction_nll='mean'):
    #if teacher_scores is not None and y.dtype != teacher_scores.dtype:
    #    teacher_scores = teacher_scores.half()
    # T how much to rely on the teacher’s soft predictions.
    # print("y",y)
    # print("teacher",teacher_scores)
    # print("labels",labels)
    t_weight=None
    if teacher_scores is not None:

        # print(F.softmax(teacher_scores / T, dim=1))
        # d_loss = nn.KLDivLoss(reduction=reduction_kd)(F.log_softmax(y / T, dim=1),
        #                                               F.softmax(teacher_scores / T, dim=1)) * T * T
        # print(d_loss)
        # d_loss=F.mse_loss(teacher_scores, y)



        t_probs = F.softmax(teacher_scores, dim=1)
        t_entropy = torch.sum(t_probs * torch.log(t_probs), dim=1)
        # norm
        t_entropy=t_entropy / torch.sum(t_probs * torch.log(t_probs))

        # re-weight
        r_t_weight =(1- t_entropy)
        r_t_weight = r_t_weight ** 2
        tmp_r_t_weight=r_t_weight
        r_t_weight=F.softmax(r_t_weight,dim=0)
        # r_t_weight=r_t_weight/torch.sum(r_t_weight)
        t_weight=t_entropy
        # t_weight = (1 - t_weight) ** 0.5
        # t_weight=t_weight/torch.sum(t_weight)
        # t_weight=F.softmax(t_weight,dim=0)


        d_loss=T*T*torch.sum(F.mse_loss(teacher_scores/T, y/T)*r_t_weight)

        # d_loss=T*T*F.mse_loss(teacher_scores/T, y/T)
        # ce loss
        nll_loss = torch.tensor(0).to(y)
        # nll_loss1= F.cross_entropy(y, labels)
        label = torch.zeros(hard_negative+1).to(y).float()
        label[0]=1
        for i in range(0,len(labels)):
            nll_loss+=F.binary_cross_entropy_with_logits(y[i],label)*t_weight[i]

        # print(nll_loss)
    else:
        assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
        d_loss = 0.0

        nll_loss = F.cross_entropy(y, labels)
    # else:
    # scale
    # print(d_loss.shape, d_loss)
    # print('\n', nll_loss.shape, nll_loss)
    # print("dloss",d_loss)
    # print("nllloss",nll_loss)
    # tol_loss = (nll_loss/d_loss)*alpha * d_loss + (d_loss/nll_loss)*(1.0 - alpha) * nll_loss
    # print(teacher_scores)
    # print(y)
    #
    # nll_loss=0
    # alpha=1
    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss


    # print('in func:', d_loss.item(), nll_loss.item(), alpha, tol_loss.item())
    return tol_loss, d_loss, nll_loss

def WSLDistiller(y, labels, teacher_scores, T, alpha, hard_negative,step,flag=False,reduction_kd='mean', reduction_nll='mean'):
    t_weight=None

    if teacher_scores is not None:
        t_probs = F.softmax(teacher_scores, dim=1)
        t_entropy = torch.sum(t_probs * torch.log(t_probs), dim=1)
        # norm
        t_entropy = t_entropy / torch.sum(t_probs * torch.log(t_probs))
        t_weight=t_entropy

        s_input_for_softmax = y / T
        t_input_for_softmax = teacher_scores / T


        fc_s_auto = y.detach()
        fc_t_auto = teacher_scores.detach()

        log_softmax_s = F.log_softmax(fc_s_auto)
        log_softmax_t = F.log_softmax(fc_t_auto)

        label = torch.zeros(y.size(0),hard_negative+1).to(y).float()
        for i in range(0,y.size(0)):
            label[i][0]=1

        softmax_loss_s = - torch.sum(label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).to(y)
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)

        if step % 20 ==0 and flag==True:
            print(teacher_scores)
            print(y)
            print(softmax_loss_t)
            print(softmax_loss_s)
            print(focal_weight)
        # mseloss_none=F.mse_loss(t_input_for_softmax, s_input_for_softmax,reduction='none')
        # mseloss_batch = torch.sum(mseloss_none, dim=1, keepdim=True)
        # softmax_loss = focal_weight * mseloss_batch
        softmax_loss = focal_weight * F.mse_loss(t_input_for_softmax,s_input_for_softmax)

        d_loss = (T ** 2) * torch.mean(softmax_loss)
        # nll_loss = torch.tensor(0).to(y)
        # nll_loss1= F.cross_entropy(y, labels)
        # label = torch.zeros(hard_negative + 1).to(y).float()
        # label[0] = 1
        # for i in range(0, len(labels)):
        #     nll_loss += F.binary_cross_entropy_with_logits(y[i], label) * t_weight[i]

    else:
        assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
        d_loss = 0.0
    nll_loss = F.cross_entropy(y, labels)

    tol_loss = alpha * d_loss +   nll_loss

    return tol_loss, d_loss, nll_loss
def distillation_vqa_loss(vqa_logits, vqa_targets, teacher_scores, T, alpha, reduction_kd='mean', reduction_nll='mean'):

    if teacher_scores is not None:
        d_loss = nn.KLDivLoss()(F.log_softmax(vqa_logits / T, dim=1),
                                                      F.softmax(teacher_scores / T, dim=1)) * T * T
    else:
        assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
        d_loss = 0.0
    nll_loss =  (
            F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
            * vqa_targets.shape[1]
    )
    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss
    # print('in func:', d_loss.item(), nll_loss.item(), alpha, tol_loss.item())
    return tol_loss, d_loss, nll_loss



def patience_loss(teacher_patience, student_patience, normalized_patience=False):
    # n_batch = teacher_patience.shape[0]
    if normalized_patience:
        teacher_patience = F.normalize(teacher_patience, p=2, dim=2)
        student_patience = F.normalize(student_patience, p=2, dim=2)
    return F.mse_loss(teacher_patience.float(), student_patience.float()).half()

    # diff = (teacher_patience - student_patience).pow(2).sum()
    # const = math.sqrt(teacher_patience.numel())
    # return diff / const / const