import numpy as np

from torch import nn
import torch
from torch.nn import functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(SoftDiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, logits, targets):
        logits = torch.softmax(logits, dim=1)
        num = targets.size(0)
        smooth = 1e-5

        m1 = logits[:, 1, :, :, :].view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class focal_loss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2, num_classes=2, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, np.shape(preds)[-1])
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_softmax = preds_softmax.clamp(min=0.0001, max=1.0)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def evaluate(logits, targets):
    smooth = 1
    linear_logit, logit = logits
    ## ACC
    preds = torch.argmax(logit, dim=1)
    accuracy = float(torch.eq(preds, targets).sum().cpu()) / linear_logit.shape[0]

    ## RECALL
    m1 = preds.view(targets.size(0), -1)
    m2 = targets.view(targets.size(0), -1)
    intersection = (m1 * m2)
    # recall = (intersection.sum() + smooth) / (m2.sum() + smooth)
    recall_tensor = (intersection.sum(1) + smooth) / (m2.sum(1) + smooth)
    recall = recall_tensor.sum() / targets.size(0)

    ## PEC
    # precision = (intersection.sum() + smooth) / (m1.sum() + smooth)
    precision_tensor = (intersection.sum(1) + smooth) / (m1.sum(1) + smooth)
    precision = precision_tensor.sum() / targets.size(0)

    ## F1
    F1 = (2. * precision * recall) / (precision + recall)

    ## Jaccard
    # Jaccard = (intersection.sum() + smooth) / (m1.sum() + m2.sum() - intersection.sum() + smooth)
    Jaccard_tensor = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
    Jaccard = Jaccard_tensor.sum() / targets.size(0)

    ## num_pos
    preds_result = preds.cuda().data.cpu().numpy()
    num_pos = np.shape(np.argwhere(preds_result == 1))[0]
    
    return [accuracy, num_pos, recall, precision, F1, Jaccard], preds


def build_metrics(model, batch, device, loss="DICE_CEL", alpha=0.2, gamma=2):
    image_ids, images, labels = batch
    labels = labels.to(device)
    logits = model(images.to(device))
    #linear_logit = (B * D * H * W , num_class)
    #logit = (B, num_class, D, H, W)
    linear_logit, logit = logits

    metrics, preds = evaluate(logits, labels)

    if loss == "CEL":
        CEL = nn.CrossEntropyLoss(ignore_index=255).to(device)
        loss_CEL = CEL(logit, labels)
        loss_seg = loss_CEL
        return (loss_seg, loss_CEL), preds, metrics
    elif loss == "DICE":
        SDL = SoftDiceLoss(n_classes=2).to(device)
        loss_SDL = SDL(logit, labels)
        loss_seg = loss_SDL
        return (loss_seg, loss_SDL), preds, metrics
    elif loss == "FL":
        FL = focal_loss(alpha=0.4, gamma=1.2)
        loss_FL = FL(linear_logit, labels)
        loss_seg = loss_FL
        return (loss_seg, loss_FL), preds, metrics
    elif loss == "DICE_CEL":
        CEL = nn.CrossEntropyLoss(ignore_index=255).to(device)
        SDL = SoftDiceLoss(n_classes=2).to(device)
        loss_CEL = CEL(logit, labels)
        loss_SDL = SDL(logit, labels)
        loss_seg = 0.9 * loss_CEL + 0.1 * loss_SDL
        return (loss_seg, loss_CEL, loss_SDL), preds, metrics
    
    print("Error: build_metrics")
    return preds, metrics


if __name__ == "__main__":
    mat1 = torch.tensor([[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], ])
    mat2 = torch.tensor([[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], ])
    FL = focal_loss().to('cpu')
    loss_seg = FL(mat1, mat2)
    # logits_SDL = mat1[:, 1] / mat1.sum(dim=1)
    print(loss_seg)
