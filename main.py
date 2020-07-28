import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from model import DeepLabV3Plus
from config import load_config
from preprocess import load_data
from utils import SegmentationLosses, calculate_weigths_labels


class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.warmup_iteration = warmup_iteration

        super(PolyLr, self).__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        return base_lr * ((1 - (step / self.max_iteration)) ** (self.gamma))

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch)) for base_lr in
                    self.base_lrs]
        else:
            lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

        return lrs


def train(train_loader, model, optimizer, criterion, scheduler, args):
    model.train()
    global_step = 0
    for batch in train_loader:
        print(global_step)
        img = batch['image']['original_scale']
        label = batch['label']['semantic_logit']

        if args.cuda:
            img, label = img.cuda(), label.cuda()

        logit = model(img)
        # loss = criterion.CrossEntropyLoss(logit, label)
        loss = criterion(logit, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1


def main(args):
    train_loader, val_loader = load_data(args)
    args.weight_labels = torch.tensor(calculate_weigths_labels('cityscape', train_loader, args.n_classes)).float()
    if args.cuda:
        args.weight_labels = args.weight_labels.cuda()

    model = DeepLabV3Plus()
    if args.cuda:
        model = model.cuda()
    # criterion = SegmentationLosses(weight=args.weight_labels, cuda=True)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_mask, weight=args.weight_labels)
    if args.cuda:
        criterion = criterion.cuda()

    backbone_params = nn.ParameterList()
    decoder_params = nn.ParameterList()

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    params_list = [{'params': backbone_params},
                   {'params': decoder_params, 'lr': args.lr * 10}]

    optimizer = optim.SGD(params_list,
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=True)
    scheduler = PolyLr(optimizer, gamma=args.gamma,
                       max_iteration=args.max_iteration,
                       warmup_iteration=args.warmup_iteration)

    train(train_loader, model, optimizer, criterion, scheduler, args)


if __name__ == '__main__':
    args = load_config()
    main(args)
