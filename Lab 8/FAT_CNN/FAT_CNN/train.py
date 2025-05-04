import os
import torch.nn as nn
import datetime
import torch
from earlystop import earlystop

def train(model, train_loader, optimizer, tau, args):
    starttime = datetime.datetime.now()
    loss_sum = 0
    bp_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # Get friendly adversarial training data via early-stopped PGD
        output_adv, output_target, output_natural, count = earlystop(model, data, target, step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps, tau=tau,
          randominit_type="uniform_randominit", loss_fn='cent', rand_init=args.rand_init, omega=args.omega)
        bp_count += count
        model.train()
        optimizer.zero_grad()
        output = model(output_adv)

        # calculate standard adversarial training loss
        loss = nn.CrossEntropyLoss(reduction='mean')(output, output_target)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    bp_count_avg = bp_count / len(train_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds

    return time, loss_sum, bp_count_avg

def adjust_tau(epoch, dynamictau, args):
    tau = args.tau
    if dynamictau:
        if epoch <= 50:
            tau = 0
        elif epoch <= 90:
            tau = 1
        else:
            tau = 2
    return tau


def adjust_learning_rate(optimizer, epoch, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 60:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 110:
        lr = args.lr * 0.005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

