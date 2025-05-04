import sys
import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
from earlystop import earlystop
import numpy as np
#from utils import Logger
import attack_generator as attack
import torch
from models.NetworkCNN import NetworkCNN
from train import train, adjust_learning_rate, adjust_tau, save_checkpoint


parser = argparse.ArgumentParser(description='PyTorch Friendly Adversarial Training')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="networkcnn",
                    help="decide which network to use,choose from networkcnn,resnet18")
parser.add_argument('--tau', type=int, default=0, help='step tau')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")
parser.add_argument('--dynamictau', type=bool, default=True, help='whether to use dynamic tau')
parser.add_argument('--out_dir', type=str, default='./FAT_results', help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')

args = parser.parse_args()

def main():    
    # training settings
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('==> Load Test Data')
    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
 
    print('==> Load Model')
    if args.net == "networkcnn":
        model = NetworkCNN().cuda()
        net = "networkcnn"
    # if args.net == "resnet18":
    #     model = ResNet18().cuda()
    #     net = "resnet18"
    print(net)

    model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 0
    # Resume
    title = 'FAT train'
    if args.resume:
        # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
        print('==> Friendly Adversarial Training Resuming from checkpoint ..')
        print(args.resume)
        assert os.path.isfile(args.resume)
        out_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('==> Friendly Adversarial Training')

    test_nat_acc = 0
    fgsm_acc = 0
    test_pgd20_acc = 0
    cw_acc = 0
    best_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch + 1, args)

        #-----------------train----------------------------
        train_time, train_loss, bp_count_avg = train(model, train_loader, optimizer, adjust_tau(epoch + 1, args.dynamictau, args), args)

        #----------------------evaluate--------------------
        loss, test_nat_acc = attack.eval_clean(model, test_loader)
        loss, fgsm_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,loss_fn="cent", category="Madry",rand_init=True)
        loss, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", rand_init=True)
        loss, cw_acc = attack.eval_robust(model, test_loader, perturb_steps=30, epsilon=0.031, step_size=0.031 / 4,loss_fn="cw", category="Madry", rand_init=True)

        print(
            'Epoch: [%d | %d] | Train Time: %.2f s | BP Average: %.2f | Natural Test Acc %.2f | FGSM Test Acc %.2f | PGD20 Test Acc %.2f | CW Test Acc %.2f |\n' % (
            epoch + 1,
            args.epochs,
            train_time,
            bp_count_avg,
            test_nat_acc,
            fgsm_acc,
            test_pgd20_acc,
            cw_acc)
            )
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'bp_avg': bp_count_avg,
            'test_nat_acc': test_nat_acc,
            'test_pgd20_acc': test_pgd20_acc,
            'optimizer': optimizer.state_dict(),
        })

if __name__ == "__main__":
    sys.exit(int(main() or 0))