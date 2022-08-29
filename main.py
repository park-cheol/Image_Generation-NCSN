import argparse
import datetime
import random
import time
import warnings

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from data.dataset import *
from models.scorenet import ScoreNet
from utils.losses import *
from utils.utils import *
from utils.langevin_dynamcis import *

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=500000, help='Number of max epochs in training (default: 100)')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--workers', type=int, default=16, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size in training (default: 32)')
parser.add_argument('--lr', default=1e-3)

parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--num_classes', type=int, default=10)

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist', 'celeba'])
parser.add_argument('--image-size', type=int, default=32)
parser.add_argument('--logit-transform', default=False)
parser.add_argument('--random-flip', default=True)

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--evaluate', '-e', default=False, action='store_true')

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    summary = SummaryWriter()

    # STFT 인자

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # A: Noisy Sample / B: Clean_sample
    # generator = Generator(args=args)
    scorenet = ScoreNet(args=args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            print("Distributed")
            torch.cuda.set_device(args.gpu)
            scorenet.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            scorenet = torch.nn.parallel.DistributedDataParallel(scorenet, device_ids=[args.gpu])

        else:
            scorenet.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            generator = torch.nn.parallel.DistributedDataParallel(scorenet)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        generator = scorenet.cuda(args.gpu)

    else:
        generator = torch.nn.DataParallel(scorenet).cuda()

    # Optimizer / criterion(wSDR)
    criterion = anneal_dsm_score_estimation
    optimizer = torch.optim.Adam(scorenet.parameters(),
                                 lr=args.lr,
                                 weight_decay=0.000,
                                 betas=(0.9, 0.999),
                                 amsgrad=False)
    # generator_scheduler = get_scheduler(generator_optimizer, args)

    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            scorenet.load_state_dict(checkpoint['scorenet'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 파일 리스트
    if not args.dataset == 'celeba':
        if args.random_flip is False:
            train_transform = test_transform = transforms.Compose([transforms.Resize(args.image_size),
                                                                   transforms.ToTensor()])
        else:
            train_transform = transforms.Compose([transforms.Resize(args.image_size),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.Resize(args.image_size),
                                                 transforms.ToTensor()])
    else:
        if args.random_filp is False:
            train_transform = transforms.Compose([transforms.CenterCrop(140),
                                                  transforms.Resize(args.image_size),
                                                  transforms.ToTensor()])
        else:
            train_transform = transforms.Compose([transforms.CenterCrop(140),
                                                  transforms.Resize(args.image_size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.CenterCrop(140),
                                             transforms.Resize(args.image_size),
                                             transforms.ToTensor()])

    train_dataset, test_dataset = get_dataset(dataset_name=args.dataset,
                                              train_transform=train_transform,
                                              test_transform=test_transform)
    # Sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        print("Sampler Use")
    else:
        train_sampler = None

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=True,
                                              drop_last=True)
    test_iter = iter(test_loader)

    # Train
    param = sum(p.numel() for p in scorenet.parameters() if p.requires_grad)
    print("Total Param: ", param)

    # sigmas = torch.tensor(
    #     np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
    #                        self.config.model.num_classes))).float().to(self.config.device)
    step = 0
    sigmas = torch.tensor(np.exp(np.linspace(np.log(1), np.log(0.01), 10))).float().cuda(args.gpu)

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, test_loader, test_iter, epoch, scorenet,
              criterion, optimizer, args, summary, step, sigmas)


def train(train_loader, test_loader, test_iter, epoch, scorenet,
          criterion, optimizer,
          args, summary, step, sigmas):
    end = time.time()

    for i, (x, y) in enumerate(train_loader):
        scorenet.train()
        step += 1

        x = x.cuda(args.gpu, non_blocking=True)
        x = x / 256. * 255. + torch.rand_like(x) / 256.
        if args.logit_transform:
            x = logit_transform(x)

        labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)

        loss = criterion(scorenet=scorenet,
                         samples=x,
                         labels=labels,
                         sigmas=sigmas,
                         anneal_power=2.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        niter = epoch * len(train_loader) + i
        if args.gpu == 0:
            summary.add_scalar('Train/loss', loss.item(), niter)

        if niter == 200001:  # End
            return 0

        if niter % 100 == 0:
            scorenet.eval()
            try:
                test_X, test_y = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_X, test_y = next(test_iter)

            test_X = test_X.cuda(args.gpu, non_blocking=True)
            test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
            if args.logit_transform:
                test_X = logit_transform(test_X)
            test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

            with torch.no_grad():
                test_loss = criterion(scorenet=scorenet,
                                      samples=test_X,
                                      labels=test_labels,
                                      sigmas=sigmas,
                                      anneal_power=2.0)

            if args.gpu == 0:
                summary.add_scalar('Test/loss', test_loss.item(), niter)
        if niter % 2000 == 0:
            os.makedirs('outputs/{}'.format(niter), exist_ok=True)
            sampling(args=args, scorenet=scorenet, niter=niter)
            torch.save({
                'epoch': epoch + 1,
                'scorenet': scorenet.state_dict(),
                'optimizer': optimizer.state_dict()}, "saved_models/checkpoint_%d.pth" % niter)

    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)


if __name__ == "__main__":
    main()
