import argparse
import random
import shutil
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

sys.path.append('../..')
import dalib.vision.models.segmentation as models
import dalib.vision.datasets.segmentation as datasets
import dalib.vision.transforms.segmentation as T
from dalib.vision.transforms import DeNormalizeAndTranspose
from dalib.utils.data import ForeverDataIterator
from dalib.utils.metric import ConfusionMatrix
from dalib.utils.meter import AverageMeter, ProgressMeter, Meter
from dalib.utils.logger import CompleteLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(
        root=args.source_root,
        transforms=T.Compose([
            T.RandomResizedCrop(size=args.train_size, ratio=args.resize_ratio, scale=(0.5, 1.)),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomHorizontalFlip(),
            T.NormalizeAndTranspose(),
        ]),
    )
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    target_dataset = datasets.__dict__[args.target]
    train_target_dataset = target_dataset(
        root=args.target_root,
        transforms=T.Compose([
            T.Resize(image_size=args.train_size, label_size=args.train_size),
            T.NormalizeAndTranspose(),
        ])
    )
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_target_dataset = target_dataset(
        root=args.target_root, split='val',
        transforms=T.Compose([
            T.Resize(image_size=args.test_input_size, label_size=args.test_output_size),
            T.NormalizeAndTranspose(),
        ]),
    )
    val_target_loader = DataLoader(val_target_dataset, batch_size=1, shuffle=False, pin_memory=True)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    num_classes = train_source_dataset.num_classes
    model = models.__dict__[args.arch](num_classes=num_classes).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(model.get_parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer,
                            lambda x: args.lr * (1. - float(x) / args.epochs / args.iters_per_epoch) ** (args.lr_power))

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    # define loss function (criterion)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).to(device)
    interp_train = nn.Upsample(size=args.train_size[::-1], mode='bilinear', align_corners=True)
    interp_val = nn.Upsample(size=args.test_output_size[::-1], mode='bilinear', align_corners=True)

    # define visualization function
    decode = train_source_dataset.decode_target

    def visualize(image, pred, label, prefix):
        """
        Args:
            image (tensor): 3 x H x W
            pred (tensor): C x H x W
            label (tensor): H x W
            prefix: prefix of the saving image
        """
        image = image.detach().cpu().numpy()
        pred = pred.detach().max(dim=0)[1].cpu().numpy()
        label = label.cpu().numpy()
        for tensor, name in [
            (Image.fromarray(np.uint8(DeNormalizeAndTranspose()(image))), "image"),
            (decode(label), "label"),
            (decode(pred), "pred")
        ]:
            tensor.save(logger.get_image_path("{}_{}.png".format(prefix, name)))

    if args.phase == 'test':
        confmat = validate(val_target_loader, model, interp_val, criterion, visualize, args)
        print(confmat)
        return

    # start training
    best_iou = 0.
    for epoch in range(args.start_epoch, args.epochs):
        stage = epoch / args.epochs
        logger.set_epoch(epoch)
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_source_iter, train_target_iter, model, interp_train, criterion, optimizer,
              lr_scheduler, epoch, visualize if args.debug else None, args, stage)

        # evaluate on validation set
        confmat = validate(val_target_loader, model, interp_val, criterion, None, args)
        print(confmat.format(train_source_dataset.classes))
        acc_global, acc, iu = confmat.compute()

        # calculate the mean iou over partial classes
        indexes = [train_source_dataset.classes.index(name) for name
                   in train_source_dataset.EVALUATE_CLASSES]
        iu = iu[indexes]
        mean_iou = iu.mean()

        # remember best acc@1 and save checkpoint
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }, logger.get_checkpoint_path(epoch)
        )
        if mean_iou > best_iou:
            shutil.copy(logger.get_checkpoint_path(epoch), logger.get_checkpoint_path('best'))
        best_iou = max(best_iou, mean_iou)
        print("Target: {} Best: {}".format(mean_iou, best_iou))

    logger.close()


conf_conv = nn.Conv2d(1, 1, (9, 9), padding=4, padding_mode='replicate').cuda()
conf_conv.requires_grad_(False)
nn.init.constant_(conf_conv.weight, 1)


def get_pseudo_confs(ori_pred_t: torch.Tensor):  # shape: 2 * 19 * 512 * 1024
    conf_t, pseudo_label_t = ori_pred_t.max(dim=1, keepdim=True)
    nconf_t = conf_conv(conf_t)

    nconf_t /= nconf_t.sum()
    conf_t /= conf_t.sum()
    conf_t += nconf_t

    return conf_t.cpu().squeeze(), pseudo_label_t.cpu().squeeze()


def get_pseudo_examples(npred_t: torch.Tensor, nconf_t: torch.Tensor, npseudo_label_t: torch.Tensor, threshold, stage): # shape: 2 * 19 * 512 * 1024
    nconf_t = nconf_t.contiguous().view(-1)
    npseudo_label_t = npseudo_label_t.long().contiguous().view(-1)

    nconf_t, nidx = torch.sort(nconf_t, descending=True)
    l = len(npseudo_label_t)

    if stage < 0.5:
        threshold /= 2
    nidx = nidx[:int(threshold * l)]

    return npred_t.permute(0, 2, 3, 1).contiguous().view(-1, 19)[nidx], npseudo_label_t[nidx]


def additional_transform(input, conf, target):
    i, j, h, w = transforms.RandomResizedCrop.get_params(input, scale=[0.5, 1.], ratio=[2., 2.])
    input = F.crop(input, i, j, h, w)
    input = F.resize(input, size=args.train_size)
    conf = F.crop(conf, i, j, h, w)
    conf = F.resize(conf, size=args.train_size, interpolation=Image.NEAREST)
    target = F.crop(target, i, j, h, w)
    target = F.resize(target, size=args.train_size, interpolation=Image.NEAREST)
    if random.random() > 0.5:
        input = F.hflip(input)
        conf = F.hflip(conf)
        target = F.hflip(target)
    return input, conf, target

def get_bal_examples(scores, target, stage):
    if stage < 0.5:
        return scores, target

    if len(scores.shape) == 4:
        scores = scores.permute(0, 2, 3, 1).contiguous().view(-1, 19)
        target = target.view(-1)

    total = 2 * 512 * 1024
    target_distribution = []
    for i in range(19):
        idx = torch.where(target == i)[0].cpu().numpy()
        np.random.shuffle(idx)
        target_distribution.append(idx)

    valid_classes = []
    idxs = [0] * 19
    lens = [0] * 19
    for i in range(19):
        l = target_distribution[i].size
        if l:
            valid_classes.append(i)
            lens[i] = l

    nidx = [0] * total
    class_choices = np.array(valid_classes).repeat(total // len(valid_classes) + 1)
    np.random.shuffle(class_choices)
    for i in range(total):
        class_id = class_choices[i]
        nidx[i] = target_distribution[class_id][idxs[class_id]]
        idxs[class_id] = (idxs[class_id] + 1) % lens[class_id]

    return scores[nidx], target[nidx]


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model, interp, criterion,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, visualize, args: argparse.Namespace, stage):
    threshold = args.threshold
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_s = AverageMeter('Loss (s)', ':3.2f')
    losses_t = AverageMeter('Loss (selftraining)', ':3.2f')
    accuracies_s = Meter('Acc (s)', ':3.2f')
    accuracies_t = Meter('Acc (t)', ':3.2f')
    iou_s = Meter('IoU (s)', ':3.2f')
    iou_t = Meter('IoU (t)', ':3.2f')

    confmat_s = ConfusionMatrix(model.num_classes)
    confmat_t = ConfusionMatrix(model.num_classes)
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_s, losses_t, accuracies_s, accuracies_t, iou_s, iou_t],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i in range(args.iters_per_epoch):
        x_s, label_s = next(train_source_iter)
        x_t, label_t = next(train_target_iter)

        x_s = x_s.to(device)
        label_s = label_s.long().to(device)
        x_t = x_t.to(device)
        ori_label_t = label_t.long().to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # Step 1: Train the segmentation network, freeze the discriminator
        y_s = model(x_s)
        pred_s = interp(y_s)
        # pred_s, label_s = get_bal_examples(pred_s, label_s, stage)
        loss_cls_s = criterion(pred_s, label_s)
        loss_cls_s.backward()

        # adversarial training to fool the discriminator
        with torch.no_grad():
            y_t = model(x_t)
            pred_t = interp(y_t)
        conf_t, pseudo_label_t = get_pseudo_confs(pred_t)
        nx_t, nconf_t, npseudo_label_t = additional_transform(x_t, conf_t, pseudo_label_t)
        ny_t = model(nx_t)
        npred_t = interp(ny_t)
        npred_t, nlabel_t = get_pseudo_examples(npred_t, nconf_t, npseudo_label_t, threshold, stage)
        loss_cls_t = criterion(npred_t, nlabel_t.to(device))
        loss_cls_t.backward()

        # compute gradient and do SGD step
        optimizer.step()
        lr_scheduler.step()

        # measure accuracy and record loss
        losses_s.update(loss_cls_s.item(), x_s.size(0))
        losses_t.update(loss_cls_t.item(), x_t.size(0))
        confmat_s.update(label_s.flatten(), pred_s.argmax(1).flatten())
        confmat_t.update(ori_label_t.flatten(), pred_t.argmax(1).flatten())
        acc_global_s, acc_s, iu_s = confmat_s.compute()
        acc_global_t, acc_t, iu_t = confmat_t.compute()
        accuracies_s.update(acc_s.mean().item())
        accuracies_t.update(acc_t.mean().item())
        iou_s.update(iu_s.mean().item())
        iou_t.update(iu_t.mean().item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if visualize is not None:
                visualize(x_s[0], pred_s[0], label_s[0], "source_{}".format(i))
                visualize(x_t[0], pred_t[0], ori_label_t[0], "target_{}".format(i))


def validate(val_loader: DataLoader, model, interp, criterion, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = Meter('Acc', ':3.2f')
    iou = Meter('IoU', ':3.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc, iou],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    confmat = ConfusionMatrix(model.num_classes)

    with torch.no_grad():
        end = time.time()
        for i, (x, label) in enumerate(val_loader):
            x = x.to(device)
            label = label.long().to(device)

            # compute output
            output = interp(model(x))
            loss = criterion(output, label)

            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))
            confmat.update(label.flatten(), output.argmax(1).flatten())
            acc_global, accs, iu = confmat.compute()
            acc.update(accs.mean().item())
            iou.update(iu.mean().item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

                if visualize is not None:
                    visualize(x[0], output[0], label[0], "val_{}".format(i))

    return confmat


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='ADVENT for Segmentation Domain Adaptation')
    # dataset parameters
    parser.add_argument('source_root', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--resize-ratio', nargs='+', type=float, default=(1.5, 8 / 3.),
                        help='the resize ratio for the random resize crop')
    parser.add_argument('--train-size', nargs='+', type=int, default=(1024, 512),
                        help='the input and output image size during training')
    parser.add_argument('--test-input-size', nargs='+', type=int, default=(1024, 512),
                        help='the input image size during test')
    parser.add_argument('--test-output-size', nargs='+', type=int, default=(2048, 1024),
                        help='the output image size during test')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='deeplabv2_resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: deeplabv2_resnet101)')
    parser.add_argument("--resume", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument('--trade-off', type=float, default=0.001,
                        help='trade-off parameter for the advent loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N',
                        help='mini-batch size (default: 2)')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--lr-power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate (only for deeplab).")
    parser.add_argument("--lr-d", default=1e-4, type=float,
                        metavar='LR', help='initial learning rate for discriminator')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="The threshold for pseudo example selecting.")
    parser.add_argument("--log", type=str, default='rebalselftraining',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--debug', action="store_true",
                        help='In the debug mode, save images and predictions during training')
    args = parser.parse_args()
    print(args)
    main(args)
