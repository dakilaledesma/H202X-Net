#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch

from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging
from sklearn.preprocessing import MinMaxScaler

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')


def main():
    inv_freq = np.load('inverse_cf.npy')

    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    loader = create_loader(
        ImageDataset(args.data),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    model.eval()

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    with torch.no_grad():
        _i = 0
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            labels_np = labels.cpu().numpy()

            baug = np.load(f"npzs/{_i}.npz")["a"] * 0.3
            labels_np = np.add(labels_np, baug)

            out_labels = []
            for idx, label in enumerate(labels_np):
                label = np.reshape(label, (-1, 1))
                scaler = MinMaxScaler()
                scaler.fit(label)
                label = scaler.transform(label)
                label = np.reshape(label, (-1))

                label = np.multiply(label, inv_freq)
                out_labels.append((-label).argsort()[:5])

            out_labels = np.array(out_labels)
            # print(np.amax(labels.cpu().numpy()[0]))
            # print(np.mean(labels.cpu().numpy()[0]), np.std(labels.cpu().numpy()[0]))
            # print(type(labels))
            topk_ids.append(out_labels)
            # print(np.argmax(labels.cpu().numpy()[0]), topk.cpu().numpy()[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

            _i += 1

    topk_ids = np.concatenate(topk_ids, axis=0).squeeze()

    with open(os.path.join(args.output_dir, 'submission_files/topk_ids.csv'), 'w') as out_file:
        filenames = loader.dataset.filenames(basename=True)
        for filename, label in zip(filenames, topk_ids):
            out_file.write('{0},{1},{2},{3},{4},{5}\n'.format(
                filename, label[0], label[1], label[2], label[3], label[4]))


if __name__ == '__main__':
    main()