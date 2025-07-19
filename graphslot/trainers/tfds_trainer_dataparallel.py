import jax
import numpy as np
import gc
import os 
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Iterable, Optional
import random
import math
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path


from graphslot.datasets.tfds import tfds_input_pipeline
from graphslot.datasets.tfds.tfds_dataset_wrapper import MOViData
import graphslot.modules as modules
from graphslot.modules.factory import build_modules as modules_flow
import graphslot.modules.evaluator

import graphslot.trainers.utils.misc as misc
import graphslot.trainers.utils.lr_sched as lr_sched
import graphslot.trainers.utils.lr_decay as lr_decay

import warnings
warnings.filterwarnings('ignore')

processors_dict = {
    'graphslot': modules.graphslot_build_modules,
    'flow': modules_flow
}

def get_args():
    parser = argparse.ArgumentParser('TFDS dataset training for graphslot.')

    def adrg(name, default, type=str, help=None):
        """ADd aRGuments to parser."""
        if help:
            parser.add_argument(name, default=default, type=type, help=help)
        else:
            parser.add_argument(name, default=default, type=type)

    # Training config
    adrg('--seed', 42, int)
    adrg('--epochs', 50, int)
    adrg('--num_train_steps', 100000, int)
    adrg('--batch_size', 1, int, help='Batch size')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--gpu', default='0,1', type=str, help='GPU id to use.')
    parser.add_argument('--slice_decode_inputs', action='store_true', help="decode in slices.")

    # Adam optimizer config
    adrg('--lr', 2e-4, float)
    adrg('--warmup_steps', 2500, int)
    adrg('--max_grad_norm', 0.05, float)

    # Logging and graphslotng config
    adrg('--log_loss_every_step', 50, int)
    adrg('--eval_every_steps', 1000, int)
    adrg('--checkpoint_every_steps', 5000, int)
    adrg('--exp', 'test', help="experiment name")
    parser.add_argument('--no_snap', action='store_true', help="don't snapshot model")

    # Loading model
    adrg('--resume_from', None, str, help="absolute path of experiment snapshot")

    # Metrics Spec
    adrg('--metrics', 'loss,ari_nobg')

    # Dataset
    adrg('--tfds_name', "movi_c/128x128:1.0.0", help="Dataset for training/eval")
    adrg('--data_dir', "/root/autodl-tmp/GraphSlot-main/graphslot/datasets/")
    
    # Model
    adrg('--max_instances', 10, int, help="Number of slots")  # For Movi-A,B,C, only up to 10. for MOVi-D,E, up to 23.
    adrg('--model_type', 'graphslot', help="model type")
    parser.add_argument('--init_weight', default='default', help='weight init')
    parser.add_argument('--init_bias', default='default', help='bias init')
    parser.add_argument('--mode', type=str, required=True, help='mode')

    # Evaluation
    adrg('--eval_slice_size', 6, int)
    parser.add_argument('--eval', action='store_true', help="Perform evaluation only")

    args = parser.parse_args()
    # Weights
    args.weight_init = {
        'param': args.init_weight,
        'linear_w': args.init_weight,
        'linear_b': args.init_bias,
        'conv_w': args.init_weight,
        'convtranspose_w': "lecun_normal_fan_out" if args.init_weight == 'lecun_normal' else args.init_weight,
        'conv_b': args.init_bias}
    # Training
    args.gpu = [int(i) for i in args.gpu.split(',')]
    # Metrics
    args.train_metrics_spec = {
        v: v for v in args.metrics.split(',')}
    args.eval_metrics_spec = {
        f"eval_{v}": v for v in args.metrics.split(',')}
    # Misc
    args.num_slots = args.max_instances + 1  # only used for metrics
    args.logging_min_n_colors = args.max_instances
    args.shuffle_buffer_size = args.batch_size * 8
    kwargs = {}
    kwargs['slice_decode_inputs'] = True if args.slice_decode_inputs else False
    args.kwargs = kwargs

    # HARDCODED
    args.targets = {"flow": 3}
    args.losses = {f"recon_{target}": {"loss_type": "recon", "key": target}
                   for target in args.targets}

    # Preprocessing
    args.preproc_train = [
        "video_from_tfds",
        f"sparse_to_dense_annotation(max_instances={args.max_instances})",
        "temporal_random_strided_window(length=6)",
        "resize_small(size=128, max_size=128)",
        "crop_or_pad(height=128, width=128, allow_crop=False)", 
        "flow_to_rgb()"
    ]

    args.preproc_eval = [
        "video_from_tfds",
        f"sparse_to_dense_annotation(max_instances={args.max_instances})",
        "temporal_crop_or_pad(length=24)",
        "resize_small(size=128, max_size=128)",
        "crop_or_pad(height=128, width=128, allow_crop=False)", 
        "flow_to_rgb()"
    ]
    return args


def build_datasets(args):
    rng = jax.random.PRNGKey(args.seed)
    train_ds, eval_ds = tfds_input_pipeline.create_datasets(args, rng)

    traindata = MOViData(train_ds)
    evaldata = MOViData(eval_ds)

    return traindata, evaldata


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, global_step, start_time,
                    max_norm: Optional[float] = None, args=None,
                    val_loader=None, evaluator=None):
    model.train(True)

    dataset = data_loader.dataset
    dataset.reset_itr()
    len_data = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)

    if epoch == 0:
        scheduler = lr_sched.get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_train_steps,
                                                             num_cycles=0.5, last_epoch=-1)
    else:
        scheduler = None

    loss = None
    grad_accum = 0

 
    
    for data_iter_step, (video, boxes, segmentations, flow, padding_mask, mask) in enumerate(data_loader):
        
        video = video.squeeze(0).to(device, non_blocking=True)  # [64, 6, 64, 64, 3]
        boxes = boxes.squeeze(0).to(device, non_blocking=True)
        flow = flow.squeeze(0).to(device, non_blocking=True)
        padding_mask = padding_mask.squeeze(0).to(device, non_blocking=True)
        mask = mask.squeeze(0).to(device, non_blocking=True) if len(mask) > 0 else None
        segmentations = segmentations.squeeze(0).to(device, non_blocking=True)
        batch = (video, boxes, segmentations, flow, padding_mask, mask)

        conditioning = boxes

        outputs = model(video=video, conditioning=conditioning,
                        padding_mask=padding_mask)
        itr_loss = criterion(outputs, batch)
        
        if loss == None:
            loss = itr_loss
        del outputs
        del batch

        grad_accum += 1
        if grad_accum != args.accum_iter:
            loss = torch.cat([loss, itr_loss], dim=0)
        else:
            loss = loss.mean()  # sum over elements, mean over batch.

            loss_value = loss.item()

            print(
                f"step: {global_step + 1} / {args.num_train_steps}, loss: {loss_value}, clock: {datetime.now() - start_time}",
                end='\r')

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            # clip grad norm
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()


            # global stepper.
            global_step += 1
            if global_step % args.eval_every_steps == 0:
                print()
                evaluate(val_loader, model, criterion, evaluator, device, args, global_step)
            if not args.no_snap and global_step % args.checkpoint_every_steps == 0:
                misc.save_snapshot(args, model.module, optimizer, global_step,
                                   f'./experiments/{args.group}_{args.exp}/snapshots/{global_step}.pt')
            # graphslot doesn't train on epochs, just on steps.
            if global_step >= args.num_train_steps:
                print('done training')
                misc.save_snapshot(args, model.module, optimizer, global_step,
                                   f'./experiments/{args.group}_{args.exp}/snapshots/{args.mode}_{global_step}.pt')
                print('exiting')
                sys.exit(0)

            grad_accum = 0
            loss = None

    return global_step, loss


@torch.no_grad()
def evaluate(data_loader, model, criterion, evaluator, device, args, name="test"):
    # switch to evaluation mode
    model.eval()

    dataset = data_loader.dataset
    dataset.reset_itr()
    len_data = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)

    loss_value = 1e12
    ari_nobg_running = {'total': 0, 'count': 0}
    
    for i_batch, (video, boxes, segmentations, flow, padding_mask, mask) in enumerate(data_loader):
        video = video.squeeze(0).to(device, non_blocking=True)  # [64, 6, 64, 64, 3]
        boxes = boxes.squeeze(0).to(device, non_blocking=True)
        flow = flow.squeeze(0).to(device, non_blocking=True)
        padding_mask = padding_mask.squeeze(0).to(device, non_blocking=True)
        mask = mask.squeeze(0).to(device, non_blocking=True) if len(mask) > 0 else None
        segmentations = segmentations.squeeze(0).to(device, non_blocking=True)
        batch = (video, boxes, segmentations, flow, padding_mask, mask)

        # conditioning = boxes
        conditioning = None

        # compute output
        if args.model_type == "graphslot":
            outputs = graphslot.modules.evaluator.eval_step(model, batch, slice_size=args.eval_slice_size)
        else:
            outputs = model(video=video, conditioning=conditioning,
                            padding_mask=padding_mask, **args.kwargs)
        
        loss = criterion(outputs, batch)
        loss = loss.mean()  # mean over devices
        loss_value = loss.item()

        ari_nobg = evaluator(outputs, batch, args)

        for k, v in ari_nobg.items():
            ari_nobg_running[k] += v.item()
        
        print(f"{i_batch + 1} / {len_data}, loss: {loss_value}, running_ari_fg: {ari_nobg_running['total'] / ari_nobg_running['count']}", end='\r')

        # visualize first 3 iterations
        if i_batch == 0:
            for i_sample in range(3):

                if args.model_type == "graphslot":
                    B, T, H, W, _ = video.shape
                    attn = outputs[2][i_sample].squeeze(1)
                    attn = attn.reshape(shape=(
                    attn.shape[0], args.num_slots, int(attn.shape[-1] ** (1 / 2)), int(attn.shape[-1] ** (1 / 2))))
                    if attn.shape[-2:] != video.shape[-3:-1]:
                        attn = F.interpolate(attn,
                                            size=video.shape[-3:-1],
                                            mode='bilinear',
                                            align_corners=True).view(attn.shape[0], args.num_slots,
                                                                    *video.shape[-3:-1])
                    pr_flow = outputs[1][i_sample]
                    pr_seg = outputs[0][i_sample].squeeze(-1)
                    pr_flow = torch.clamp(pr_flow, 0.0, 1.0)
                else:
                    pr_flow = outputs[2][i_sample]
                    B, T, H, W, _ = video.shape
                    pr_flow = torch.cat([torch.zeros(1, H, W, 2).to(pr_flow.get_device()), pr_flow], dim=0)
                    pr_flow = torchvision.utils.flow_to_image(
                        pr_flow.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(shape=(T, H, W, 3))
                    attn = outputs[4][i_sample]
                    attn = attn.reshape(shape=(T, H, W, args.num_slots)).permute(0, 3, 1, 2)
                    pr_seg = outputs[1][i_sample].squeeze(-1)
                    pr_vid = outputs[0][i_sample]
                    pr_vid = torch.clamp(pr_vid, 0.0, 1.0)
                    pr_flow = torch.clamp(pr_flow, 0.0, 1.0)
    final_loss = loss_value
    final_ari_nobg = ari_nobg_running['total'] / ari_nobg_running['count']

    print(f"{name}: loss: {final_loss}, ari_fg: {final_ari_nobg}")

    # switch back to training
    model.train()

    return final_loss, final_ari_nobg


def run(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.gpu[0])

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train, dataset_val = build_datasets(args)

    # Not using DistributedDataParallel ... only DataParallel
    # Need to set batch size to 1 because only passing through the torch dataset interface
    train_loader = torch.utils.data.DataLoader(dataset_train, 1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset_val, 1, shuffle=False)

    # Model setup
    model, criterion, evaluator = processors_dict[args.model_type](args)
    model = model.to(device)
    criterion = criterion.to(device)
    evaluator = evaluator.to(device)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.resume_from is not None:
        _, resume_step = misc.load_snapshot(model, optimizer, device, args.resume_from)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print("lr: %.2e" % args.lr)
    print(f"effective batch size: {args.batch_size * args.accum_iter}")

    # Loss
    print("criterion = %s" % str(criterion))

    # make dataparallel
    model = nn.DataParallel(model, device_ids=args.gpu)
    criterion = nn.DataParallel(criterion, device_ids=args.gpu)

    print(f"Start training for {args.num_train_steps} steps.")
    start_time = datetime.now()
    global_step = resume_step if args.resume_from is not None else 0

    # eval only
    if args.eval:
        evaluate(val_loader, model, criterion, evaluator, device, args, f"eval")
        sys.exit(1)

    for epoch in range(args.epochs):
        step_add, loss = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch,
            global_step, start_time,
            args.max_grad_norm, args,
            val_loader, evaluator
        )
        global_step += step_add
        print(f"epoch: {epoch + 1}, loss: {loss}, clock: {datetime.now() - start_time}")

        evaluate(val_loader, model,
                 criterion, device, args,
                 f"epoch_{epoch + 1}")

        if not args.no_snap:
            misc.save_snapshot(args, model.module, optimizer, global_step,
                               f'./experiments/{args.exp}/snapshots/{epoch + 1}.pt')

        # global stepper
        if global_step >= args.num_train_steps:
            break


def main():
    args = get_args()
    run(args)


def test():
    main()
