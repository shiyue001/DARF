# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
import math
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.checkpoint import checkpoint as run_checkpoint

# import CLIP.clip_utils as clip_utils
import clip as clip_utils
from ibrnet.data_loaders import dataset_dict
from ibrnet.render_ray import render_rays
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.criterion import Criterion, PerceptualLoss
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, save_current_code
import config
import torch.distributed as dist
from ibrnet.projection import Projector
from ibrnet.data_loaders.create_training_dataset import create_training_dataset
from ibrnet.spynet import SPyNet



def compute_flow(spynet, lqs, device):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        
        n, t, c, h, w = lqs.size()

        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        # pdb.set_trace()
        flows_backward = spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

def get_embed_fn(device, model_type, num_layers=-1, spatial=False, checkpoint=False, clip_cache_root=None):
    if model_type.startswith('clip_'):
        if model_type == 'clip_rn50':
            assert clip_cache_root
            clip_utils.load_rn(jit=False, root=clip_cache_root)
            if spatial:
                _clip_dtype = clip_utils.clip_model_rn.clip_model.dtype
                assert num_layers == -1
                def embed(ims):
                    ims = clip_utils.CLIP_NORMALIZE(ims).type(_clip_dtype)
                    return clip_utils.clip_model_rn.clip_model.visual.featurize(ims)  # [N,C,56,56]
            else:
                embed = lambda ims: clip_utils.clip_model_rn(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers).unsqueeze(1)
            assert not clip_utils.clip_model_rn.training
        elif model_type.startswith('clip_vit'):
            assert clip_cache_root
            if model_type == 'clip_vit':
                clip_utils.load_vit(root=clip_cache_root)
            elif model_type == 'clip_vit_b16':
                clip_utils.load_vit('ViT-B/16', root=clip_cache_root)
            if spatial:
                def embed(ims):
                    emb = clip_utils.clip_model_vit(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers)  # [N,L=50,D]
                    return emb[:, 1:].view(emb.shape[0], 7, 7, emb.shape[2]).permute(0, 3, 1, 2)  # [N,D,7,7]
            else:
                embed = lambda ims: clip_utils.clip_model_vit(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers)  # [N,L=50,D]
            assert not clip_utils.clip_model_vit.training
        elif model_type == 'clip_rn50x4':
            assert not spatial
            clip_utils.load_rn(name='RN50x4', jit=False)
            assert not clip_utils.clip_model_rn.training
            embed = lambda ims: clip_utils.clip_model_rn(images_or_text=clip_utils.CLIP_NORMALIZE(ims), featurize=False)
    elif model_type.startswith('timm_'):
        assert num_layers == -1
        assert not spatial

        model_type = model_type[len('timm_'):]
        encoder = timm.create_model(model_type, pretrained=True, num_classes=0)
        encoder.eval()
        normalize = torchvision.transforms.Normalize(
            encoder.default_cfg['mean'], encoder.default_cfg['std'])  # normalize an image that is already scaled to [0, 1]
        encoder = nn.DataParallel(encoder).to(device)
        embed = lambda ims: encoder(normalize(ims)).unsqueeze(1)
    elif model_type.startswith('torch_'):
        assert num_layers == -1
        assert not spatial

        model_type = model_type[len('torch_'):]
        encoder = torch.hub.load('pytorch/vision:v0.6.0', model_type, pretrained=True)
        encoder.eval()
        encoder = nn.DataParallel(encoder).to(device)
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize an image that is already scaled to [0, 1]
        embed = lambda ims: encoder(normalize(ims)).unsqueeze(1)
    else:
        raise ValueError

    if checkpoint:
        return lambda x: run_checkpoint(embed, x)

    return embed


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

# def flows

def train(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.outdir, args.expname)
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               worker_init_fn=lambda _: np.random.seed(),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)

    # create validation dataset
    # val_dataset = dataset_dict[args.eval_dataset](args, 'validation',
                                                #   scenes=args.eval_scenes)
    # pdb.set_trace()
    val_dataset = dataset_dict[args.eval_dataset](args, 'test',
                                                scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create IBRNet model
    model = IBRNetModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)

    # spynet
    # spynet = SPyNet(pretrained='/cluster/project/cvl/jiezcao_share/spynet_20210409-c6c1bd09.pth').to(device)
    # spynet.requires_grad_(False)
    
    # create projector
    projector = Projector(device=device)
    # Create criterion
    criterion = Criterion()
    # Create consistency loss
    criterion_cons = PerceptualLoss([2,2,2], [0.6,0.3,0.1], device)
    tb_dir = os.path.join(args.outdir, args.expname)
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        print('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0
    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            # load training rays
            # print(train_data['src_rgbs'].shape,train_data['rgb'].shape)
            ray_sampler = RaySamplerSingleImage(train_data, device)
            N_rand = int(1.0 * args.N_rand * args.num_source_views / train_data['src_rgbs'][0].shape[0])
            ray_batch = ray_sampler.random_sample(N_rand,
                                                  sample_mode=args.sample_mode,
                                                  center_ratio=args.center_ratio,
                                                  )
            # print(ray_batch['ray_d'].shape)
            # print(torch.ones_like(ray_batch['ray_d'][..., 0]).shape)
            # ray_batch['ray_d'].shape = [300,400]


            # comp flows
            # pdb.set_trace()
            # n, t, h, w, c = ray_batch['src_rgbs'].shape
            # h, w = int(math.ceil(h/16)*16), int(math.ceil(w/16)*16)
            # dwon_src_rgbs = F.interpolate(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2), (h//4, w//4)).unsqueeze(0) #[1, 9, 96, 128, 3] 
            # src_rgbs = F.interpolate(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2), (h, w)).unsqueeze(0)
            
            # flow-guided feature
            # flows = compute_flow(spynet, dwon_src_rgbs, device) #[1, 10, 2, 96, 128]
            # # print(flows[0].min(), flows[0].max(), flows[1].min(), flows[1].max())
            # # pdb.set_trace()
            # featmaps = model.feature_net(src_rgbs.contiguous(), flows) #[11, 64, 96, 128]
            # # print(featmaps[0].min(), featmaps[0].max(), featmaps[1].min(), featmaps[1].max())
            
            # FFC
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)) 
            # order_index = ray_batch['order_ids']
            # order_index = None
            # old UResNet
            # featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)) #[9, 32, 96, 128]

            ret = render_rays(ray_batch=ray_batch,
                              model=model,
                              projector=projector,
                              featmaps=featmaps,
                              N_samples=args.N_samples,
                              inv_uniform=args.inv_uniform,
                              N_importance=args.N_importance,
                              det=args.det,
                              white_bkgd=args.white_bkgd,
                              )
            # order_index=order_index
            # print(ret['outputs_coarse']['rgb'].shape)

            a = ray_batch['rgb'].unsqueeze(0)
            b = ret['outputs_coarse']['rgb'].unsqueeze(0)
            c = ret['outputs_fine']['rgb'].unsqueeze(0)
            
            # print(b.min(), b.max(), c.min(), c.max())
            # compute loss
            model.optimizer.zero_grad()
            loss, scalars_to_log = criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)
            # consistency_loss0 = criterion_cons(a.view(-1, 3, 120, 160),b.view(-1, 3, 120, 160))
            # loss += consistency_loss0
            # if args.MSC_loss:
            #     coarse_consistency_loss = criterion_cons(a.view(-1, 3, args.MSC_x, args.MSC_y),b.view(-1, 3, args.MSC_x, args.MSC_y))
            #     loss += args.coarse_MSC_weight * coarse_consistency_loss

            if ret['outputs_fine'] is not None:
                fine_loss, scalars_to_log = criterion(ret['outputs_fine'], ray_batch, scalars_to_log)
                loss += fine_loss
                # if args.MSC_loss:
                #     fine_consistency_loss = criterion_cons(a.view(-1, 3, args.MSC_x, args.MSC_y),c.view(-1, 3, args.MSC_x, args.MSC_y))
                #     loss += args.fine_MSC_weight * fine_consistency_loss

            loss.backward()
            scalars_to_log['loss'] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item()
                    scalars_to_log['train/coarse-loss'] = mse_error
                    scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
                    if ret['outputs_fine'] is not None:
                        mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                        scalars_to_log['train/fine-loss'] = mse_error
                        scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

                    logstr = '{} Epoch: {}  step: {} '.format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    print(logstr)
                    print('each iter time {:.05f} seconds'.format(dt))

                if global_step % args.i_weights == 0:
                    print('Saving checkpoints at {} to {}...'.format(global_step, out_folder))
                    fpath = os.path.join(out_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                # if global_step % args.i_img == 0:
                #     print('Logging a random validation view...')
                #     val_data = next(val_loader_iterator)
                #     tmp_ray_sampler = RaySamplerSingleImage(val_data, device, render_stride=args.render_stride)
                #     H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                #     gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                #     log_view_to_tb(writer, global_step, args, model, tmp_ray_sampler, projector,
                #                    gt_img, render_stride=args.render_stride, prefix='val/')
                #     torch.cuda.empty_cache()

                #     print('Logging current training view...')
                #     tmp_ray_train_sampler = RaySamplerSingleImage(train_data, device,
                #                                                   render_stride=1)
                #     H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                #     gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                #     log_view_to_tb(writer, global_step, args, model, tmp_ray_train_sampler, projector,
                #                    gt_img, render_stride=1, prefix='train/')
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


def log_view_to_tb(writer, global_step, args, model, ray_sampler, projector, gt_img,
                   render_stride=1, prefix=''):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  projector=projector,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  inv_uniform=args.inv_uniform,
                                  det=True,
                                  N_importance=args.N_importance,
                                  white_bkgd=args.white_bkgd,
                                  render_stride=render_stride,
                                  featmaps=featmaps)

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret['outputs_coarse']['rgb'].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

    depth_im = ret['outputs_coarse']['depth'].detach().cpu()
    acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()

    if ret['outputs_fine'] is None:
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
    else:
        rgb_fine = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_im = torch.cat((depth_im, ret['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = torch.cat((acc_map, torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

    # write scalar
    pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    model.switch_to_train()


if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)
