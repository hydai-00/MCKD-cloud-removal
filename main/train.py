import os
import argparse
import torch
import torch.nn as nn
from utils.checkpoint import save_checkpoint
from torch.utils.data import DataLoader
from loss import *
from dataloader import *
import numpy as np
from utils.common import AverageMeter, initialize_logger, record_loss
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import warnings
from tqdm import tqdm
from utils.metric import *
import time
import cv2
import random

from model.mckd import Net



torch.manual_seed(100)

global KD_step
KD_step = 0

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='My', choices=['My'],
                        help='Select the model to train')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model save path (if not set, will be set automatically based on model_type)')
    parser.add_argument('--add_message', default='My_end_KD_teacher', type=str,
                        help='suffix for the run name under result/')
    parser.add_argument('--num_workers', default=6, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.0001, type=float, help='experimnt setting')
    parser.add_argument('--optimizer', default='adamw', type=str, help='GPUs used for training')
    parser.add_argument('--batch_size', default=12, type=int, help='')
    parser.add_argument('--backup_dir', default='./backup', type=str, help='')
    parser.add_argument('--star_epoch', default=0, type=int, help='')
    parser.add_argument('--total_epoch', default=15, type=int, help='')
    parser.add_argument('--checkpoint_interval', default=1, type=int, help='')
    parser.add_argument('--weight_path', default=None, type=str,
                        help='optional stage-2 warm start')
    parser.add_argument('--cuda_num', default='0', type=str,
                        help='GPU index written to CUDA_VISIBLE_DEVICES')

    parser.add_argument('--KD', type=bool, default=True)
    parser.add_argument('--random_sim', type=bool, default=False)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--dataset_name', type=str, default='Sen12', choices=['Sen12'],
                        help='this release implements the SEN12MS-CR path only')
    parser.add_argument('--input_data_folder', type=str, default='../SEN12MS_dataset')
    parser.add_argument('--data_list_filepath', type=str, default='../splits/splits.csv')
    parser.add_argument('--student_only', type=bool, default=False)
    parser.add_argument('--t_copy', type=bool, default=True)
    parser.add_argument('--test_pre', type=bool, default=True)
    parser.add_argument('--copy_begin', type=bool, default=True)
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--teacher_path', type=str,
                        default='./result/all_My_My_end_KD_teacher/last.pkl',
                        help='stage-1 checkpoint read when --student_only is set')
    parser.add_argument('--is_test', type=bool, default=False)
    parser.add_argument('--seed', default=None, type=int,
                        help='if set, seeds random/numpy/torch after model construction;'
                             ' default None keeps the historical behaviour')

    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.path.join('/', f'all_{args.model_type}')
        args.data_list_filepath = '../splits/splits.csv'
        args.total_epoch = 15
        if args.add_message is not None:
            args.model_path = args.model_path + "_" + args.add_message
    if args.load_size != 256:
        args.model_path = args.model_path + "_" + str(args.load_size)
    if args.KD:
        args.model_path = args.model_path + '_KD' + '_teacher'
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args


def train(train_loader, network, criterion, optimizer, network_t=None,optimizer_t=None,epoch_idx=None):
    losses = AverageMeter()
    torch.cuda.empty_cache()
    network.train()

    idx_iter = 0
    pbar = tqdm(train_loader, disable=True)
    for i, batch in enumerate(pbar):
        cloudy_img = batch['cloudy_data'].cuda()
        sim_img = batch['sim_data'].cuda()
        s1_img = batch['s1_data'].cuda()
        target_img = batch['target'].cuda()

        a = random.randint(0,1)

        if KD_step == 0:
            output = network(torch.concat([cloudy_img, sim_img], dim=1))
        else:
            output = network(torch.concat([cloudy_img, s1_img], dim=1))

        if network_t is not None:
            if True:
                m1 = network.module.RGB_pre
                m2 = network_t.module.RGB_pre

                m2.weight.data.copy_(m1.weight.data)
                m2.bias.data.copy_(m1.bias.data)

                for m1, m2 in zip(network.module.encoder_list_rgb, network_t.module.encoder_list_rgb):
                    m2.load_state_dict(m1.state_dict())

                for m1, m2 in zip(network.module.processor_list_edge, network_t.module.processor_list_edge):
                    m2.Down_rgb.load_state_dict(m1.Down_rgb.state_dict())

                for p in network_t.module.RGB_pre.parameters():
                    p.requires_grad = False

                for p in network_t.module.encoder_list_rgb.parameters():
                    p.requires_grad = False

                for m in network_t.module.processor_list_edge:
                    for p in m.Down_rgb.parameters():
                        p.requires_grad = False

            output_t = network_t(torch.concat([cloudy_img, sim_img], dim=1))

            pred_KD = network.module.KD_label
            gt_KD = network_t.module.KD_label

            t_pred_KD = network.module.D_KD_label
            t_gt_KD = network_t.module.D_KD_label

        if criterion.input == 2:
            loss = criterion.forward(output, target_img)
        elif criterion.input == 3:
            mask = batch['mask'].cuda()
            loss = criterion.forward(output, target_img, mask)
        elif criterion.input == 'KD':
            loss = criterion.forward(output, target_img, pred_KD, gt_KD,epoch=epoch_idx)
            loss_t = criterion.forward(output_t, target_img, t_pred_KD, t_gt_KD,True,pred_KD,gt_KD,epoch=epoch_idx)
        elif criterion.input == 'KD2':
            loss = criterion.forward(output, target_img, [pred_KD,network.module.KD_out], [gt_KD,network_t.module.KD_out])
        else:
            mask = batch['mask'].cuda()
            loss = criterion.forward(output, target_img, mask, cloudy_img)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        if optimizer_t is not None:
            optimizer_t.zero_grad()
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(network_t.parameters(), max_norm=1.0)
            optimizer_t.step()

        losses.update(loss.item())
        idx_iter += 1
    return losses.avg


def validate(eval_loader, network, criterion, epoch, result_path):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    SAM = AverageMeter()
    MAE = AverageMeter()
    losses = AverageMeter()
    epoch_path = os.path.join(result_path, str(epoch))
    os.makedirs(epoch_path, exist_ok=True)

    pbar = tqdm(eval_loader, desc='Evaluating', unit="batch",disable=True)

    for i, batch in enumerate(pbar):
        source_img = batch['cloudy_data'].cuda()
        target_img = batch['target'].cuda()
        s1_img = batch['s1_data'].cuda()
        sim_img = batch['sim_data'].cuda()
        source = batch['source'].cuda()
        idx_img = batch['file_name']        

        if KD_step == 0:
            output = network(torch.concat([source_img, sim_img], dim=1)).clamp_(0, 1)
        else:
            output = network(torch.concat([source_img, s1_img], dim=1)).clamp_(0, 1)

        loss = criterion(target_img, output)
        PSNR_val = Psnr(target_img, output)
        SSIM_val = Ssim(target_img, output)
        MAE_val = Mae(target_img, output)
        SAM_val = Sam(target_img, output)

        losses.update(loss.item())
        SAM.update(SAM_val)
        PSNR.update(PSNR_val)
        SSIM.update(SSIM_val)
        MAE.update(MAE_val)

        if i % 100 == 0:
            save_image(output, i, 'out', epoch_path)
            save_image(target_img, i, 'gt', epoch_path)

    return losses.avg, SAM.avg, PSNR.avg, SSIM.avg, MAE.avg


def save_image(t, i, target, path):
    """Write one preview tile: bands 4/3/2 of the SEN12MS-CR stack, as 8-bit BGR."""
    t = t[0]
    t = t[[3, 2, 1], ...]          # (3, H, W), false colour
    t = torch.clamp(t * 5, 0, 1) * 255.0
    t = t.detach().cpu().numpy().astype(np.uint8)
    image = np.transpose(t, (1, 2, 0))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path, f'{i}_{target}.jpg'), bgr)


if __name__ == '__main__':
    print('---------------------------start_train_teacher_model---------------------------')
    args = arg_parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num


    network = Net(13, 2)
    criterion = sl1_ssim_sam_loss().cuda()

    network = nn.DataParallel(network).cuda()

    # Applied here: after the model exists, so both arms draw the same random
    # initialisation, and before any DataLoader is built, so both arms shuffle alike.
    # The module-level manual_seed(100) alone does not cover random/numpy.
    if getattr(args, 'seed', None) is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not args.student_only:
        model_path = './result/' + args.model_path
        result_path = os.path.join(model_path, 'vis')
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(result_path, exist_ok=True)
        loss_csv = open(os.path.join(model_path, 'loss.csv'), 'w+')
        log_dir = os.path.join(model_path, 'train.log')
        logger = initialize_logger(log_dir)

    criterion_test = L1_Loss().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    if args.dataset_name == 'Sen12':
        from dataloader import *

        train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(args.data_list_filepath)
        train_data = AlignedDataset(args, train_filelist)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_data = AlignedDataset(args, val_filelist, False)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        if args.test_pre:
            test_data = AlignedDataset(args, test_filelist, False)
            test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)

    if not args.student_only:
        best_loss = float('inf')

        for epoch_idx in range(args.star_epoch, args.star_epoch + args.total_epoch):
            start_time = time.time()
            train_loss = train(train_loader, network, criterion, optimizer)
            val_loss, sam, psnr, ssim, mae = validate(val_loader, network, criterion_test, epoch_idx, result_path)
            if args.lr_decay:
                scheduler.step()
            if val_loss < best_loss:
                save_checkpoint(model_path, epoch_idx, network, optimizer, name='best')
                logger.info(f"best epoch")
                best_loss = val_loss

            epoch_time = time.time() - start_time

            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch_idx}], Time:{epoch_time:.4f}, lr:{lr:.6f}, "
                  f"Train Loss:{train_loss:.6f}, Test Loss:{val_loss:.6f}, PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, MAE:{mae:.4f}, SAM:{sam:.4f}")
            record_loss(loss_csv, epoch_idx, epoch_time, lr, train_loss, val_loss)
            logger.info(f"Epoch [{epoch_idx}], Time:{epoch_time:.4f}, lr:{lr:.6f}, "
                        f"Train Loss:{train_loss:.6f}, Test Loss:{val_loss:.6f}, PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, MAE:{mae:.4f}, SAM:{sam:.4f}")
            save_checkpoint(model_path, epoch_idx, network, optimizer, name='last')
            logger.info(f"save epoch{epoch_idx} to last.pkl")
    print('---------------------------start_train_student_model---------------------------')

    network = Net(13, 2, ist=True)
    network = nn.DataParallel(network).cuda()
    if args.student_only:
        teacher_path = args.teacher_path
    else:
        teacher_path = os.path.join('./result/' + args.model_path, 'last.pkl')
    if args.t_copy:
        weight = torch.load(teacher_path)["state_dict"]
        network.load_state_dict(weight,strict=False)
        print("Load weight from " + teacher_path)

    args.model_path = args.model_path[:-7] + 'student'

    KD_step = 1
    network_s = Net(13, 2, iss=True)
    criterion = KD_loss_nl2_select_sarea(13)

    network_s = nn.DataParallel(network_s).cuda()
    if args.copy_begin:
        network_s.load_state_dict(weight,strict=False)
        print("Load student weight from " + teacher_path)

    if args.weight_path is not None:
        weight = torch.load(args.weight_path, weights_only=True)["state_dict"]
        network_s.load_state_dict(weight)
        print("Load student weight from " + args.weight_path)

    model_path = './result/' + args.model_path
    result_path = os.path.join(model_path, 'vis')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    loss_csv = open(os.path.join(model_path, 'loss.csv'), 'w+')
    log_dir = os.path.join(model_path, 'train.log')
    logger = initialize_logger(log_dir)

    optimizer = torch.optim.AdamW(network_s.parameters(), lr=args.lr)
    optimizer_t = torch.optim.AdamW(network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    best_loss = float('inf')

    args.batch_size = 12
    if args.dataset_name == 'Sen12':
        from dataloader import *

        train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(args.data_list_filepath)
        train_data = AlignedDataset(args, train_filelist)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_data = AlignedDataset(args, val_filelist, False)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        if args.test_pre:
            test_data = AlignedDataset(args, test_filelist, False)
            test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)

    for epoch_idx in range(args.star_epoch, args.star_epoch + args.total_epoch):
        start_time = time.time()
        train_loss = train(train_loader, network_s, criterion, optimizer, network,optimizer_t,epoch_idx)
        val_loss, sam, psnr, ssim, mae = validate(val_loader, network_s, criterion_test, epoch_idx, result_path)
        if args.test_pre and epoch_idx >= 8:
            test_loss, test_sam, test_psnr, test_ssim, test_mae = validate(test_loader, network_s, criterion_test, epoch_idx, result_path)
        if val_loss < best_loss:
            save_checkpoint(model_path, epoch_idx, network_s, optimizer, name='best')
            logger.info(f"save epoch{epoch_idx} to best.pkl")
            if epoch_idx < 10:
                save_checkpoint(model_path, epoch_idx, network_s, optimizer, name='best10')
            best_loss = val_loss
        epoch_time = time.time() - start_time

        lr = optimizer.param_groups[0]['lr']
        if args.test_pre and epoch_idx >= 8:
            print(f"Epoch [{epoch_idx}], Time:{epoch_time:.4f}, lr:{lr:.6f}, "
                  f"Train Loss:{train_loss:.6f}, Val Loss:{val_loss:.6f}, PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, MAE:{mae:.4f}, SAM:{sam:.4f}, Test Loss:{test_loss:.6f}, PSNR:{test_psnr:.4f}, SSIM:{test_ssim:.4f}, MAE:{test_mae:.4f}, SAM:{test_sam:.4f}")
            logger.info(f"Epoch [{epoch_idx}], Time:{epoch_time:.4f}, lr:{lr:.6f}, "
                  f"Train Loss:{train_loss:.6f}, Val Loss:{val_loss:.6f}, PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, MAE:{mae:.4f}, SAM:{sam:.4f}, Test Loss:{test_loss:.6f}, PSNR:{test_psnr:.4f}, SSIM:{test_ssim:.4f}, MAE:{test_mae:.4f}, SAM:{test_sam:.4f}")
        else:
            print(f"Epoch [{epoch_idx}], Time:{epoch_time:.4f}, lr:{lr:.6f}, "
                  f"Train Loss:{train_loss:.6f}, Test Loss:{val_loss:.6f}, PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, MAE:{mae:.4f}, SAM:{sam:.4f}")
            logger.info(f"Epoch [{epoch_idx}], Time:{epoch_time:.4f}, lr:{lr:.6f}, "
                  f"Train Loss:{train_loss:.6f}, Test Loss:{val_loss:.6f}, PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, MAE:{mae:.4f}, SAM:{sam:.4f}")       
        record_loss(loss_csv, epoch_idx, epoch_time, lr, train_loss, val_loss)
        if args.lr_decay:
            scheduler.step()
    save_checkpoint(model_path, epoch_idx, network_s, optimizer, name='last')
    logger.info(f"save epoch{epoch_idx} to last.pkl")
