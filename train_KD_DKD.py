import os
import argparse
import torch
import torch.nn as nn
from utils.check_point_rw import save_checkpoint
from torch.utils.data import DataLoader
from model.hpn import hpn_cr
from loss import *
# from dataloader import *
from dataloader_KD import *
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

from model.My_end import Net



torch.manual_seed(100)

global net_mode

global KD_step
net_mode = -1
KD_step = 0
dataset_mode = 0

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='My',
                        choices=['DSen2_CR', 'CAC', 'USSDRN', 'HPN', 'My', 'USSRN', 'Align_cr', "GLF_CR", "HSSP",
                                 "E_My"],
                        help='Select the model to train')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model save path (if not set, will be set automatically based on model_type)')
    parser.add_argument('--is_cropland', type=bool, default=False,
                        help='Use cropland dataset only')
    parser.add_argument('--add_message', default='My_end', type=str, help='messege on name')
    # parser.add_argument('--add_message', default=None, type=str, help='messege on name')
    parser.add_argument('--num_workers', default=6, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.0001, type=float, help='experimnt setting')
    parser.add_argument('--optimizer', default='adamw', type=str, help='GPUs used for training')
    parser.add_argument('--batch_size', default=12, type=int, help='')
    parser.add_argument('--backup_dir', default='./backup', type=str, help='')
    parser.add_argument('--star_epoch', default=0, type=int, help='')
    parser.add_argument('--total_epoch', default=15, type=int, help='')
    parser.add_argument('--checkpoint_interval', default=1, type=int, help='')
    parser.add_argument('--weight_path', default=None, type=str, help='./backup/weight_10.pth')
    parser.add_argument('--cuda_num', default='0', type=str, help='')

    parser.add_argument('--KD', type=bool, default=True)
    parser.add_argument('--random_sim', type=bool, default=False)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--dataset_name', type=str, default='Smile', choices=['Sen12', 'Smile'])
    parser.add_argument('--input_data_folder', type=str, default='../SEN12MS_dataset')
    #parser.add_argument('--input_data_folder', type=str, default='../smile_cr')
    parser.add_argument('--data_list_filepath', type=str, default='../csv_script/splits.csv')
    parser.add_argument('--student_only', type=bool, default=False)
    parser.add_argument('--t_copy', type=bool, default=True)
    parser.add_argument('--test_pre', type=bool, default=True)
    parser.add_argument('--copy_begin', type=bool, default=True)
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--teacher_path', type=str, default="./result/all_My_My_end_Sen12_KD_teacher/last.pkl")
    #parser.add_argument('--teacher_path', type=str, default="./result/all_My_My_end_Smile_KD_teacher/last.pkl")
    parser.add_argument('--is_test', type=bool, default=False)
 
    args = parser.parse_args()

    if args.model_path is None:
        if args.is_cropland:
            args.model_path = os.path.join('/', f'cropland_{args.model_type}')
            args.data_list_filepath = '../csv_script/splits.csv'
            args.total_epoch = 50
        else:
            args.model_path = os.path.join('/', f'all_{args.model_type}')
            args.data_list_filepath = '../csv_script/splits_ori.csv'
            args.total_epoch = 15
        if not args.add_message is None:
            args.model_path = args.model_path + "_" + args.add_message
    if args.dataset_name != 'Sen12':
        args.model_path = args.model_path + "_" + args.dataset_name
        global dataset_mode
        dataset_mode = 1
        args.total_epoch = 50
        args.input_data_folder = '../smile_cr'
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
    
    #if KD_step != 0:
        #train_loader.dataset.set_L(4+epoch_idx)

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
    t = t[0]
    if dataset_mode == 0:
        t = t[[3, 2, 1], ...] 
        t = torch.clamp(t * 5, 0, 1) * 255.0
    elif dataset_mode == 1:
        t = t[[2, 1, 0], ...] 
        t = torch.clamp(t * 3, 0, 1) * 255.0
    t = t.detach().cpu().numpy().astype(np.uint8)
    image = np.transpose(t, (1, 2, 0))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path, f'{i}_{target}.jpg'), bgr)


if __name__ == '__main__':
    print('---------------------------start_train_teacher_model---------------------------')
    args = arg_parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num


    if args.dataset_name == 'Sen12':
        network = Net(13, 2)
        criterion = sl1_ssim_sam_loss().cuda()
    elif args.dataset_name == 'Smile':
        network = Net(6, 2)
        criterion = sl1_ssim_sam_loss(6).cuda()

    #if args.weight_path is not None:
        #network.load_state_dict(torch.load(args.weight_path), strict=True)
    network = nn.DataParallel(network).cuda()

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
        from dataloader_KD import *

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

    elif args.dataset_name == 'Smile':
        from dataloader_smile_KD import *

        train_data = AlignedDataset(args, 'train')
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_data = AlignedDataset(args, 'val')
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)

    #args.total_epoch = 5
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
    
    if args.dataset_name == 'Sen12':
        network = Net(13, 2,ist=True)
    elif args.dataset_name == 'Smile':
        network = Net(6, 2,ist=True)
    network = nn.DataParallel(network).cuda()
    if args.student_only:
        teacher_path = args.teacher_path
    else:
        teacher_path = os.path.join('./result/' + args.model_path, 'last.pkl')
        #teacher_path = os.path.join('./result/' + args.model_path, 'best.pkl')
    if args.t_copy:
        #weight = torch.load(teacher_path, weights_only=True)["state_dict"]
        weight = torch.load(teacher_path)["state_dict"]
        network.load_state_dict(weight,strict=False)
        print("Load weight from " + teacher_path)

    #for param in network.parameters():
        #param.requires_grad = False

    args.model_path = args.model_path[:-7] + 'student'

    KD_step = 1
    if args.dataset_name == 'Sen12':
        network_s = Net(13, 2,iss=True)
        criterion =  KD_loss_nl2_select_sarea(13)
    elif args.dataset_name == 'Smile':
        network_s = Net(6, 2,iss=True)
        criterion = KD_loss_nl2_select_sarea(6)
            
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
    #args.total_epoch = 10
    if args.dataset_name == 'Sen12':
        from dataloader_KD import *

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

    elif args.dataset_name == 'Smile':
        from dataloader_smile_KD import *

        train_data = AlignedDataset(args, 'train')
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_data = AlignedDataset(args, 'val', False)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
        if args.test_pre:
            test_data = AlignedDataset(args, 'test', False)
            test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)

    for epoch_idx in range(args.star_epoch, args.star_epoch + args.total_epoch):
        start_time = time.time()
        train_loss = train(train_loader, network_s, criterion, optimizer, network,optimizer_t,epoch_idx)
        val_loss, sam, psnr, ssim, mae = validate(val_loader, network_s, criterion_test, epoch_idx, result_path)
        if val_loss < best_loss:
            save_checkpoint(model_path, epoch_idx, network_s, optimizer, name='best')
            logger.info(f"save epoch{epoch_idx} to best.pkl")
            best_loss = val_loss
        epoch_time = time.time() - start_time
        
        #criterion.show()
        #criterion.reset()

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
