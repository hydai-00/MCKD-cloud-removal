from torch.utils.data import DataLoader
from utils.common import AverageMeter
from utils.metric import *
from tqdm import tqdm
import warnings
import argparse
import os
# Respect CUDA_VISIBLE_DEVICES if the caller sets it; otherwise use device 0.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
import torch.nn as nn


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='My', choices=['My'],
                        help='Select the model to train')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model save path (if not set, will be set automatically based on model_type)')
    parser.add_argument('--add_message', default=None, type=str, help='messege on name')
    parser.add_argument('--num_workers', default=6, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.0001, type=float, help='experiment setting')
    parser.add_argument('--optimizer', default='adamw', type=str, help='GPUs used for training')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--backup_dir', default='./backup', type=str, help='')
    parser.add_argument('--star_epoch', default=0, type=int, help='')
    parser.add_argument('--total_epoch', default=15, type=int, help='')
    parser.add_argument('--checkpoint_interval', default=1, type=int, help='')
    parser.add_argument('--weight_path', default='./result/all_My_My_end_KD_student/last.pkl',
                        type=str, help='checkpoint written by train.py')
    parser.add_argument('--cuda_num', default='0', type=str, help='')

    parser.add_argument('--dataset_name', type=str, default='Sen12', choices=['Sen12'],
                        help='this release implements the SEN12MS-CR path only')
    parser.add_argument('--KD', type=bool, default=True)
    parser.add_argument('--random_sim', type=bool, default=False)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--input_data_folder', type=str, default='../SEN12MS_dataset')
    parser.add_argument('--data_list_filepath', type=str, default='../splits/splits.csv')
    parser.add_argument('--is_test', type=bool, default=True)

    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.path.join('/', f'all_{args.model_type}')
        args.data_list_filepath = '../splits/splits.csv'
        args.total_epoch = 15
        if args.add_message is not None:
            args.model_path = args.model_path + "_" + args.add_message
    if args.load_size != 256:
        args.model_path = args.model_path + "_" + str(args.load_size)
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args



def eval(eval_loader, network,save_path):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    SAM = AverageMeter()
    MAE = AverageMeter()

    vis_dir = os.path.join(save_path, "testvis")
    os.makedirs(vis_dir, exist_ok=True)

    i = 0
    for batch in tqdm(eval_loader, desc='Evaluating', unit="batch"):
        source_img = batch['cloudy_data'].cuda()
        target_img = batch['target'].cuda()
        s1_img = batch['s1_data'].cuda()
        source = batch['source'].cuda()
        idx_img = batch['file_name']


        output = network(torch.concat([source_img, s1_img], dim=1))
        output = output.clamp_(0, 1)

        PSNR_val = Psnr(target_img, output)
        SSIM_val = Ssim(target_img, output)
        SAM_val = Sam(target_img, output)
        MAE_val = Mae(target_img, output)

        PSNR.update(PSNR_val)
        SSIM.update(SSIM_val)
        SAM.update(SAM_val)
        MAE.update(MAE_val)

        i = i + 1

    print('PSNR: %f\n'
          'SSIM: %f\n'
          'SAM: %f\n'
          'MAE: %f' % (PSNR.avg, SSIM.avg,SAM.avg, MAE.avg))
    fd = open('metric.txt', 'a')
    fd.write('  PSNR.avg:' + str(PSNR.avg) + '  SSIM.avg:' + str(SSIM.avg) + '  SAM.avg:' + str(SAM.avg) +'  MAE.avg:' + str(MAE.avg) + '\n')
    fd.close()


if __name__ == '__main__':
    args = arg_parse()

    from dataloader import *

    _, _, test_filelist = get_train_val_test_filelists(args.data_list_filepath)
    eval_data = AlignedDataset(args, test_filelist)
    eval_loader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    weight_path = args.weight_path
    from model.mckd import Net

    network = Net(13, 2)

    network.cuda()
    print('_____load______')
    print(weight_path)
    weight = torch.load(weight_path)["state_dict"]
    network = nn.DataParallel(network).cuda()
    network.load_state_dict(weight,strict=False)
    network.eval()
    for _, param in network.named_parameters():
        param.requires_grad = False
    eval(eval_loader, network,os.path.dirname(args.weight_path))
