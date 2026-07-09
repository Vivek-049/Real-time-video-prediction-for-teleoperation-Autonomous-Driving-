"""
Fine-tune IFRNet_S on demo videos for t+3 prediction
Uses pretrained k+1 weights as starting point for faster convergence
"""
import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from metric import calculate_psnr, calculate_ssim
from utils import AverageMeter
import logging
from PIL import Image
from pathlib import Path


class DemoT3Dataset(Dataset):
    """Dataset for demo video t+3 triplets"""
    def __init__(self, dataset_dir, crop_size=512, augment=False):
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.augment = augment
        
        # Find all triplet folders
        self.triplet_folders = sorted([
            d for d in Path(dataset_dir).iterdir() 
            if d.is_dir() and (d / 'im1.png').exists()
        ])
        
        print(f"Loaded {len(self.triplet_folders)} triplets from {dataset_dir}")
    
    def __len__(self):
        return len(self.triplet_folders)
    
    def __getitem__(self, idx):
        folder = self.triplet_folders[idx]
        
        # Load images
        img0 = Image.open(folder / 'im1.png').convert('RGB')
        img1 = Image.open(folder / 'im2.png').convert('RGB')
        imgt = Image.open(folder / 'im3.png').convert('RGB')  # Target: t+3
        
        # Get dimensions
        w, h = img0.size
        
        # Random crop if image is larger than crop_size
        if w > self.crop_size and h > self.crop_size:
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            img0 = img0.crop((x, y, x + self.crop_size, y + self.crop_size))
            img1 = img1.crop((x, y, x + self.crop_size, y + self.crop_size))
            imgt = imgt.crop((x, y, x + self.crop_size, y + self.crop_size))
        
        # Convert to tensor
        img0 = torch.from_numpy(np.array(img0)).permute(2, 0, 1).float() / 255.0
        img1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float() / 255.0
        imgt = torch.from_numpy(np.array(imgt)).permute(2, 0, 1).float() / 255.0
        
        # embt = 1.0 for single-step prediction (will use recurrent for t+3)
        embt = torch.tensor([1.0])
        
        return img0, img1, imgt, embt


def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, ddp_model):
    local_rank = args.local_rank
    print(f'Distributed Data Parallel Training IFRNet_S on Rank {local_rank}')

    if local_rank == 0:
        os.makedirs(args.log_path, exist_ok=True)
        log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
        os.makedirs(log_path, exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('INFO')
        fhlr = logging.FileHandler(os.path.join(log_path, 'train.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)
        logger.info(args)
        
        logger.info("Fine-tuning from pretrained k+1 weights!")
        logger.info(f"Pretrained: {args.pretrained_path}")

    # Create dataset
    dataset_train = DemoT3Dataset(
        dataset_dir=args.dataset_dir, 
        crop_size=args.crop_size,
        augment=False  # No augmentation for demo overfitting
    )
    
    sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True, 
        sampler=sampler
    )
    
    args.iters_per_epoch = len(dataloader_train)
    iters = args.resume_epoch * args.iters_per_epoch
    total_iters = args.epochs * args.iters_per_epoch

    if local_rank == 0:
        logger.info(f"Starting fine-tuning:")
        logger.info(f"  Dataset: {args.dataset_dir}")
        logger.info(f"  Triplets: {len(dataset_train)}")
        logger.info(f"  Batch size: {args.batch_size} x {args.world_size} GPUs = {args.batch_size * args.world_size}")
        logger.info(f"  Iterations per epoch: {args.iters_per_epoch}")
        logger.info(f"  Total epochs: {args.epochs}")

    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr_start, weight_decay=1e-4)

    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_geo = AverageMeter()

    for epoch in range(args.resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img0, img1, imgt, embt = data
            embt = embt.view(-1, 1, 1, 1)

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(optimizer, lr)

            optimizer.zero_grad()

            imgt_pred, loss_rec, loss_geo, loss_dis = ddp_model(img0, img1, embt, imgt)

            loss = loss_rec + loss_geo
            loss.backward()
            optimizer.step()

            avg_rec.update(loss_rec.cpu().data)
            avg_geo.update(loss_geo.cpu().data)
            train_time_interval = time.time() - time_stamp

            if (iters+1) % 50 == 0 and local_rank == 0:
                logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e} loss_rec:{:.4e} loss_geo:{:.4e}'.format(
                    epoch+1, args.epochs, iters+1, total_iters, 
                    data_time_interval, train_time_interval, lr, 
                    avg_rec.avg, avg_geo.avg))
                avg_rec.reset()
                avg_geo.reset()

            iters += 1
            time_stamp = time.time()

        # Save checkpoint every N epochs
        if (epoch+1) % args.eval_interval == 0 and local_rank == 0:
            save_path = os.path.join(log_path, f'IFRNet_S_epoch{epoch+1}.pth')
            torch.save(ddp_model.module.state_dict(), save_path)
            logger.info(f'Saved checkpoint: {save_path}')

        dist.barrier()

    # Save final model
    if local_rank == 0:
        final_path = os.path.join(log_path, 'IFRNet_S_demo_finetuned.pth')
        torch.save(ddp_model.module.state_dict(), final_path)
        logger.info(f'Training complete! Final model saved: {final_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune IFRNet_S on demo videos')
    parser.add_argument('--dataset_dir', default='/storage_drive_0/vivek/video/datasets/demo_t3_triplets', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--world_size', default=3, type=int)
    parser.add_argument('--distributed', action='store_true', default=True)
    parser.add_argument('--epochs', default=200, type=int)  # More epochs for overfitting
    parser.add_argument('--eval_interval', default=20, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--lr_start', default=5e-5, type=float)  # Lower LR for fine-tuning
    parser.add_argument('--lr_end', default=1e-6, type=float)
    parser.add_argument('--log_path', default='checkpoint_demo_finetune', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--pretrained_path', 
                        default='IFRVP_k+1_laploss/IFRNet_S_latest.pth', 
                        type=str, help='Path to pretrained k+1 weights')
    args = parser.parse_args()

    # Get local rank from environment (set by torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.local_rank = local_rank
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    args.device = torch.device('cuda', local_rank)
    args.world_size = dist.get_world_size()
    args.num_workers = 8

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Use IFRNet_S (same as pretrained)
    from models.IFRNet_S import Model
    args.log_path = args.log_path + '/IFRNet_S'

    model = Model().to(args.device)
    
    # Load pretrained weights
    if os.path.exists(args.pretrained_path):
        state_dict = torch.load(args.pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        if local_rank == 0:
            print(f"✓ Loaded pretrained weights from: {args.pretrained_path}")
    else:
        if local_rank == 0:
            print(f"✗ Pretrained weights not found: {args.pretrained_path}")
            print("  Training from scratch...")
        
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    train(args, ddp_model)
    
    dist.destroy_process_group()






