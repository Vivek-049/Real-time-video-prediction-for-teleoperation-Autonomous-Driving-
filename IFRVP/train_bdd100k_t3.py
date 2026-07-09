"""
Train IFRNet on BDD100K for t+3 independent prediction.

For t+3 prediction:
- Input: frame_t, frame_t+1
- Target: frame_t+4 (3 frames after frame_t+1)
- embt = 1.0 (prediction mode, not interpolation)

This trains a dedicated model for 3-frame-ahead prediction.
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
from utils import AverageMeter, read
import logging
import cv2


# ============================================================================
# Dataset for t+3 prediction triplets
# ============================================================================

def random_crop(img0, img1, imgt, crop_size=(256, 256)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, max(1, ih-h+1))
    y = np.random.randint(0, max(1, iw-w+1))
    img0 = img0[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    imgt = imgt[x:x+h, y:y+w, :]
    return img0, img1, imgt


def random_horizontal_flip(img0, img1, imgt, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        img1 = img1[:, ::-1]
        imgt = imgt[:, ::-1]
    return img0, img1, imgt


def random_vertical_flip(img0, img1, imgt, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        img1 = img1[::-1]
        imgt = imgt[::-1]
    return img0, img1, imgt


def random_reverse_channel(img0, img1, imgt, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        img1 = img1[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
    return img0, img1, imgt


class BDD100K_T3_Dataset(Dataset):
    """
    Dataset for t+3 prediction training.
    
    Triplet format:
    - im1.png = frame_t (input 0)
    - im2.png = frame_t+1 (input 1)
    - im3.png = frame_t+4 (target, 3 frames after input 1)
    """
    def __init__(self, dataset_dir, augment=True, crop_size=256, max_triplets=None):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.crop_size = crop_size
        self.triplet_list = []
        
        # Read triplet list
        list_file = os.path.join(dataset_dir, 'tri_trainlist.txt')
        with open(list_file, 'r') as f:
            for line in f:
                name = line.strip()
                if len(name) > 0:
                    self.triplet_list.append(name)
        
        # Limit to max_triplets if specified (random subset for faster training)
        if max_triplets and max_triplets < len(self.triplet_list):
            random.shuffle(self.triplet_list)
            self.triplet_list = self.triplet_list[:max_triplets]
            print(f"Using random subset of {max_triplets} triplets for training")
        
        print(f"Loaded {len(self.triplet_list)} triplets from {dataset_dir}")
    
    def __len__(self):
        return len(self.triplet_list)
    
    def __getitem__(self, idx):
        triplet_name = self.triplet_list[idx]
        triplet_dir = os.path.join(self.dataset_dir, 'sequences', triplet_name)
        
        # Load images
        img0 = read(os.path.join(triplet_dir, 'im1.png'))  # frame_t
        img1 = read(os.path.join(triplet_dir, 'im2.png'))  # frame_t+1
        imgt = read(os.path.join(triplet_dir, 'im3.png'))  # frame_t+4 (target)
        
        # Data augmentation (driving-specific - no flips!)
        if self.augment:
            img0, img1, imgt = random_crop(img0, img1, imgt, crop_size=(self.crop_size, self.crop_size))
            # NOTE: Disabled for driving videos:
            # - Horizontal flip: Cars drive on right, flipping breaks lane logic
            # - Vertical flip: Sky is up, road is down - flipping makes no sense
            # - Channel reverse: Traffic light colors must be correct
            # - Time reversal: Cars don't drive backwards
        
        # Convert to tensors
        img0 = torch.from_numpy(img0.copy().transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.copy().transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.copy().transpose((2, 0, 1)).astype(np.float32) / 255.0)
        
        # embt = 1.0 for prediction (extrapolation beyond img1)
        embt = torch.from_numpy(np.array(1.0).reshape(1, 1, 1).astype(np.float32))
        
        return img0, img1, imgt, embt


# ============================================================================
# Training functions
# ============================================================================

def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, ddp_model):
    local_rank = args.local_rank
    print('Distributed Data Parallel Training IFRNet on Rank {}'.format(local_rank))

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

    # Create dataset
    dataset_train = BDD100K_T3_Dataset(
        dataset_dir=args.dataset_dir,
        augment=True,
        crop_size=args.crop_size,
        max_triplets=args.max_triplets
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
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch

    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr_start, weight_decay=1e-4)

    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_geo = AverageMeter()
    avg_dis = AverageMeter()
    best_psnr = 0.0

    if local_rank == 0:
        logger.info(f"Starting training:")
        logger.info(f"  Dataset: {args.dataset_dir}")
        logger.info(f"  Triplets: {len(dataset_train)}")
        logger.info(f"  Batch size: {args.batch_size} x {args.world_size} GPUs = {args.batch_size * args.world_size}")
        logger.info(f"  Iterations per epoch: {args.iters_per_epoch}")
        logger.info(f"  Total epochs: {args.epochs}")

    for epoch in range(args.resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img0, img1, imgt, embt = data

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(optimizer, lr)

            optimizer.zero_grad()

            imgt_pred, loss_rec, loss_geo, loss_dis = ddp_model(img0, img1, embt, imgt)

            loss = loss_rec + loss_geo + loss_dis
            loss.backward()
            optimizer.step()

            avg_rec.update(loss_rec.cpu().data)
            avg_geo.update(loss_geo.cpu().data)
            avg_dis.update(loss_dis.cpu().data)
            train_time_interval = time.time() - time_stamp

            if (iters+1) % 100 == 0 and local_rank == 0:
                logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e}'.format(
                    epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch, 
                    data_time_interval, train_time_interval, lr, 
                    avg_rec.avg, avg_geo.avg, avg_dis.avg))
                avg_rec.reset()
                avg_geo.reset()
                avg_dis.reset()

            iters += 1
            time_stamp = time.time()

        # Save checkpoint every eval_interval epochs
        if (epoch+1) % args.eval_interval == 0 and local_rank == 0:
            checkpoint_path = os.path.join(log_path, f'{args.model_name}_epoch{epoch+1}.pth')
            torch.save(ddp_model.module.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Also save as latest
            latest_path = os.path.join(log_path, f'{args.model_name}_latest.pth')
            torch.save(ddp_model.module.state_dict(), latest_path)

        dist.barrier()

    # Final save
    if local_rank == 0:
        final_path = os.path.join(log_path, f'{args.model_name}_final.pth')
        torch.save(ddp_model.module.state_dict(), final_path)
        logger.info(f"Training complete! Final model saved: {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IFRNet for t+3 prediction')
    parser.add_argument('--model_name', default='IFRNet', type=str, 
                        help='IFRNet, IFRNet_L, IFRNet_S')
    parser.add_argument('--dataset_dir', type=str,
                        default='/storage_drive_0/vivek/video/datasets/bdd100k_t3_triplets',
                        help='Path to preprocessed t+3 triplets')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--world_size', default=3, type=int)
    parser.add_argument('--distributed', action='store_true', default=True,
                        help='Use distributed training')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--max_triplets', default=None, type=int,
                        help='Max triplets to use (for faster training). None = use all.')
    parser.add_argument('--eval_interval', default=3, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--crop_size', default=512, type=int,
                        help='Training crop size (use 512 for best quality)')
    parser.add_argument('--lr_start', default=2e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint_bdd100k_t3', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help='Path to pretrained model for fine-tuning')
    args = parser.parse_args()

    # Get local rank from environment (set by torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.local_rank = local_rank
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    args.device = torch.device('cuda', local_rank)
    args.world_size = dist.get_world_size()

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Load model
    if args.model_name == 'IFRNet':
        from models.IFRNet import Model
    elif args.model_name == 'IFRNet_L':
        from models.IFRNet_L import Model
    elif args.model_name == 'IFRNet_S':
        from models.IFRNet_S import Model
    
    args.log_path = args.log_path + '/' + args.model_name
    args.num_workers = min(8, os.cpu_count())

    model = Model().to(args.device)
    
    # Load pretrained weights for fine-tuning
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from: {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    # Resume from checkpoint
    if args.resume_epoch != 0 and args.resume_path:
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
        
    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    train(args, ddp_model)
    
    dist.destroy_process_group()

