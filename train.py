import os 
import argparse
import logging
import warnings 
import random

import numpy as np 
import torch 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader
from tqdm import tqdm

import open_clip 
import utils
import datasets
import model_OFFSET as model
import test

from torch.cuda.amp import autocast as autocast, GradScaler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
# mp.set_start_method('spawn')

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default = 'dress', help = "data set type")
parser.add_argument('--fashioniq_split', default = 'val-split')
parser.add_argument('--fashioniq_path', default = "")
parser.add_argument('--shoes_path', default = "")
parser.add_argument('--cirr_path', default = "")

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--eps', type=float)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--hidden_dim', type=int, default=512)

parser.add_argument('--P', type=int)
parser.add_argument('--Q', type=int)
parser.add_argument('--tau_', type=float)
parser.add_argument('--mu_', type=float)

parser.add_argument('--seed', type=int, default=42)   
parser.add_argument('--lr', type=float, default=1e-4) 
parser.add_argument('--clip_lr', type=float, default=1e-5) 

parser.add_argument('--backbone', type=str)

parser.add_argument('--lr_decay', type=int, default=8)
parser.add_argument('--lr_div', type=float, default=0.1)  
parser.add_argument('--max_decay_epoch', type=int, default=10)  
parser.add_argument('--tolerance_epoch', type=int, default=5)
parser.add_argument('--ifSave', type=int, default=0)

 
parser.add_argument('--model_dir', default='./',
                    help="Directory containing params.json")

parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--i', type=str, default='0')

args = parser.parse_args()


def load_dataset():
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14')
    
    if args.dataset in ['dress', 'shirt', 'toptee']:
        print('Loading FashionIQ-{} dataset'.format(args.dataset))
        fashioniq_dataset = datasets.FashionIQ_Segment(path = args.fashioniq_path, category = args.dataset, transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        return [fashioniq_dataset]
    elif args.dataset == 'fashioniq':
        print('Reading fashioniq')
        fashioniq_dataset = datasets.FashionIQ_SavedSegment_all(path = args.fashioniq_path, transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        return [fashioniq_dataset]
    elif args.dataset == 'shoes':
        print('Reading shoes')
        shoes_dataset = datasets.Shoes_SavedSegment(path = args.shoes_path, transform = [preprocess_train, preprocess_val])
        return [shoes_dataset]
    elif args.dataset == 'cirr':
        print('Reading cirr')
        cirr_dataset = datasets.CIRR_SavedSegment(path = args.cirr_path, transform = [preprocess_train, preprocess_val])
        return [cirr_dataset]



def set_bn_eval(m): 
    classname = m.__class__.__name__ 
    if classname.find('BatchNorm2d') != -1: 
        m.eval() 

def create_model_and_optimizer():
    OFFSET = model.OFFSET(hidden_dim=args.hidden_dim, dropout=args.dropout_rate, local_token_num=args.Q, global_token_num = args.P, t = args.tau_)
    OFFSET.cuda()

    params = list(OFFSET.named_parameters())
    param_group = [
        {'params': [p for n, p in params if any(nd in n for nd in ['clip'])], 'lr': args.clip_lr},
        {'params': [p for n, p in params if not any(nd in n for nd in ['clip'])], 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(param_group, lr=args.lr, weight_decay = args.weight_decay)
    return OFFSET, optimizer


def train(model, optimizer, dataloader, scaler):
    model.train()
    model.apply(set_bn_eval)
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for i, data in enumerate(dataloader):

            img1 = data['source_img_data'].cuda()
            img1_seg = data['source_img_data_seg'].cuda()
            img2 = data['target_img_data'].cuda()
            img2_seg = data['target_img_data_seg'].cuda()
            mods = data['mod']['str']

            optimizer.zero_grad()
            with autocast():
                loss = model.compute_loss(img1, mods, img2, img1_seg, img2_seg)
                total_loss = \
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()     

            if i % args.save_summary_steps == 0:
                summary_batch = {}
                summary_batch['total_loss'] = total_loss.item()
                summ.append(summary_batch)
            loss_avg.update(total_loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

if __name__ == '__main__':

    # Load the parameters from json file

    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    utils.set_logger(os.path.join(args.model_dir, '{}_{}_train.log'.format(args.dataset, args.i)))
    logging.info('Loading the datasets and model...')
    # fetch dataloaders

    dataset_list = load_dataset()
 
    model, optimizer = create_model_and_optimizer()
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, optimizer, dataset_list)
