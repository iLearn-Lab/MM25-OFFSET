import numpy as np
import torch
from tqdm import tqdm as tqdm
import torch.nn.functional as F
import argparse
import os
import json
import model as model_offset
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


def test(params, model, testset, category):
    model.eval()
    (test_queries, test_targets, name) = (testset.test_queries, testset.test_targets, category)
    with torch.no_grad():
        all_queries = []
        all_imgs = []
        if test_queries:
            # compute test query features

            imgs = []
            imgs_seg = []
            mods = []
            for t in tqdm(test_queries, disable=False if params.local_rank == 0 else True):
                imgs += [t['source_img_data']]
                imgs_seg += [t['source_img_data_seg']]
                mods += [t['mod']['str']]
                if len(imgs) >= params.batch_size or t is test_queries[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs_seg = [torch.from_numpy(d).float() for d in imgs_seg]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs_seg = torch.stack(imgs_seg).float().cuda()
                    f = model.extract_retrieval_compose(imgs, mods, imgs_seg)
                    f = f.data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    imgs_seg = []
                    mods = []

            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            imgs_seg = []
            logits = []
            for t in tqdm(test_targets, disable=False if params.local_rank == 0 else True):
                imgs += [t['target_img_data']]
                imgs_seg += [t['target_img_data_seg']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs_seg = [torch.from_numpy(d).float() for d in imgs_seg]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs_seg = torch.stack(imgs_seg).float().cuda()
                    imgs = model.extract_retrieval_target(imgs, imgs_seg).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
                    imgs_seg = []
            all_imgs = np.concatenate(all_imgs)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
    
    
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    
    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])

    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

    # compute recalls
    out = []
    for k in [1, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        out += [('{}_r{}'.format(name, k), r)]
    print(out)

    return out


def test_figAll(params, model, testset, category):
    model.eval()
    if category == 'dress':
        (test_queries, test_targets, name) = (testset.test_queries_dress, testset.test_targets_dress, 'dress')
    elif category == 'shirt':
        (test_queries, test_targets, name) = (testset.test_queries_shirt, testset.test_targets_shirt, 'shirt')
    elif category == 'toptee':
        (test_queries, test_targets, name) = (testset.test_queries_toptee, testset.test_targets_toptee, 'toptee')
    # (test_queries, test_targets, name) = (testset.test_queries, testset.test_targets, category)
    with torch.no_grad():
        all_queries = []
        all_imgs = []
        if test_queries:
            # compute test query features

            imgs = []
            imgs_seg = []
            mods = []
            for t in tqdm(test_queries):
                imgs += [t['source_img_data']]
                imgs_seg += [t['source_img_data_seg']]
                mods += [t['mod']['str']]
                if len(imgs) >= params.batch_size or t is test_queries[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs_seg = [torch.from_numpy(d).float() for d in imgs_seg]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs_seg = torch.stack(imgs_seg).float().cuda()
                    f = model.extract_retrieval_compose(imgs, mods, imgs_seg)
                    f = f.data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    imgs_seg = []
                    mods = []

            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            imgs_seg = []
            logits = []
            for t in tqdm(test_targets):
                imgs += [t['target_img_data']]
                imgs_seg += [t['target_img_data_seg']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs_seg = [torch.from_numpy(d).float() for d in imgs_seg]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs_seg = torch.stack(imgs_seg).float().cuda()
                    imgs = model.extract_retrieval_target(imgs, imgs_seg).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
                    imgs_seg = []
            all_imgs = np.concatenate(all_imgs)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
    
    
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    
    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    
    if name != 'birds':
        for i, t in enumerate(test_queries):
            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

    out = []
    for k in [1, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        out += [('{}_r{}'.format(name, k), r)]
    return out

def test_cirr_valset(params, model, testset):
    
    model.eval()
    test_queries, test_targets = testset.val_queries, testset.val_targets
    with torch.no_grad():
        all_queries = []
        all_imgs = []

        if test_queries:
            # compute test query features
            imgs = []
            imgs_seg = []
            mods = []
            for t in tqdm(test_queries, disable=False if params.local_rank == 0 else True):
                imgs += [t['source_img_data']]
                imgs_seg += [t['source_img_data_seg']]
                mods += [t['mod']['str']]
                if len(imgs) >= params.batch_size or t is test_queries[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs_seg = [torch.from_numpy(d).float() for d in imgs_seg]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs_seg = torch.stack(imgs_seg).float().cuda()
                    f = model.extract_retrieval_compose(imgs, mods, imgs_seg)
                    f = f.data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    imgs_seg = []
                    mods = []
            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            imgs_seg = []
            logits = []
            for t in tqdm(test_targets, disable=False if params.local_rank == 0 else True):
                imgs += [t['target_img_data']]
                imgs_seg += [t['target_img_data_seg']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs_seg = [torch.from_numpy(d).float() for d in imgs_seg]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs_seg = torch.stack(imgs_seg).float().cuda()
                    imgs = model.extract_retrieval_target(imgs, imgs_seg).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
                    imgs_seg = []
            all_imgs = np.concatenate(all_imgs)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])

    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
    
    
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)


    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


    nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])] # (m,n)

    # all set recalls
    cirr_out = []
    for k in [1, 5, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        cirr_out += [('{}_r{}'.format('cirr',k), r)]

    # subset recalls
    for k in [1, 2, 3]:
        r = 0.0
        for i, nns in enumerate(nn_result):

            subset = np.array([test_targets_id.index(idx) for idx in test_queries[i]['subset_id']]) 
            subset_mask = (nns[..., None] == subset[None, ...]).sum(-1).astype(bool) 
            subset_label = nns[subset_mask] 
            if test_targets_id.index(test_queries[i]['target_img_id']) in subset_label[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        cirr_out += [('{}_subset_r{}'.format('cirr', k), r)]

    return cirr_out


if __name__ == '__main__':
    from transformers import CLIPImageProcessor
    import datasets
    print("Loading dataset")

    import open_clip
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14')
    OFFSET = torch.load(args.model_dir).to('cuda')

    if args.dataset == "shoes":
        shoes_dataset = datasets.Shoes_SavedSegment(path = args.shoes_path, transform = [preprocess_train, preprocess_val])
        t = test(args, OFFSET, shoes_dataset, 'shoes')
        print(t)
    elif args.dataset == "fashioniq":
        testset = datasets.FashionIQ_SavedSegment_all(path = args.fashioniq_path, transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        for ci, category in enumerate(['dress', 'shirt', 'toptee']):
            t = test_figAll(args, OFFSET, testset, category)
            print(t)
    elif args.dataset == "cirr":
        cirr_dataset = datasets.CIRR_SavedSegment(path = args.cirr_path, transform = [preprocess_train, preprocess_val])
        t = test_cirr_valset(args, OFFSET, cirr_dataset)
        print(t)
    
