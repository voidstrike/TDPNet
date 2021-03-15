import argparse
import torch
import os
import time
import imageio
import numpy as np
import torchvision.transforms as tfs
import sklearn.cluster as cls

from model.TDPNet import TDPNet
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from collections import defaultdict


from datasets.mv_dataset import MultiViewDataSet, ShapeNet55
from metrics.evaluation_metrics import distChamferCUDA
from utils import visualize_point_clouds, tdp_validate
from pointnet.model import PointNetfeat

_transform = tfs.Compose([
    tfs.CenterCrop(550),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])

_transform_shape = tfs.Compose([
    tfs.CenterCrop(256),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])


def main(conf):
    # Load 3D Prototype features
    if conf.prototypes_npy != 'NOF':
        proto_corpus = np.load(conf.prototypes_npy)
        assert proto_corpus.shape[0] == conf.num_prototypes
    else:
        if not conf.reclustering:
            raise RuntimeError('Prototypes are not provided, must re-clustering or train from scratch')

        proto_corpus = np.zeros((conf.num_prototypes, 1024), dtype=np.float)

    # Basic setting, make checkpoint folder, initialize dataset, dataloaders and models
    checkpoint_path = os.path.join(conf.model_path, opt.name)
    checkpoint_imgs = os.path.join(checkpoint_path, 'images')

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.exists(checkpoint_imgs):
        os.mkdir(checkpoint_imgs)

    root, ply_root, tgt_category = conf.root, conf.proot, conf.cat
    tgt_category = tgt_category

    if conf.dataset == 'modelnet':
        mv_ds = MultiViewDataSet(root, ply_root, 'train', transform=_transform,
                                 sub_cat=tgt_category, number_of_view=1)
        mv_ds_test = MultiViewDataSet(root, ply_root, 'test', transform=_transform,
                                      sub_cat=tgt_category, number_of_view=1)
    elif conf.dataset == 'shapenet':
        mv_ds = ShapeNet55(root, tgt_category, 'train', transform=_transform_shape)
        mv_ds_test = ShapeNet55(root, tgt_category, 'test', transform=_transform_shape)
    else:
        raise RuntimeError(f'Dataset is suppose to be [modelnet|shapenet], but {conf.dataset} is given')

    ds_loader = DataLoader(mv_ds, batch_size=conf.batch_size, drop_last=True, shuffle=True)
    ds_loader_test = DataLoader(mv_ds_test, batch_size=conf.batch_size)
    num_classes = len(mv_ds.classes)

    print(f'Dataset summary : Categories: {mv_ds.classes} with length {len(mv_ds)}')
    print(f'Num of classes is {len(mv_ds.classes)}')

    # Initialize Model
    model = TDPNet(conf, proto_corpus)
    point_feat_extractor = PointNetfeat()  # Required for re-clustering

    if conf.from_scratch:
        # In this branch, we have to train the point_feat_extractor from scratch, which is a pc self-reconstruction task
        model.cuda()
        point_feat_extractor.cuda()

        # Fixed optimizer
        pre_optimizer = Adam(
            list(model.parameters()) + list(point_feat_extractor.parameters()),
            lr=1e-3,
            betas=(.9, .999),
        )

        print('Start Training 3D self-reconstruction-------------------------')

        for i in range(100):
            total_loss = 0.
            print('Start Epoch {}'.format(str(i + 1)))
            for idx, (_, pc, stat, label) in enumerate(ds_loader):
                # Get input image and add gaussian noise
                pc = Variable(pc.transpose(2, 1).cuda())  # BatchSize * 2048 * 3
                pc_feat, _, _ = point_feat_extractor(pc)  # Extract Feature using PointNet

                pre_optimizer.zero_grad()

                syn_pc = model(pc_feat, False)
                ori_pc = pc.transpose(2, 1).contiguous()
                gen2gr, gr2gen = distChamferCUDA(syn_pc, ori_pc)
                cd_loss = gen2gr.mean(1) + gr2gen.mean(1)

                loss = cd_loss.sum()
                total_loss += loss.detach().item()

                loss.backward()
                pre_optimizer.step()

            print('Epoch {}  -- Recon CD {}'.format(str(i + 1), total_loss / float(len(mv_ds))))

        # Saving trained network so we can skip this part in the future
        print(f'Saving models at {checkpoint_path}')
        torch.save(point_feat_extractor.state_dict(),
                   os.path.join(checkpoint_path, 'pretrained_point_encoder.pt'))
        torch.save(model.state_dict(),
                   os.path.join(checkpoint_path, 'pretrained_point_decoder.pt'))
    else:
        print('Training from a pretrained encoder-decoder, loading pretrained models')
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'pretrained_point_decoder.pt')))
        model.cuda()

    if conf.reclustering:
        print(f'Clustering from scratch, the number of cluster centroids would be {conf.num_prototypes}')
        point_feat_extractor.load_state_dict(torch.load(os.path.join(checkpoint_path, 'pretrained_point_encoder.pt')))
        point_feat_extractor.cuda()
        point_feat_extractor.eval()

        tmp_ds = MultiViewDataSet(root, ply_root, 'train', transform=_transform, sub_cat=tgt_category, number_of_view=1)
        corpus_builder = DataLoader(tmp_ds, batch_size=1)

        feature_list = list()

        for idx, (_, pc, _, _) in enumerate(corpus_builder):
            with torch.no_grad():
                pc = Variable(pc.transpose(2, 1).cuda())
                point_feat, _, _ = point_feat_extractor(pc)
                feature_list.append(point_feat.detach().squeeze().cpu().numpy())

        # K-Means Clustering
        feature_list = np.asarray(feature_list)

        operator = cls.KMeans(n_clusters=conf.num_prototypes, random_state=0).fit(feature_list)
        proto_corpus = operator.cluster_centers_
        model.update_prototypes(proto_corpus)

    print('Start Training 2D to 3D -------------------------------------------')

    optimizer = Adam(
        model.parameters(),
        lr=conf.lrate,
        betas=(.9, .999),
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(conf.nepoch / 3), gamma=.5)

    start_time = time.time()

    for i in range(conf.nepoch):
        total_loss = 0.
        print('Start Epoch {}'.format(str(i + 1)))
        if i == min(50, int(conf.nepoch / 3)):
            print('Activated prototype finetune')
            model.activate_prototype_finetune()

        for idx, (multi_view, pc, _, _) in enumerate(ds_loader):
            # Get input image and add gaussian noise
            mv = np.stack(multi_view, axis=1).squeeze(axis=1)
            mv = torch.from_numpy(mv).float()

            mv = Variable(mv.cuda())
            pc = Variable(pc.cuda())  # BatchSize * 2048, currently

            # Optimize process
            optimizer.zero_grad()

            syn_pc = model(mv)
            gen2gr, gr2gen = distChamferCUDA(syn_pc, pc)
            cd_loss = gen2gr.mean(1) + gr2gen.mean(1)

            loss = cd_loss.sum()

            total_loss += loss.detach().item()

            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print(
                    'Epoch %d Batch [%2d/%2d] Time [%3.2fs] Recon Nat %.10f' %
                    (i + 1, idx + 1, len(ds_loader), duration, loss.item() / float(conf.batch_size)))

        print('Epoch {}  -- Recon Nat {}'.format(str(i + 1), total_loss / float(len(mv_ds))))

        # Save model configuration
        if conf.save_interval > 0 and i % opt.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_path,
                                                        '{0}_iter_{1}.pt'.format(conf.name, str(i + 1))))

        # Validate the model on test split
        if conf.sample_interval > 0 and i % opt.sample_interval == 0:
            with torch.no_grad():
                tdp_validate(model, ds_loader_test)
            model.train()

        scheduler.step()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=101, help='number of epochs to train for')
    parser.add_argument('--random_seed', action="store_true", help='Fix random seed or not')
    parser.add_argument('--lrate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=120, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=140, help='learning rate decay 2')
    parser.add_argument('--lr_decay_3', type=int, default=145, help='learning rate decay 2')
    parser.add_argument('--device', type=str, default='cuda', help='Gpu usage')
    parser.add_argument('--dim_template', type=int, default=2, help='Template dimension')

    # Data
    parser.add_argument('--number_points', type=int, default=2048,
                        help='Number of point sampled on the object during training, and generated by atlasnet')
    parser.add_argument('--prototypes_npy', type=str, default='NOF', help='Path of the prototype npy file')

    # Save dirs and reload
    parser.add_argument('--name', type=str, default="0", help='training name')
    parser.add_argument('--dir_name', type=str, default="", help='name of the log folder.')

    # Network
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
    parser.add_argument('--nb_primitives', type=int, default=1, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SQUARE", choices=["SPHERE", "SQUARE"],
                        help='dim_out_patch')

    parser.add_argument('--bottleneck_size', type=int, default=1536, help='dim_out_patch')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='dim_out_patch')
    parser.add_argument('--num_prototypes', type=int, default=8, help='Number of prototypes')
    parser.add_argument('--num_slaves', type=int, default=4, help='Number of slave mlps per prototype')

    # Loss
    parser.add_argument('--no_metro', action="store_true", help='Compute metro distance')

    # Additional arguments
    parser.add_argument('--root', type=str, required=True, help='The path of multi-view dataset')
    parser.add_argument('--proot', type=str, required=True, help='The path of corresponding pc dataset')
    parser.add_argument('--cat', type=str, required=True, help='Target category')
    parser.add_argument('--model_path', type=str, default='../checkpoint')

    parser.add_argument('--sample_interval', type=int, default=10, help='The gap between each sampling process')
    parser.add_argument('--save_interval', type=int, default=20, help='The gap between each model saving')

    parser.add_argument('--from_scratch', action="store_true", help='Train the point_feature_extractor from scratch')
    parser.add_argument('--reclustering', action="store_true", help='Flag that controls the re-clustering behavior')
    parser.add_argument('--dataset', type=str, default='modelnet', help='The dataset to use, chose from [modelnet|shapenet]')

    opt = parser.parse_args()
    main(opt)
