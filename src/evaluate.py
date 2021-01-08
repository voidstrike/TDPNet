import argparse
import torch
import os
import time
import imageio
import numpy as np
import torchvision.transforms as tfs
import sklearn.cluster as cls

from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable


from datasets.mv_dataset import MultiViewDataSet, ShapeNet55
from utils import visualize_point_clouds, tdp_validate, gaussian_noise

# Loading model code
from model.TDPNet import TDPNet

# Transformation for ModelNet and ShapeNet
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

def one_forward(model, model_name, mv, label_info=None):
    mv = np.stack(mv, axis=1).squeeze(axis=1)
    mv = torch.from_numpy(mv).float()
    mv = Variable(mv.cuda())

    syn_pc = model(mv)

    return syn_pc


def main(conf):
    # Load 3D Prototype features -- dummy
    proto_corpus = np.zeros((conf.num_prototypes, 1024), dtype=np.float)

    # Basic setting, create checkpoint folder, initialize dataset, dataloaders and models
    checkpoint_path = os.path.join(conf.model_path, opt.name)
    checkpoint_imgs = os.path.join(checkpoint_path, 'eval_images')

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.exists(checkpoint_imgs):
        os.mkdir(checkpoint_imgs)

    root, ply_root, tgt_category = conf.root, conf.proot, conf.cat
    tgt_category = None if conf.cat == 'none' else conf.cat
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

    ds_loader = DataLoader(mv_ds, batch_size=conf.batch_size, drop_last=True, shuffle=False)
    ds_loader_test = DataLoader(mv_ds_test, batch_size=conf.batch_size, shuffle=False)

    # Initialize Model

    model = TDPNet(conf, proto_corpus)
    # model = TDPNet(conf, 10, proto_corpus)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, f'{conf.name}_iter_fin.pt')))

    model.cuda()

    print('Start Evaluate 2D to 3D -------------------------------------------')

    start_time = time.time()

    with torch.no_grad():
        tdp_validate(model, ds_loader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size')
    parser.add_argument('--workers', type=int, help='Number of data loading workers, (default=4)', default=4)
    parser.add_argument("--random_seed", action="store_true", help="Fix random seed or not")
    parser.add_argument('--device', type=str, default='cuda', help='GPU usage')
    parser.add_argument('--dim_template', type=int, default=2, help='Template dimension')

    # Data
    parser.add_argument('--number_points', type=int, default=2048,
                        help='Number of point sampled on the object during training, and generated by atlasnet')

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
    parser.add_argument("--no_metro", action="store_true", help="Compute metro distance")

    # Additional arguments
    parser.add_argument('--root', type=str, required=True, help='The path of multi-view dataset')
    parser.add_argument('--proot', type=str, required=True, help='The path of corresponding pc dataset')
    parser.add_argument('--cat', type=str, required=True, help='Target category')
    parser.add_argument('--model_path', type=str, default='../checkpoint/')

    parser.add_argument('--dataset', type=str, default='modelnet', help='The testing dataset, chose from [modelnet|shapenet]')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use')

    opt = parser.parse_args()
    main(opt)