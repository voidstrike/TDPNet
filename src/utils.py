import matplotlib
import torch
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist

from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from math import log, pi


# Most code of this file is borrowed from: https://github.com/stevenygd/PointFlow/blob/master/utils.py

def gaussian_noise(num, width, height):
    return np.random.normal(0.0, 1.0, size=(num, 1, width, height))

# Visualization
def visualize_point_clouds(pts, gtr, idx, pert_order=[0, 1, 2]):
    pts = pts.cpu().detach().numpy()[:, pert_order]
    gtr = gtr.cpu().detach().numpy()[:, pert_order]

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Sample:%s" % idx)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Ground Truth:%s" % idx)
    ax2.scatter(gtr[:, 0], gtr[:, 1], gtr[:, 2], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res

# Original validate function
def validate(model, tloader, image_flag=False):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    for idx, (multi_view, pc, stat) in enumerate(tloader):
        mv = np.stack(multi_view, axis=1).squeeze(1)
        mv = torch.from_numpy(mv)

        multi_view = mv.cuda()

        tr_pc = pc.cuda()

        out_pc = model.reconstruct(multi_view, 2048)

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        all_sample.append(tr_pc)
        all_ref.append(out_pc)

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    result = compute_all_metrics(sample_pcs, ref_pcs, 64, accelerated_cd=True)
    result = {k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in result.items()}

    print("Chamfer Distance  :%s" % cd.item())
    print("Earth Mover Distance :%s" % emd.item())

# Modified validate function for TDPNet
def tdp_validate(model, tloader, image_flag=False, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    for idx, (multi_view, pc, stat, label) in enumerate(tloader):
        mv = np.stack(multi_view, axis=1).squeeze(axis=1)
        mv = torch.from_numpy(mv).float()
        mv = mv.cuda()

        tr_pc = pc.cuda()

        if old_version:
            out_pc = model(mv)
        else:
            out_pc = model(mv, label)

        all_sample.append(tr_pc)
        all_ref.append(out_pc)

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    print("Chamfer Distance  :%s" % cd.item())
    print("Earth Mover Distance :%s" % emd.item())

def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2
