from torch.utils.data.dataset import Dataset
import os
import torch
from PIL import Image

from torch.utils.data import DataLoader
import torchvision.transforms as tfs
import numpy as np


# Pre-defined ShapeNet id2cat mapping
# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
}


# PLY file handler
def ply_reader(file_path):
    n_verts = 2048
    with open(file_path, 'r') as f:
        while True:
            cur = f.readline().strip()
            if cur == 'end_header':
                break

            cur = cur.split(' ')
            if len(cur) > 2 and cur[1] == 'vertex':
                n_verts = min(int(cur[2]), n_verts)

        vertices = [[float(s) for s in f.readline().strip().split(' ')] for _ in range(n_verts)]

    return vertices


class MultiViewDataSet(Dataset):
    def find_classes(self, dir):
        classes = self.target_label
        classes_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, classes_to_idx

    def __init__(self, root, ply_root, data_type, loader=ply_reader, transform=None, tgt_transform=None,
                 data_augment=False, sub_cat=None, number_of_view=1, number_of_points=2048):
        self.x, self.y = list(), list()
        self.z = list()
        self.label = list()
        self.root = root
        self.ply_root = ply_root
        self.num_of_view = number_of_view
        self.num_of_points = number_of_points

        self.loader = loader
        self.data_augment = data_augment
        if not sub_cat:
            raise NotImplementedError('Single Category Mode, target class must be specified')
        else:
            self.target_label = [sub_cat]

        self.classes, self.class_to_idx = self.find_classes(root)

        self.tfs, self.tgt_tfs = transform, tgt_transform

        # Current Dataset structure: root / <label> / <train/test> / <item> / <view>.png
        for label in os.listdir(root):
            if label not in self.target_label:
                continue

            c_path = os.path.join(os.path.join(root, label), data_type)
            ply_path = os.path.join(os.path.join(ply_root, label), data_type)
            for item in os.listdir(c_path):
                cc_path = os.path.join(c_path, item)
                ply_item_path = os.path.join(ply_path, item.replace('.off', '.ply'))
                views = list()
                for view in os.listdir(cc_path):
                    views.append(os.path.join(cc_path, view))
                self.x.append(views)
                self.z.append(self.class_to_idx[label])
                self.y.append(ply_item_path)
                self.label.append(self.class_to_idx[label])

    def __getitem__(self, index):
        view_paths = self.x[index]
        views = list()

        for view in view_paths[:self.num_of_view]:
            im = Image.open(view)
            im = im.convert('RGB')
            if self.tfs is not None:
                im = self.tfs(im)
            views.append(im)

        sample = self.loader(self.y[index])
        point_set = np.asarray(sample, dtype=np.float32)

        if point_set.shape[0] < self.num_of_points:
            choice = np.random.choice(len(point_set), self.num_of_points - point_set.shape[0], replace=True)
            aux_pc = point_set[choice, :]
            point_set = np.concatenate((point_set, aux_pc))

        # Rescale the point cloud into a unit ball
        center_point = np.expand_dims(np.mean(point_set, axis=0), 0)
        point_set = point_set - center_point  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augment:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set).contiguous()

        return views, point_set, self.z[index], self.label[index]

    def __len__(self):
        return len(self.x)


# Class for ShapeNet 55
class ShapeNet55(Dataset):
    def __init__(self, root, category, split, transform=None, tgt_transform=None,
                 data_augment=False, number_of_view=1):
        self.root = root
        self.item_root = os.path.join(root, category)
        self.config = os.path.join(root, '{}_{}.txt'.format(category, split))
        self.num_of_view = number_of_view
        self.tfs, self.tgt_tfs = transform, tgt_transform
        self.data_augment = data_augment
        self.x, self.y = list(), list()

        with open(self.config, 'r') as f:
            lines = f.readlines()
            for filename in lines:
                filename = filename.strip()
                item_path = os.path.join(os.path.join(self.item_root, filename), 'models')
                npy_file = os.path.join(item_path, 'npy_file.npy')
                # print(npy_file)
                view_root = os.path.join(item_path, 'images')
                if not os.path.exists(npy_file):
                    # print('Error')
                    continue

                views = list()
                for view in os.listdir(view_root):
                    views.append(os.path.join(view_root, view))

                self.y.append(npy_file)
                self.x.append(views)

        self.pc_data = list()
        for idx in range(len(self.y)):
            try:
                pc = np.load(self.y[idx])
            except:
                raise Exception('Unexpected Error!')

            choice = np.random.choice(15000, 2048)
            pc = pc[choice, :]
            self.pc_data.append(pc)

    def __getitem__(self, index):
        view_paths = self.x[index]
        views = list()

        for view in view_paths[:self.num_of_view]:
            im = Image.open(view)
            im = im.convert('RGB')
            if self.tfs is not None:
                im = self.tfs(im)
            views.append(im)

        sample = self.pc_data[index]
        point_set = np.asarray(sample, dtype=np.float32)

        # Rescale the point cloud into a unit ball
        center_point = np.expand_dims(np.mean(point_set, axis=0), 0)
        point_set = point_set - center_point  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        stat = (center_point, dist)

        if self.data_augment:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set)

        return views, point_set, stat

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    pass
