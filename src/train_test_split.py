import os
import shutil
import random
import argparse

# Train / Test split for ShapeNet
def main(opt):
    cat_path = os.path.join(opt.src, opt.cat)
    train, test = list(), list()
    for item in os.listdir(cat_path):
        if random.random() < .8:
            train.append(item + '\n')
        else:
            test.append(item + '\n')

    train_file = os.path.join(opt.src, opt.cat + '_train.txt')
    test_file = os.path.join(opt.src, opt.cat + '_test.txt')

    with open(train_file, 'w') as f:
        f.writelines(train)

    with open(test_file, 'w') as f:
        f.writelines(test)


def mv_npy(npy_root, target_root):
    # Train / test/ split
    for split in os.listdir(npy_root):
        sub_split_root = os.path.join(npy_root, split)
        for item in os.listdir(sub_split_root):
            item_name = item.split('.')[0]
            item_path = os.path.join(sub_split_root, item)
            if item_name not in os.listdir(target_root):
                print(item_name)
                continue
            tgt_path = os.path.join(target_root, '{}/models/npy_file.npy'.format(item_name))
            shutil.copyfile(item_path, tgt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--cat', type=str, required=True)

    conf = parser.parse_args()
    mv_npy(conf.src, conf.cat)
    # main(conf)
