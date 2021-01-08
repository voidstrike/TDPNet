import os
import argparse

# ShapeNet view image generation

def check_npoints(path, npoints):
    # Open generated image
    with open(path, 'r') as f:
        while True:
            cur = f.readline().strip()
            if cur.startswith('POINTS'):
                cur = int(cur.split(' ')[1])
                break
    return npoints == cur


def write_ply_header(file_handler, vertex, face, content=None):
    if content is None:
        file_handler.write('ply\n')
        file_handler.write('format ascii 1.0\n')
        file_handler.write('element vertex {}\n'.format(vertex))
        file_handler.write('property float32 x\n')
        file_handler.write('property float32 y\n')
        file_handler.write('property float32 z\n')
        file_handler.write('element face {}\n'.format(face))
        file_handler.write('property list uint8 int32 vertex_indices\n')
        file_handler.write('end_header\n')
    else:
        raise NotImplementedError('Pending operation')


def off2ply(src_root, filename, tgt_root=None, new_name=None):
    if tgt_root is None:
        tgt_root = src_root

    if new_name is None:
        new_name = filename.split('.')[0] + '.ply'

    with open(os.path.join(src_root, filename), 'r') as f:
        firstLine = f.readline().strip()
        if 'OFF' != firstLine:
            # Not a valid OFF - incorrect format
            n_verts, n_faces, _ = tuple([int(s) for s in firstLine[3:].split(' ')])
        else:
            n_verts, n_faces, _ = tuple([int(s) for s in f.readline().strip().split(' ')])

        wf = open(os.path.join(tgt_root, new_name), 'w')
        write_ply_header(wf, n_verts, n_faces)
        wf.writelines(f.readlines())
        wf.close()


def pc_sampling(prefix_root, filename, npoints):
    init_leaf_size = 0.01
    filename_ply = filename.split('.')[0] + '.ply'
    filename_pcd = filename.split('.')[0] + '.pcd'
    os.system('pcl_mesh_sampling -n_samples {} -no_vis_result {} {}'.format(npoints,
                                                                            os.path.join(prefix_root, filename_ply),
                                                                            os.path.join(prefix_root, filename_pcd)))

    while not check_npoints(os.path.join(prefix_root, filename_pcd), npoints):
        init_leaf_size /= 10
        os.system('pcl_mesh_sampling -n_samples {} -leaf_size {} -no_vis_result {} {}'.format(npoints, init_leaf_size,
                                                                                              os.path.join(prefix_root, filename_ply),
                                                                                              os.path.join(prefix_root, filename_pcd)))
    os.remove(os.path.join(prefix_root, filename_ply))
    os.system('pcl_pcd2ply -format 0 {} {}'.format(os.path.join(prefix_root, filename_pcd),
                                                   os.path.join(prefix_root, filename_ply)))
    os.remove(os.path.join(prefix_root, filename_pcd))


def generate_view(root, category):
    cate_path = os.path.join(root, category)
    for item in os.listdir(cate_path):
        image_path = os.path.join(os.path.join(cate_path, item), 'models/images')
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        item_path = os.path.join(os.path.join(cate_path, item), 'models/model_normalized.obj')
        os.system('blender phong.blend --background --python phong.py -- {} {}'.format(item_path, image_path))


def main(opt):
    root, cate = opt.root, opt.category
    generate_view(root, cate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='The root of model path')
    parser.add_argument('--category', type=str, required=True, help='Target category, number|id')
    operators = parser.parse_args()

    main(operators)
