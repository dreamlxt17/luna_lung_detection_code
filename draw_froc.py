# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

RESIZE_SPACING = [1, 1, 1]

def seq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n + 1)])
    else:
        return ([])

def world_2_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = abs(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord

def voxel_2_world(voxel_coord, origin, spacing):
    stretched_voxel_coord = voxel_coord * spacing
    world_coord = stretched_voxel_coord + origin
    return world_coord

def draw_circles(shape, cands_values, origin, spacing):
    # create 3-D nodule_mask, pixel value=1
    # make empty matrix, which will be filled with the mask
    image_mask = np.zeros(shape)

    # run over all the nodules in the lungs
    for ca in cands_values:

        # get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4]) / 2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_x, coord_y, coord_z))

        # determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord, origin, spacing)

        # determine the range of the nodule
        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])

        # create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_x + x, coord_y + y, coord_z + z)), origin, spacing)
                    if (np.linalg.norm(image_coord - coords) * RESIZE_SPACING[0]) < radius:
                        image_mask[int(np.round(coords[2])), int(np.round(coords[0])), int(np.round(coords[1]))] = int(1)

    return image_mask

cads = pd.read_csv("/home/didia/didia/luna16/result/csv/annotations.csv")
preds = pd.read_csv('/home/didia/didia/luna16/result/submission_1.csv')

preds_name_list = list(set(preds['seriesuid']))
preds_name_list.sort()
path = '/home/didia/didia/luna16/result/tensor/'


def main():
    spacing_dict = {}
    nodule_dict = {}
    tp = 0
    fp = 0
    truth_num = 0
    pred_num = 0

    for preds_name in np.random.choice(preds_name_list,5):

        # build a dict to store the spacing and origin of same preds_name
        if not preds_name in spacing_dict:
            image_path = path + preds_name + '.npy'
            pred_tensor, _, spacing, origin = np.load(image_path)
            spacing_dict[preds_name] = [spacing, origin, pred_tensor.shape]
            # print spacing

        # 找到　annotaion　中对应name的结核, image_cads.values, 类型为ndarray, 取出所有与preds_name相同的行，[1,2,3]分别对应x, y, z坐标
        if not preds_name in nodule_dict:
            image_cads = cads[cads['seriesuid'] == preds_name]
            nodule_dict[preds_name] = image_cads.values
            # print image_cads.values

        spacing, origin, image_shape = spacing_dict[preds_name]
        cands_values = nodule_dict[preds_name]
        # 获取preds_name对应的结核3D box
        image_mask = draw_circles(image_shape, cands_values, origin[0], spacing[0])

        # 获取preds_name预测到的结核位置，将其填充为只含一个白点的3D box
        image_preds = preds[preds['seriesuid'] == preds_name]
        image_preds_values = image_preds.values

        pred_num += len(image_preds_values)
        truth_num += len(cands_values)

        for values in image_preds_values:
            pred_box = np.zeros(image_shape)-1
            coord = world_2_voxel(values[1:-1], origin, spacing)[0]
            pred_box[int(np.round(coord[2])), int(np.round(coord[0])), int(np.round(coord[1]))] = int(1)

            l=pred_box==image_mask
            ind = np.argwhere(l==True)
            if len(ind):
                tp += 1

            else:
                fp += 1
        print tp
        # break

    print 'predicted nodules: %d' % pred_num
    print 'real nodules: %d' % truth_num
    print 'True positive: %d' %tp
    print 'False Positive: %d' %fp

main()












