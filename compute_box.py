# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import cPickle as pickle
import glob
import gzip

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
cad_list = pd.read_csv("/home/didia/didia/luna16/result/csv/annotations.csv")
seriesuid_list = pd.read_csv('/home/didia/didia/luna16/evaluationScript/annotations/seriesuids.csv')
path_list = glob.glob('/home/didia/didia/luna16/result/tensor/*')
path_list.sort()

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


coord_list = cad_list[[coordX_label, coordY_label, coordZ_label]]
max_dist = 0
max_dir = 0
min_dir = 512

def main():
    for i, path in enumerate(path_list):
        print i
        seriesuid = path.split('/')[-1].replace('.npy', '')
        # print seriesuid
        _, _, spacing, origin = np.load(path)
        # print spacing, origin
        cad_value = cad_list[cad_list[seriesuid_label]==seriesuid].values      # 读取每个seriesuid对应的矩阵数据，可能是n行５列，　第一列是病人序列号，第２－４列是坐标，第５列是nodule直径
        for cad in cad_value:
            radius = np.ceil(cad[4]) / 2   # 取半径
            # print 'radius:', radius
            voxel_coord = world_2_voxel([cad[1], cad[2], cad[3]], origin, spacing)[0]
            # 获取欧式距离
            # distance = np.linalg.norm([256,256] - voxel_coord[:-1])
            # if max_dist < distance:
            #     max_dist = distance
            #     print voxel_coord[:-1]
            #     print distance

            # 获取最大坐标和最小坐标
            max1 = np.round((voxel_coord + radius)[:-1].max())
            min1 = np.round((voxel_coord - radius)[:-1].min())
            if max_dir < max1:
                max_dir = max1
                print 'min_dir', min_dir

            if min_dir > min1:
                min_dir = min1
                print 'max_dir', max_dir

    print  max_dir, min_dir


def main2():
    ''' 统计结核边界位置'''
    for subset in range(10):
    # for subset in [7]:

        max = 0
        min = 500
        tmp = 0
        subset_path = '/home/didia/didia/data/1_1_1mm_slices_nodule/subset{}/*'.format(subset)
        image_paths = glob.glob(subset_path)
        for i, path in enumerate(image_paths):

            file = gzip.open(path,'rb')
            l_slice = pickle.load(file)
            location = np.argwhere(l_slice==1)

            max_now = location.max()
            min_now = location.min()
            if min_now < 94:
                print 'danger!',min_now
                tmp+=1
            if max < max_now:
                max = max_now
            if min > min_now:
                min = min_now
            file.close()
        print max, min, tmp

# main2()

import matplotlib.pyplot as plt
cad_list = pd.read_csv("/home/didia/didia/luna16/result/csv/annotations.csv")

def count_nodule_radius():
    '''统计结核半径大小分布'''
    diameter_list = list(cad_list['diameter_mm'])
    diam = np.round(diameter_list)
    di_num_list = []
    for di in set(diam):
        di_num = len(np.argwhere(diam==di))
        di_num_list.append(di_num)
    print zip(list(set(diam)), di_num_list)

    plt.bar(list(set(diam)), di_num_list)
    plt.xlabel('diameter_mm')
    plt.ylabel('number')
    plt.show()

# 直方图
# cad_list['diameter_mm'].hist(bins=100).get_figure().savefig('/home/didia/tmp.png')

cad_list['diameter_mm'].plot()
plt.legend(loc='best')
plt.show()








# count_nodule_radius()