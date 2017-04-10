# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import glob
from time import time

def voxel_2_world(voxel_coord, origin, spacing):
    stretched_voxel_coord = voxel_coord * spacing
    world_coord = stretched_voxel_coord + origin
    return world_coord

def normalize_probability(index, conv_out):
    pro = [conv_out[ind[2], ind[0], ind[1]] for ind in index]
    max = np.max(pro)
    min = np.min(pro)
    normed_pro = [(p-min)/(max-min) for p in pro]
    return normed_pro

m=4  # conv
n=19 # pool
threshold = 0.495

def conv_pool(path):
    # data = np.load('/home/didia/didia/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.npy')
    data = np.load(path)

    pred_tensor, label_tensor, spacing, origin = data
    l= len(pred_tensor)

    pred_tensor = np.reshape(pred_tensor, [-1, l, 512, 512, 1])
    input = tf.placeholder("float", [None, l, 512, 512, 1])
    # 设置w为nxnxn的常量
    w = tf.constant(value=1.0/(m*m*m), shape=[m,m,m,1,1], dtype='float')

    conv_out = tf.nn.conv3d(input, w, strides=[1, 1, 1, 1, 1], padding='SAME')
    pool_out = tf.nn.max_pool3d(conv_out, ksize=[1,n,n,n,1], strides=[1,1,1,1,1],padding='SAME')
    time1 = time()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        time2 = time()
        print 'start',time2 - time1


        tf.global_variables_initializer().run()
        conv_out, pool_out = sess.run([conv_out, pool_out], feed_dict={input: pred_tensor})
        conv_out = np.reshape(conv_out, [-1,512,512])
        pool_out= np.reshape(pool_out, [-1,512,512])
        time3 = time()
        print 'conv',time3 - time1

        pool_out[pool_out<threshold] = -1
        coord_list = conv_out==pool_out
        candidates = np.argwhere(coord_list==True)

        # candidates= np.array([list(ind) for ind in index if conv_out[ind[0], ind[1], ind[2]]>threshold])
        # print len(index), len(candidates)
        # print candidates

        candidates = np.transpose(np.stack([candidates[:,1], candidates[:,2], candidates[:,0]]), [1,0])
        time3 = time()
        print 'cand',time3 - time1
        # print candidates

        patname = path.split('/')[-1]
        fun = voxel_2_world
        world_candids = [[patname, fun(ind, origin, spacing), conv_out[ind[2], ind[0], ind[1]]] for ind in candidates]
        # normed_pro = normalize_probability(candidates, conv_out)
        # print len(normed_pro)
        # world_candids = [[patname, fun(ind, origin, spacing), normed_pro[i]] for i,ind in enumerate(candidates)]
        # print world_candids


        # np.save(path.replace('tensor', 'candi_voxel_coord'), candidates)     # 保存voxel坐标 及概率
        # np.save(path.replace('tensor', 'candi_world_coord'), world_candids)    # 保存world坐标 及概率
        time3 = time()
        print time3 - time1

        fname = '/home/didia/didia/luna16/result/submission_0_495_4_16.csv'
        with open(fname,'a') as f:
            for array in world_candids[1:]:
                # print array
                name,coords,prob = array
                x,y,z = coords[0]
                f.write(name.replace('.npy',''))
                f.write(',')
                f.write(str(x))
                f.write(',')
                f.write(str(y))
                f.write(',')
                f.write(str(z))
                f.write(',')
                f.write(str(prob))
                f.write('\n')
        # raw_input('end')
        time3 = time()
        print time3 - time1



path_list = glob.glob('/home/didia/didia/luna16/result/tensor/*')
path_list.sort()

if __name__ == '__main__':

    fname = '/home/didia/didia/luna16/result/submission_0_495_4_16.csv'
    with open(fname, 'w') as f:
        f.write('')
    for i, path in enumerate(path_list):
        print i
        conv_pool(path)







def plot_pred(path):
    data = np.load(path)
    candidates = np.load('conv_pool_pred.npy')
    world_coord = np.load('world_coord.npy')

    # print candidates
    print world_coord

    print len(candidates[:,0])
    pred_tensor, label_tensor, spacing, origin = data

    pred_tensor = np.zeros([349, 512,512])
    for ind in candidates:
        pred_tensor[list(ind)[0], list(ind)[1], list(ind)[2]]=1

    for i in candidates[:,0][400:500]:

        im1 = pred_tensor[i]
        im2 = label_tensor[i]
        plt.imshow(im1, cmap='gray')
        plt.figure()
        plt.imshow(im2, cmap='gray')
        plt.show()

