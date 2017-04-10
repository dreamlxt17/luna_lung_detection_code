# encoding: utf-8
#!/usr/bin/env python

import logging
from params import params as P
import numpy as np
from parallel import ParallelBatchIterator
import dataset
import os
from tqdm import tqdm

import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import glob
from collections import OrderedDict

def output_size_for_input(in_size, depth):
    in_size -= 4
    for _ in range(depth-1):
        in_size = in_size//2
        in_size -= 4
    for _ in range(depth-1):
        in_size = in_size*2
        in_size -= 4
    return in_size

NET_DEPTH = P.DEPTH  # Default 5
INPUT_SIZE = P.INPUT_SIZE  # Default 512
OUTPUT_SIZE = output_size_for_input(INPUT_SIZE, NET_DEPTH)
epoch_num = 100
batch_size = 2

def filter_for_depth(depth):
    return 2**(P.BRANCHING_FACTOR+depth)

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def unpool(x, scope='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param x: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.variable_scope(scope):
        sh = x.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(x, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            # out = tf.concat(i, [out, tf.zeros_like(out)])
            out = tf.concat([out, out],i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

def get_file_names():
    # get list of subset-name
    path_list = glob.glob(data_path+'*')
    path_list.sort()
    tr_file=[]
    te_file=[]
    for subset in path_list[:8]:
        tr_file.extend(glob.glob(subset+'/*'))
    for subset in path_list[8:]:
        te_file.extend(glob.glob(subset+'/*'))
    return tr_file, te_file

def define_network(input_var):

    net = OrderedDict()
    net['input'] = input_var
    nonlinearity = tf.nn.relu

    if P.GAUSSIAN_NOISE > 0:
        net['input'] = gaussian_noise_layer(net['input'], std=P.GAUSSIAN_NOISE)

    def contraction(depth, deepest):
        # downsampling with kernel(3,3)
        # iter_time = depth - 1
        # each iter: 2 conv(3,3)valid + 1 maxpool(2,2) + 1 batch_normalization

        n_filters = filter_for_depth(depth)
        incoming = net['input'] if depth == 0 else net['pool{}'.format(depth-1)]

        # tflayers.conv2d returns a tensor representing the output of the operation.
        net['conv{}_1'.format(depth)] = tflayers.conv2d(incoming, num_outputs=n_filters, kernel_size=3, padding='same', activation_fn=nonlinearity)

        net['conv{}_2'.format(depth)] = tflayers.conv2d(net['conv{}_1'.format(depth)], n_filters, kernel_size=3, padding='same', activation_fn=nonlinearity)

        if P.BATCH_NORMALIZATION:
            net['conv{}_2'.format(depth)] = tflayers.batch_norm(inputs=net['conv{}_2'.format(depth)], center= False,
                                                                scale=True, param_initializers=[P.BATCH_NORMALIZATION_ALPHA, None])
        if not deepest:
            net['pool{}'.format(depth)] = tflayers.max_pool2d(net['conv{}_2'.format(depth)], kernel_size=2, stride=2)

    def expansion(depth, deepest):
        # upsampling with kernel(2,2)
        # iter_time = depth - 1
        # each iter:  2 conv(3,3)pad=0 + 1 upconv(2,2)

        n_filters = filter_for_depth(depth)
        incoming = net['conv{}_2'.format(depth+1)] if deepest else net['_conv{}_2'.format(depth+1)]
        upscaling = unpool(incoming)

        net['upconv{}'.format(depth)] = tflayers.conv2d(upscaling, num_outputs=n_filters, kernel_size=3, activation_fn=nonlinearity)

        if P.SPATIAL_DROPOUT > 0:
            bridge_from = tflayers.dropout(net['conv{}_2'.format(depth)], keep_prob=P.SPATIAL_DROPOUT)

        else:
            bridge_from = net['conv{}_2'.format(depth)]

        net['bridge{}'.format(depth)] = tf.concat([net['upconv{}'.format(depth)], bridge_from], 3)

        net['_conv{}_1'.format(depth)] = tflayers.conv2d(net['bridge{}'.format(depth)], n_filters, kernel_size=3, padding='same',
                                                         activation_fn=nonlinearity)

        if P.BATCH_NORMALIZATION:
            net['_conv{}_1'.format(depth)] = tflayers.batch_norm(inputs=net['_conv{}_1'.format(depth)], center= False,
                                                               scale=True, param_initializers=[P.BATCH_NORMALIZATION_ALPHA, None])
        if P.DROPOUT > 0:
            net['_conv{}_1'.format(depth)] = tflayers.dropout(net['_conv{}_1'.format(depth)], P.DROPOUT)

        net['_conv{}_2'.format(depth)] = tflayers.conv2d(net['_conv{}_1'.format(depth)], n_filters, kernel_size=3, padding='same', activation_fn=nonlinearity)

    for d in range(NET_DEPTH):  # [0,1,2,3,4]
        # No pooling at the last layer
        deepest = d == NET_DEPTH-1
        contraction(d, deepest)

    for d in reversed(range(NET_DEPTH-1)):   # [3,2,1,0]
        deepest = d == NET_DEPTH-2
        expansion(d, deepest)

    # Output layer with conv_kernel(1,1), P.N_CLASSES=2
    net['out'] = tflayers.conv2d(net['_conv0_2'], P.N_CLASSES, kernel_size=1, padding='same')

    logging.info('Network output shape '+ str(net['out'].get_shape()))

    return net

x = tf.placeholder("float", [None, 512, 512, 1])
w = tf.placeholder("float", [None])
y = tf.placeholder("float", [None, 2])
p_keep_conv = tf.placeholder("float")
phase_train = tf.placeholder(tf.bool, name='phase_train')
data_path = '/home/didia/didia/data/1_1_1mm_slices_lung/'


def main():
    filenames_train, filenames_val = get_file_names()

    model = define_network(x)
    net = model['out']
    net = tf.reshape(net, [-1, 2])
    prob = tf.nn.softmax(net)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y)
    cost = cost * w
    cost = tf.reduce_mean(cost)
    train_op = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(cost)
    pred_op = tf.argmax(net, 1)

    generator_train = dataset.load_images

    train_gen = ParallelBatchIterator(generator_train, filenames_train, ordered=False,
                                      batch_size=batch_size,
                                      multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                      n_producers=P.N_WORKERS_LOAD_AUGMENTATION)
    val_gen = ParallelBatchIterator(generator_train, filenames_val, ordered=True,
                                    batch_size=batch_size,
                                    multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                    n_producers=P.N_WORKERS_LOAD_AUGMENTATION)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    model_dir = '/home/didia/didia/data/result/model/'
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()

        for epoch in range(epoch_num):
            saver.save(sess, os.path.join(model_dir, 'model'))

            for i, batch in enumerate(tqdm(train_gen)):
                inputs, targets, weights, _ = batch

                inputs = np.reshape(inputs, [-1, 512, 512, 1])
                targets = np.reshape(targets, [-1, 1])
                targets = np.concatenate((1 - targets, targets), axis=1)
                # print 'sum',np.sum(targets,axis=0)
                weights = np.reshape(weights, [-1])
                _, preds, py = sess.run([train_op, prob, pred_op], feed_dict={
                    x: inputs,
                    w: weights,
                    y: targets,
                    p_keep_conv: 0.8,
                    phase_train: True})

                np.save('/home/didia/didia/data/result/labels.npy', targets)
                np.save('/home/didia/didia/data/result/preds.npy', preds)

                if i % 30:
                    break
                    # continue
                    # pass
                targets = targets[:, 1]
                num = len(py)
                acc = np.sum(py == targets)
                p = np.sum(py)
                n = num - p

                fp = np.sum(py - targets == 1)
                fn = np.sum(py - targets == -1)
                tp = p - fp
                tn = n - fn
                print '\n\tT', tp, tn
                print '\tF', fp, fn

                print '\tY', np.sum(targets)
                print '\tN', num - np.sum(targets)

            tp_list = []
            fp_list = []
            tn_list = []
            fn_list = []
            for i, batch in enumerate(tqdm(val_gen)):
                if i > 100:
                    # break
                    pass
                inputs, targets, weights, _ = batch
                inputs = np.reshape(inputs, [-1, 512, 512, 1])
                # targets = np.reshape(targets,[-1])
                targets = np.reshape(targets, [-1, 1])
                targets = np.concatenate((1 - targets, targets), axis=1)
                weights = np.reshape(weights, [-1])
                py, preds = sess.run([pred_op, prob], feed_dict={
                    x: inputs,
                    p_keep_conv: 1,
                    phase_train: False, })

                np.save('/home/didia/didia/data/result/tensorflow/{:d}_labels.npy'.format(i), targets)
                np.save('/home/didia/didia/data/result/tensorflow/{:d}_preds.npy'.format(i), preds)

                targets = targets[:, 1]
                num = len(py)
                acc = np.sum(py == targets)
                p = np.sum(py)
                n = num - p

                fp = np.sum(py - targets == 1)
                fn = np.sum(py - targets == -1)
                tp = p - fp
                tn = n - fn

                tp_list.append(tp)
                tn_list.append(tn)
                fp_list.append(fp)
                fn_list.append(fn)

            tp, tn = np.mean(tp_list) / num, np.mean(tn_list) / num
            fp, fn = np.mean(fp_list) / num, np.mean(fn_list) / num
            print
            print '\n\tt', np.mean(tp_list) / num, np.mean(tn_list) / num
            print '\tf', np.mean(fp_list) / num, np.mean(fn_list) / num
            print '\taccuracy', tp + tn
            print '\trecall', tp / (tp + fn)
            print '\tdice', tp / (tp + fn + fp)
            print '\tprecision', tp / (tp + fp)
            # print

    return


main()
