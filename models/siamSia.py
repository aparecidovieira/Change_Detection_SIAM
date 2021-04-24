import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os



def conv_nested(inputs, filters, kernel_size=3, stride=1, n_blocks=2):
    kernels = [kernel_size, kernel_size]
    
    for _ in range(n_blocks):
        inputs = conv_nested(inputs, filters=filters, kernel_size = kernels, stride=[stride, stride])
    
    return inputs

def conv_block(inputs, n_filters, kernel_size=3, stride=1, n_blocks=1, name=None):
    kernels = [kernel_size, kernel_size]
    inputs = slim.conv2d(inputs, n_filters, kernel_size=kernels, stride=stride)
    indentity = inputs
    inputs = slim.batch_norm(inputs, fused=True)
    inputs = tf.nn.relu(inputs)
    inputs = slim.conv2d(inputs, n_filters, kernel_size=kernels, stride=[stride, stride])
    
    inputs = slim.batch_norm(inputs, fused=True)
    inputs = tf.nn.relu(inputs + indentity)
    
    return inputs

def conv_tr_block(inputs, n_filters, kernel_size=2, stride=2):
    kernels = [kernel_size, kernel_size]

    inputs = slim.conv2d_transpose(inputs, n_filters, kernel_size=kernels, stride=[stride, stride])
    inputs = slim.batch_norm(inputs, fused=True)
    inputs = tf.nn.relu(inputs)
        
    return inputs

# def conv_tr_block(inputs, filters, filters_f=None, kernel_size=2, stride=1, n_blocks=2):
#     kernels = [kernel_size, kernel_size]
#     if not filters_f:
#         filters_f = filters
#     last = n_blocks - 1
#     for i in range(n_blocks):
#         inputs = slim.conv2d_transpose(inputs, n_filters if i != last else filters_f, kernel_size=kernels, stride=stride)
#         inputs = slim.batch_norm(inputs, fused=True)
#         inputs = tf.nn.relu(inputs)
        
#     return inputs

def add_nested(*args):
    out = args[0]
    for tensor in args:
        out = tf.add(out, tensor)
    return out

def pool_block(inputs, kernel_size=2, stride=2, _type = 'MAX'):
    
    kernels = [kernel_size, kernel_size]
    stride = [stride, stride]
    net = slim.pool(inputs, kernels, stride=stride, pooling_type=_type)
    return net

def adapPooling(inputs, outputsize=1, _type='MAX'):
    shape = 256, 256, 64#K.int_shape(inputs)
    h, w, ch = shape
    stride = 256#np.floor(h/outputsize).astype(np.int32)
    kernel = 256#h - (outputsize-1) * stride
    if _type == 'MAX':
        adapPooling = slim.pool(inputs, [kernel, kernel], stride=[stride, stride], pooling_type='MAX')
    else:
        adapPooling = slim.avg_pool2d(inputs, [kernel, kernel], stride=[stride, stride])
        
    return adapPooling * inputs
    

def channel_atttention(inputs, ch_output, ratio=16):
        
    ### Converting to 512, 512, 1
#     inputs = slim.conv2d(inputs, 1, kernel_size=[1, 1], stride=1)
#     average_pool = pool_block(inputs, _type='AVG')

#     average_pool = slim.pool(inputs, kernels, stride, pooling_type='MAX')
#     average_pool = slim.avg_pool2d(inputs, [256, 256], 1)
    average_pool = adapPooling(inputs, 1, _type='AVG')
    
#     fc1 = slim.layers.fully_connected(average_pool, ch_output//ratio, activation_fn=tf.nn.relu)
#     fc1 = slim.layers.fully_connected(fc1, ch_output,activation_fn=None)
    
    fc1 = slim.conv2d(average_pool, ch_output//ratio, kernel_size=[1, 1], stride=[1, 1])
    fc1 = tf.nn.relu(fc1)
    fc1 = slim.conv2d(fc1, ch_output, kernel_size=[1, 1], stride=[1,1])
   #     max_pool = slim.pool(inputs, [256, 256], stride=1, pooling_type='MAX')
    max_pool = adapPooling(inputs, 1, _type='MAX')
#     fc2 = slim.layers.fully_connected(max_pool, ch_output//ratio, activation_fn=tf.nn.relu)
#     fc2 = slim.layers.fully_connected(fc2, ch_output,activation_fn=None)
    
    fc2 = slim.conv2d(max_pool, ch_output//ratio, kernel_size=[1, 1], stride=[1, 1])
    fc2 = tf.nn.relu(fc2)
    fc2 = slim.conv2d(fc2, ch_output, kernel_size=[1, 1], stride=[1, 1])

    out = tf.add(fc1, fc2)
    out = tf.nn.sigmoid(out)
    return out

def build_siamSia(inputs, num_classes=2, ch_output=64):
    
#     input_0 = inputs[:, 
#     input_0, input_1 = tf.slice(inputs)
    input_0, input_1 = inputs[:, :, :256, :], inputs[:, :, 256:, :]
    net_0 = conv_block(input_0, ch_output)
    pool_0 = pool_block(net_0)
    
    net_1 = conv_block(pool_0, ch_output * 2)
    pool_1 = pool_block(net_1)
    
    net_2 = conv_block(pool_1, ch_output * 4, n_blocks=3)
    pool_2 = pool_block(net_2)
    
    net_3 = conv_block(pool_2, ch_output * 8, n_blocks=3)
    pool_3 = pool_block(net_3)
    
    ##############
    # Stage 1
    _net_0 = conv_block(input_1, ch_output)
    _pool_0 = pool_block(_net_0)
    
    # Stage 2
    _net_1 = conv_block(_pool_0, ch_output * 2)
    _pool_1 = pool_block(_net_1)
    
    _net_2 = conv_block(_pool_1, ch_output * 4, n_blocks=3)
    _pool_2 = pool_block(_net_2)
    
    _net_3 = conv_block(_pool_2, ch_output * 8, n_blocks=3)
    _pool_3 = pool_block(_net_3)
    
    _net_4 = conv_block(_pool_3, ch_output * 16, n_blocks=3)
    _pool_4 = pool_block(_net_4)
    
#     _trans_3 = conv_tr_block(_pool_4, output * 8, n_blocks=3)
    
#     concat_3 = tf.concat([net_3, _net_3], axis=-1)
#     concat_2 = tf.concat([net_2, _net_2], axis=-1)
#     concat_1 = tf.concat([net_1, _net_1], axis=-1)
#     concat_0 = tf.concat([net_0, _net_0], axis=-1)
    concat_3 = tf.concat([pool_3, _pool_3], axis=-1)
    concat_2 = tf.concat([pool_2, _pool_2], axis=-1)
    concat_1 = tf.concat([pool_1, _pool_1], axis=-1)
    concat_0 = tf.concat([pool_0, _pool_0], axis=-1)
#     _trans_4 = conv_tr_block(_pool_4, output * 8)
#     concat_4 = tf.concat([pool_3, _pool_3], axis=-1)
    
    
    _trans_0 = conv_tr_block(_pool_1, ch_output)
    _concat_0 = conv_block(tf.concat([concat_0, _trans_0], axis=-1), ch_output)
#     _concat_0 = conv_tr_block(_concat_0, ch_output)
    
    _trans_1 = conv_tr_block(_pool_2, ch_output * 2)
    _concat_1 = conv_block(tf.concat([concat_1, _trans_1], axis=-1), ch_output * 2)
    _trans_1_1 = conv_tr_block(_concat_1, ch_output)
    _concat_1_1 = conv_block(tf.concat([_concat_0, concat_0 ,_trans_1_1], axis=-1), ch_output)
#     _concat_1_1 = conv_tr_block(_concat_1_1, ch_output)
    
    
    _trans_2 = conv_tr_block(_pool_3, ch_output * 4)
    _concat_2 = conv_block(tf.concat([concat_2, _trans_2], axis=-1), ch_output * 4)
    _trans_2_1 = conv_tr_block(_concat_2, ch_output * 2)
    _concat_2_1 = conv_block(tf.concat([_concat_1, concat_1,_trans_2_1], axis=-1), ch_output * 2)
    _trans_2_2 = conv_tr_block(_concat_2_1, ch_output)
    _concat_2_2 = conv_block(tf.concat([_concat_1_1, concat_0,_trans_2_2, _concat_0], axis=-1), ch_output)
#     _concat_2_2 = conv_tr_block(_concat_2_2, ch_output)
    
    
    _trans_3 = conv_tr_block(_pool_4, ch_output * 8)
    _concat_3 = conv_block(tf.concat([concat_3, _trans_3], axis=-1), ch_output * 8)
    _trans_3_1 = conv_tr_block(_concat_3, ch_output * 4)
    _concat_3_1 = conv_block(tf.concat([_concat_2, concat_2 ,_trans_3_1], axis=-1), ch_output * 4)
    _trans_3_2 = conv_tr_block(_concat_3_1, ch_output * 2)
    
    _concat_3_2 = conv_block(tf.concat([_concat_1, _trans_3_2, _concat_2_1, concat_1], axis=-1), ch_output * 2)
#     _concat_3_2 = conv_block(_concat_3_2, output * 8)
    _trans_3_3 = conv_tr_block(_concat_3_2, ch_output)
    _concat_3_3 = conv_block(tf.concat([_concat_0, _concat_1_1, _trans_3_3, _concat_2_2, concat_0], axis=-1), ch_output)
#     _concat_3_3 = conv_tr_block(_concat_3_3, ch_output)
    
    _concat_0 = conv_tr_block(_concat_0, ch_output)
    _concat_1_1 = conv_tr_block(_concat_1_1, ch_output)
    _concat_2_2 = conv_tr_block(_concat_2_2, ch_output)
    _concat_3_3 = conv_tr_block(_concat_3_3, ch_output)
    
    
    
    out = tf.concat([_concat_0, _concat_1_1, _concat_2_2, _concat_3_3], axis=-1)
#     scp = out
#     out = conv_tr_block(out, 4 * ch_output)
    
    add_out = add_nested(_concat_0, _concat_1_1, _concat_2_2, _concat_3_3) # intra
    
    cam = channel_atttention(out, ch_output * 4)
    cam1 = channel_atttention(add_out, ch_output, ratio=4)
    
    cam_cat = tf.concat([cam1, cam1, cam1, cam1], axis=-1)
    add_out_1 = tf.add(out, cam_cat)
#     out = tf.multiply(cam, add_out_1)
    out = cam * add_out_1
#     out = conv_tr_block(out, ch_output)
    
#     out = conv_tr_block(out, ch_output)
    out = slim.conv2d(out, num_classes, kernel_size=[1, 1], stride=1)
    return out                           
    
    
    
    
    
    
    
    
    
    
    