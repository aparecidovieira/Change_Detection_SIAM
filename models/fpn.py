import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os



class FPN(object):
    
    def __init__(self, inputs, n_classes, n_filters=64):
        self.inputs = inputs
        self.n_classes = n_classes
        self.n_filters = n_filters
    
    def conv_layer(self, inputs, n_filters, kernel_size=3, stride=1, norm = True, activation=True):
        kernels = [kernel_size, kernel_size]
        conv1 = slim.conv2d(inputs, n_filters, kernel_size = kernels, stride=[stride, stride])
#         conv1 = slim.group_norm(conv1, scale=True)
        if norm:
            conv1 = slim.batch_norm(conv1, fused=True)

        if activation:
            conv1 = tf.nn.relu(conv1)
        return conv1
        
    def conv_block(self, inputs, n_filters, kernel_size=3, stride=1):
        
        conv1 = self.conv_layer(inputs, n_filters)
        conv2 = self.conv_layer(conv1, n_filters)
        conv2 = self.conv_layer(conv2, n_filters, activation=False)
        
        identity = self.conv_layer(inputs, n_filters, kernel_size=1, activation=False)
      
        out = tf.add(conv2, identity)
        out = tf.nn.relu(out)
        return out        
    
    def SAC(self, inputs, n_filters, kernel_size=3, stride=1):
        
        conv1 = tf.nn.atrous_conv2d(inputs, n_filters, kernel_size=3, rate=1)
        conv2 = tf.nn.atrous_conv2d(inputs, n_filters, kernel_size=3, rate=3)
        avgPool = self.pool_layer(inputs, pool_size=5, stride=5, _type='AVG')
        conv1x1 = self.conv_layer(avgPool, n_filters, kernel_size=1, norm=False, activation=False)
        conv1 = tf.multiply(conv1, conv1x1)
        conv2 = tf.multiply(conv2, 1-conv1x1)
        out = tf.add(conv1, conv2)
        return out

    def resnet_block(self, inputs, n_filters, stride=2):
        
        conv1 = self.conv_layer(inputs, n_filters, kernel_size=1)
        conv2 = self.conv_layer(conv1, n_filters, stride=stride)
        conv2 = self.conv_layer(conv2, n_filters, kernel_size = 1, activation=False)
        
        identity = self.conv_layer(inputs, n_filters, kernel_size=1, stride=stride, activation=False)
      
        out = tf.add(conv2, identity)
        out = tf.nn.relu(out)
        return out    
   
    def resneXt_block(self, inputs, n_filters, stride=2):
        
        conv1 = self.conv_layer(inputs, n_filters, kernel_size=1)
        conv2 = self.grouped_conv(conv1, n_filters, stride=stride)
        conv2 = self.conv_layer(conv2, n_filters, kernel_size = 1, activation=False)
        
        identity = self.conv_layer(inputs, n_filters, kernel_size=1, stride=stride, activation=False)
      
        out = tf.add(conv2, identity)
        out = tf.nn.relu(out)
        return out    
       
    def grouped_conv(self, inputs, n_filters, stride=1, card=8):
        n_filters /= card
#         g_conv = []
#         for c in range(card):
#             inputs_split = tf.split(inputs, n_filters: (c + 1) * n_filters)
        inputGroups = tf.split(axis=3, num_or_size_splits=card, value=inputs)
        g_conv =  [self.conv_layer(x, n_filters, kernel_size=3, stride=stride, activation=False, norm=False) for x in inputGroups]
#         g_conv.append(conv_x)
#         g_conv = tf.concat([g_conv], axis=-1)
        g_conv = tf.concat(axis=3, values=g_conv)
        g_conv = slim.batch_norm(g_conv, fused=True)

        g_conv = tf.nn.relu(g_conv) 
        return g_conv
    
    def pool_layer(self, inputs, pool_size=2, n_filters=64, stride=2, _type='MAX', method='pool'):
        if method == 'pool':
            if _type == 'MAX':
                out = slim.pool(inputs, [pool_size, pool_size], stride=[stride, stride], pooling_type=_type)
            else:
                out = slim.avg_pool2d(inputs, [pool_size, pool_size], stride=[stride, stride])

        else:
            out = slim.conv2d(inputs, n_filters, kernel_size=[1, 1], stride=[2, 2])
        return out
    
    def bottleneck(self, inputs, n_filters):
        conv1 = self.conv_layer(inputs, n_filters, kernel_size=1)
        conv2 = self.conv_layer(conv1, n_filters, activation=False)
        conv3 = self.conv_layer(conv2, n_filters, kernel_size=1)
        
        identity = self.conv_layer(inputs, n_filters, kernel_size=1, activation=False)
      
        out = tf.add(conv3, identity)
        out = tf.nn.relu(out)
        return out
    

    
    def Upsample(self, inputs, rate=2):
        return tf.image.resize_bilinear(inputs, (tf.shape(inputs)[1] * rate, tf.shape(inputs)[2] * rate))
    
    
    def GlobalContext(self, inputs, n_filters, kernel_size=1):
        pass
        
    def UpSampling(self, inputs, rate=2, n_filters=64, _type='bilinear'):
        if _type == 'bilinear':
            out = self.Upsample(inputs, rate)
            out = self.conv_layer(out, n_filters)
        else:
            out = slim.conv2d_transpose(inputs, n_filters, kernel_size = [2, 2], stride = [2, 2])
#             out = slim.group_norm(out)
            out = tf.nn.relu(out)
        return out
    
    def FPN(self, inputs, blocks=4):
        
        layers = []
        resBlocks = [3, 4, 6, 3]
        for i in range(blocks):
#             inputs = self.conv_block(inputs, self.n_filters * (2 ** i))
            for j in range(resBlocks[i]):
                inputs = self.resnet_block(inputs,  self.n_filters * (2 ** i), stride= 1)
            layers.append(inputs)
            inputs = self.resnet_block(inputs,  self.n_filters * (2 ** i), stride =  2)
            
#             inputs = self.pool_layer(inputs)
#             layers.append(inputs)
        
        bottleUpLayers = []
#         bottleUpLayers.append(inputs)
        inputs = self.conv_layer(inputs, self.n_filters * 16, stride=1)
        
#         bottleUpLayers = []
#         bottleUpLayers.append(inputs)
        layers = layers[::-1][:-1]
        
        for i in range(blocks-1):
            up = layers[i]
            inputs = self.UpSampling(inputs, rate=2, n_filters = self.n_filters * (2 ** (3 - 1)), _type='trans')
            up = self.conv_layer(up, n_filters = self.n_filters * (2 ** (3 - 1)), kernel_size = 1)
            inputs = tf.add(up, inputs)
            inputs = self.conv_layer(inputs, n_filters = self.n_filters * (2 ** (3 - 1)), kernel_size = 3)
            
            bottleUpLayers.append(inputs)

        fp_layers = []
        for i, fp_layer in enumerate(bottleUpLayers[:]):
#             fp_layer = self.resneXt_block(fp_layer, self.n_filters * (2 ** (3-2)), stride=1)
            fp_layer = self.conv_layer(fp_layer, n_filters = self.n_filters * (2 ** (3 - 2)), kernel_size = 3)
            fp_layer = self.conv_layer(fp_layer, n_filters = self.n_filters * (2 ** (3 - 2)), kernel_size = 3)

#             fp_layer = self.conv_block(fp_layer, self.n_filters * (2 ** (3-i)))            
            fp_layers.append(fp_layer)
#         out = tf.concat([fp_layers], axis=-1)    
        return fp_layers
            
        
        
    def model(self):
        
        inputs1 = self.inputs[:, :, :256, :]
        inputs2 = self.inputs[:, :, 256:, :]
        
        fp_layers_1 = self.FPN(inputs1) # FP4, FP3, FP@, FP1
        fp_layers_2 = self.FPN(inputs2)     
#         out0, out1, out2, out4 = fp_layers_1
        
                                   
#         out = []
        for i in range(len(fp_layers_1)):
            fp1 = fp_layers_1[i]
            fp2 = fp_layers_2[i]                      
            fp_l = tf.concat([fp1, fp2], axis=-1)
            fp_l = self.UpSampling(fp_l, rate=2**(3-i), n_filters=self.n_filters * 2**(3-i))
#             fp_l = self.conv_layer(fp_l, self.n_filters * (2 **(3-2)), stride=1)
            
            out = fp_l if i == 0 else (tf.concat([fp_l, out], axis=-1))
#             out.append(fp_l)
#         out0, out1, out2, out3 = outresneXt_block
#         out = tf.concat([out0, out1, out2, out3], axis=-1)
#         out = self.UpSampling(out, self.n_filters,resneXt_block _type='transpose')
        out = self.conv_layer(out, n_filters = self.n_filters , kernel_size = 3)
                                   
        return slim.conv2d(out, self.n_classes, kernel_size=[1, 1])
                                   
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  