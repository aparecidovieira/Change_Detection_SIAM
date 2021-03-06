# Change_Detection_SIAM

Change Detection Siamese implementation using multi-gpu model tensorflow

Based on siamese network, different convolution blocks for change detection.

# TensorFlow Segmentation



Details can be found in these papers:

* [FULLY CONVOLUTIONAL SIAMESE NETWORKS FOR CHANGE DETECTION](https://arxiv.org/pdf/1810.08462.pdf)

## Siamese Architecture

![Siamese](Images/change1.png)


## Requirements
File environment.yaml
* Python 3.6
* CUDA 10.0
* TensorFlow 1.9
* Keras 2.0


## Modules
utils.py and helper.py 
functions for preprocessing data and saving it.



## Training model:
```
python mainGPU_v2.py --dataset /path/to/dataset/ --model fpn --batch_size 15 --gpu 0,1,2 --checkpoint checkpoint


usage: mainGPU.py [-h] [--num_epochs NUM_EPOCHS] [--save SAVE] [--gpu GPU]
                  [--mode MODE] [--checkpoint CHECKPOINT]
                  [--class_balancing CLASS_BALANCING] [--image IMAGE]
                  [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
                  [--load_data LOAD_DATA] [--act ACT]
                  [--crop_height CROP_HEIGHT] [--crop_width CROP_WIDTH]
                  [--batch_size BATCH_SIZE] [--num_val_images NUM_VAL_IMAGES]
                  [--h_flip H_FLIP] [--v_flip V_FLIP]
                  [--brightness BRIGHTNESS] [--rotation ROTATION]
                  [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --save SAVE           Interval for saving weights
  --gpu GPU             Choose GPU device to be used
  --mode MODE           Select "train", "test", or "predict" mode. Note that
                        for prediction mode you have to specify an image to
                        run the model on.
  --checkpoint CHECKPOINT
                        Checkpoint folder.
  --class_balancing CLASS_BALANCING
                        Whether to use median frequency class weights to
                        balance the classes in the loss
  --image IMAGE         The image you want to predict on. Only valid in
                        "predict" mode.
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --load_data LOAD_DATA
                        Dataset loading type.
  --act ACT             True if sigmoid or false for softmax
  --crop_height CROP_HEIGHT
                        Height of cropped input image to network
  --crop_width CROP_WIDTH
                        Width of cropped input image to network
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --num_val_images NUM_VAL_IMAGES
                        The number of images to used for validations
  --h_flip H_FLIP       Whether to randomly flip the image horizontally for
                        data augmentation
  --v_flip V_FLIP       Whether to randomly flip the image vertically for data
                        augmentation
  --brightness BRIGHTNESS
                        Whether to randomly change the image brightness for
                        data augmentation. Specifies the max bightness change.
  --rotation ROTATION   Whether to randomly rotate the image for data
                        augmentation. Specifies the max rotation angle.
  --model MODEL         The model you are using. Currently supports: encoder-
                        decoder, deepUNet,attentionNet, deep, UNet


