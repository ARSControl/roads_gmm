"""
This code uses DataGen to directly load images from the drive in batches
for training. This is best for datasets that do not fit in memory in its entirety.
The code also uses DataGen for validation and testing.
"""

from UNet import build_unet
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.optimizers import Adam
import tensorflow as tf

#New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
#This gives a binary mask rather than a mask with interpolated values. 
seed = 24
batch_size = 2
from keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(rescale = 1/255.,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect')

mask_data_gen_args = dict(rescale = 1/255.,  #Original pixel values are 0 and 255. So rescaling to 0 to 1
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect',
                        preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow_from_directory(r"C:\Users\albon\Desktop\Test\Dataset256x256\train-images", 
                                                           seed=seed,
                                                           #target_size=(256,256), 
                                                           batch_size=batch_size,
                                                           class_mode=None)  #Very important to set this otherwise it returns multiple numpy arrays 
                                                                             #thinking class mode is binary.

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow_from_directory(r"C:\Users\albon\Desktop\Test\Dataset256x256\train-masks", 
                                                         seed=seed, 
                                                         batch_size=batch_size,
                                                         #target_size=(256,256),
                                                         color_mode = 'grayscale',   #Read masks in grayscale
                                                         class_mode=None)


valid_img_generator = image_data_generator.flow_from_directory(r"C:\Users\albon\Desktop\Test\Dataset256x256\val-images", 
                                                               seed=seed, 
                                                               #target_size=(256,256),
                                                               batch_size=batch_size, 
                                                               class_mode=None) #Default batch size 32, if not specified here

valid_mask_generator = mask_data_generator.flow_from_directory(r"C:\Users\albon\Desktop\Test\Dataset256x256\val-masks", 
                                                               seed=seed, 
                                                               #target_size=(256,256),
                                                               batch_size=batch_size, 
                                                               color_mode = 'grayscale',   #Read masks in grayscale
                                                               class_mode=None)  #Default batch size 32, if not specified here

train_generator = zip(image_generator, mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)

x = image_generator.next()
y = mask_generator.next()

#for i in range(0,1):
#    image = x[i]
#    mask = y[i]
#    plt.subplot(1,2,1)
#    plt.imshow(image[:,:,0], cmap='gray')
#    plt.subplot(1,2,2)
#    plt.imshow(mask[:,:,0])
#    plt.show()

####################################################################
#Jaccard distance loss mimics IoU. 
#from keras import backend as K
#def jaccard_distance_loss(y_true, y_pred, smooth=100):
#    """
#    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
#            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
#    
#    The jaccard distance loss is usefull for unbalanced datasets. This has been
#    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
#    gradient.
#    
#    Ref: https://en.wikipedia.org/wiki/Jaccard_index
#    
#    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
#    @author: wassname
#    """
#    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
#    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
#    jac = (intersection + smooth) / (sum_ - intersection + smooth)
#    return (1 - jac) * smooth

#Dice metric can be a great metric to track accuracy of semantic segmentation.
#def dice_metric(y_pred, y_true):
#    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
#    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
     #if y_pred.sum() == 0 and y_pred.sum() == 0:
     #    return 1.0
#    return 2*intersection / union

IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = build_unet(input_shape)

#STANDARD BINARY CROSS ENTROPY AS LOSS AND ACCURACY AS METRIC
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

#JACCARD LOSS AND DICE METRIC 
# model.compile(optimizer=Adam(lr = 1e-3), loss=jaccard_distance_loss, 
#               metrics=[dice_metric])

num_train_imgs = len(os.listdir(r"C:\Users\albon\Desktop\Test\Dataset256x256\train-images\train"))

steps_per_epoch = num_train_imgs //batch_size

history = model.fit(train_generator, validation_data=val_generator,
                    batch_size=batch_size, validation_batch_size=batch_size, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=steps_per_epoch, epochs=20)

model.save('cnn.h5')

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#acc = history.history['dice_metric']
acc = history.history['accuracy']
#val_acc = history.history['val_dice_metric']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Dice')
plt.plot(epochs, val_acc, 'r', label='Validation Dice')
plt.title('Training and validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()

###############################################################
##Test the model on images we held out for testing.
##We can use the generator again for predictions. 
##Remember that this is not test time augmentation. This is only using augmentaton
##to load images from the drive, perfomr augmentation and predict. 
## TTA means predicting on augmenting images and then combining predictions for better
##accuracy. 
#
#model = tf.keras.models.load_model("mitochondria_load_from_disk_focal_dice_50epochs.hdf5", compile=False)
#
#test_img_generator = image_data_generator.flow_from_directory(r"C:\Users\albon\Desktop\Semantic Segmentation 2.0\test-images", 
#                                                              seed=seed, 
#                                                              batch_size=32, 
#                                                              class_mode=None) #Default batch size 32, if not specified here
#
#test_mask_generator = mask_data_generator.flow_from_directory(r"C:\Users\albon\Desktop\Semantic Segmentation 2.0\test-masks", 
#                                                              seed=seed, 
#                                                              batch_size=32, 
#                                                              color_mode = 'grayscale',   #Read masks in grayscale
#                                                              class_mode=None)  #Default batch size 32, if not specified here
#
#### Testing on a few test images
#
#a = test_img_generator.next()
#b = test_mask_generator.next()
#for i in range(0,5):
#    image = a[i]
#    mask = b[i]
#    plt.subplot(1,2,1)
#    plt.imshow(image[:,:,0], cmap='gray')
#    plt.subplot(1,2,2)
#    plt.imshow(mask[:,:,0])
#    plt.show()
#
#import random
#test_img_number = random.randint(0, a.shape[0]-1)
#test_img = a[test_img_number]
#ground_truth=b[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
#test_img_input=np.expand_dims(test_img, 0)
#prediction = (model.predict(test_img_input)[0,:,:,0] > 0.6).astype(np.uint8)
#
#plt.figure(figsize=(16, 8))
#plt.subplot(231)
#plt.title('Testing Image')
#plt.imshow(test_img, cmap='gray')
#plt.subplot(232)
#plt.title('Testing Label')
#plt.imshow(ground_truth[:,:,0], cmap='gray')
#plt.subplot(233)
#plt.title('Prediction on test image')
#plt.imshow(prediction, cmap='gray')
#
#plt.show()
#
#IoU for a single image
#from keras.metrics import MeanIoU
#n_classes = 2
#IOU_keras = MeanIoU(num_classes=n_classes)  
#IOU_keras.update_state(ground_truth[:,:,0], prediction)
#print("Mean IoU =", IOU_keras.result().numpy())
#
##Calculate IoU and average
# 
#import pandas as pd
#
#IoU_values = []
#for img in range(0, a.shape[0]):
#    temp_img = a[img]
#    ground_truth=b[img]
#    temp_img_input=np.expand_dims(temp_img, 0)
#    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
#    
#    IoU = MeanIoU(num_classes=n_classes)
#    IoU.update_state(ground_truth[:,:,0], prediction)
#    IoU = IoU.result().numpy()
#    IoU_values.append(IoU)
#
#    print(IoU)
#
#df = pd.DataFrame(IoU_values, columns=["IoU"])
#df = df[df.IoU != 1.0]    
#mean_IoU = df.mean().values
#print("Mean IoU is: ", mean_IoU)    