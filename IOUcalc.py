from unet_model_with_functions_of_blocks import build_unet
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.optimizers import Adam
import tensorflow as tf
from keras.metrics import MeanIoU
from keras.preprocessing.image import ImageDataGenerator

#Testing on a few test images
seed = 24
n_classes = 2

model = tf.keras.models.load_model('cnn_200epochs.h5', compile=False)

img_data_gen_args = dict(rescale = 1/255.,
                         rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')

mask_data_gen_args = dict(rescale = 1/255.,  #Original pixel values are 0 and 255. So rescaling to 0 to 1
                        rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect',
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

image_data_generator = ImageDataGenerator(**img_data_gen_args)
test_img_generator = image_data_generator.flow_from_directory(r"C:\Users\albon\Desktop\Semantic Segmentation 2.0\test-images", 
                                                              seed=seed, 
                                                              batch_size=32, 
                                                              class_mode=None) #Default batch size 32, if not specified here

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
test_mask_generator = mask_data_generator.flow_from_directory(r"C:\Users\albon\Desktop\Semantic Segmentation 2.0\test-masks", 
                                                              seed=seed, 
                                                              batch_size=32, 
                                                              color_mode = 'grayscale',   #Read masks in grayscale
                                                              class_mode=None)  #Default batch size 32, if not specified here


a = test_img_generator.next()
b = test_mask_generator.next()

import random
test_img_number = random.randint(0, a.shape[0]-1)
test_img = a[test_img_number]
ground_truth=b[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.6).astype(np.uint8)

#plt.figure(figsize=(12, 12))
#plt.subplot(231)
#plt.title('Testing Image')
#plt.imshow(test_img, cmap='gray')
#plt.subplot(232)
#plt.title('Testing Label')
#plt.imshow(ground_truth[:,:,0], cmap='gray')
#plt.subplot(233)
#plt.title('Prediction on test image')
#plt.imshow(prediction, cmap='gray')
#plt.show()

##IoU for a single image
#from keras.metrics import MeanIoU
#n_classes = 2
#IOU_keras = MeanIoU(num_classes=n_classes)  
#IOU_keras.update_state(ground_truth[:,:,0], prediction)
#print("Mean IoU =", IOU_keras.result().numpy())


#Calculate IoU and average
IoU_values = []
for img in range(0, a.shape[0]):
    temp_img = a[img]
    ground_truth=b[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    print(IoU)

df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)