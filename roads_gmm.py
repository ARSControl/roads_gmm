#!/usr/bin/env python3

import time, cv2, math, os
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import tensorflow as tf

from smooth_predictions_by_belnding_patches import predict_img_with_smooth_windowing
from patchify import patchify, unpatchify

my_dpi = 96
DIM_MT_IMG = [300, 200]                         # Real size of environment in the picture
COMPONENTS_NUM = 20

# convert pixel to mt
def distmt(x, y, img_size, mt):
    dist_x = (x*mt[0])/img_size[1]
    dist_y = (y*mt[1])/img_size[0]
    return [math.sqrt(dist_x**2+dist_y**2), dist_x, dist_y]

# probability function for multivariate gaussian (X,Y: meshgrid)
def multigauss_pdf(X, Y, means, covariances, weights):
    # Flatten the meshgrid coordinates
    points = np.column_stack([X.flatten(), Y.flatten()])

    # Number of components in the mixture model
    num_components = len(means)

    # Initialize the probabilities
    probabilities = np.zeros_like(X)

    # Calculate the probability for each component
    for i in range(num_components):
        mean = means[i]
        covariance = covariances[i]
        weight = weights[i]

        # Calculate the multivariate Gaussian probability
        exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
        coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
        component_prob = coefficient * np.exp(exponent)

        # Add the component probability weighted by its weight
        probabilities += weight * component_prob.reshape(X.shape)

    return probabilities


def gmm_model(image):
    xp, yp = [], []
    IMG_SIZE = image.shape

    for i in range(IMG_SIZE[0]):
        for j in range(IMG_SIZE[1]):
            if image[i,j] == 255: 
                #xp.append(j)
                #yp.append(IMG_SIZE[1]-i)

                # Da pos pixel a mt
                k,x,y = distmt(j, i, IMG_SIZE, DIM_MT_IMG)
                xp.append(x)
                yp.append(DIM_MT_IMG[1]-y)

    GMModel = GaussianMixture(n_components=COMPONENTS_NUM, covariance_type='full', max_iter=1000)
    GMModel.fit(np.column_stack((xp, yp)))
    # calculate BIC
    # bic = GMModel.bic(np.column_stack((xp, yp)))

    # get means and covariances
    means = GMModel.means_
    covariances = GMModel.covariances_
    mix = GMModel.weights_
    #print("Means: {}".format(means))
    #print("Coveriances: {}".format(covariances))
    #print("Mixture proportions: {}".format(mix))

    return means, covariances, mix




def recompone_images(pat, x, y):
    row = []
    backtoimg = []
    for i in range(len(pat)):
        row.append(np.array(pat[i]))  
        if (i+1) % x == 0:
            backtoimg.append(row)
            row = []
    backtoimg = np.array(backtoimg)
    img = unpatchify(backtoimg, (y*256, x*256))
    return img


def predict(image_path, model_path):
    image = cv2.imread(image_path, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nr1 = int(image.shape[0] / 256.)
    nc1 = int(image.shape[1] / 256.)
    image = image[0:nr1*256, 0:nc1*256]

    if nr1 > nc1:
        n = int( nr1/nc1 )
        image1 = cv2.resize(image, (256,256*n))
    else:
        n = int( nc1/nr1 )
        image1 = cv2.resize(image, (256*n,256))

    nr = int(image1.shape[0] / 256.)
    nc = int(image1.shape[1] / 256.)

    model = tf.keras.models.load_model(model_path, compile=False)

    patch_size = 256
    patches = []

    patches_img = patchify(image1, (patch_size, patch_size, 3), step=patch_size)
    for k in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[k,j,:,:]
            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds
            patches.append(single_patch_img)

    #Prediction without using blending patches
    mask_patches = []
    for i in range(len(patches)):
        img = patches[i] / 255.0 
        p0 = model.predict(np.expand_dims(img, axis=0))[0][:, :, 0]
        p1 = model.predict(np.expand_dims(np.fliplr(img), axis=0))[0][:, :, 0]
        p1 = np.fliplr(p1)
        p2 = model.predict(np.expand_dims(np.flipud(img), axis=0))[0][:, :, 0]
        p2 = np.flipud(p2)
        p3 = model.predict(np.expand_dims(np.fliplr(np.flipud(img)), axis=0))[0][:, :, 0]
        p3 = np.fliplr(np.flipud(p3))
        thresh = 0.2
        p = (p0 + p1 + p2 + p3) / 4
        mask_patches.append(p)

    prediction = recompone_images(mask_patches, nc, nr)
    pred = (prediction > thresh).astype(np.uint8)

    #Prediction using blending patches
    input_img = image1/255.
    predictions_smooth = predict_img_with_smooth_windowing(
                                                        input_img,
                                                        window_size=patch_size,
                                                        subdivisions=2,
                                                        nb_classes=1,
                                                        pred_func=(lambda img_batch_subdiv: model.predict((img_batch_subdiv)))
                                                        )

    final_prediction = (predictions_smooth > thresh).astype(np.uint8)
    union_prediction = (((prediction + 2*predictions_smooth[:,:,0]) / 2) > thresh).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    iimmgg = image.copy()
    z = cv2.resize(union_prediction, (nc1*256, nr1*256))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i,j] == 1:
                iimmgg[i,j] = 255

    #fig = plt.figure(figsize=(12,12))
    #fig.add_subplot(3, 1, 1)
    #plt.title('Original Image')
    #plt.imshow(image)
    #fig.add_subplot(3, 1, 2)
    #plt.title('Pred')
    #plt.imshow(union_prediction, cmap='gray')
    #fig.add_subplot(3, 1, 3)
    #plt.title('Test: Image Resize n*256 x n*256')
    #plt.imshow(iimmgg)
    #plt.show()

    return union_prediction*255

if __name__ == '__main__':

    #image must be RGB
    image_path = '/home/ubuntu/env.png'
    model_path = '/home/ubuntu/RoadExtractionModel.h5'
    image = predict(image_path, model_path)
    
    mns, cov, mix = gmm_model(image)

    print("Means: {}".format(mns))
    print("Coveriances: {}".format(cov))
    print("Mixture proportions: {}".format(mix))


