# import the necessary packages
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    #print title+" MSE: %.2f, SSIM: %.2f" % (m,s)
    
    imgA_dct = np.float32(imageA)/255.0
    dct_obj_A = cv2.dct(imgA_dct)
    imgB_dct = np.float32(imageB)/255.0
    dct_obj_B = cv2.dct(imgB_dct)
    
    row1, cols1 = dct_obj_A.shape
    row2, cols2 = dct_obj_B.shape
    mismatch_rate = np.sum(np.absolute(np.subtract(dct_obj_A,dct_obj_B)))/(row1*cols1)
    print title+" MSE: %.5f, SSIM: %.5f, DCT_Diff: %.5f" % (m,s,mismatch_rate)

sample_1 = cv2.imread("sample_1.jpg")
image_1337 = cv2.imread("image_1337.jpg")
image_1338 = cv2.imread("image_1338.jpg")
image_1339 = cv2.imread("image_1339.jpg")
image_1340 = cv2.imread("image_1340.jpg")
 
# convert the images to grayscale
sample_1 = cv2.cvtColor(sample_1, cv2.COLOR_BGR2GRAY)
image_1337 = cv2.cvtColor(image_1337, cv2.COLOR_BGR2GRAY)
image_1338 = cv2.cvtColor(image_1338, cv2.COLOR_BGR2GRAY)
image_1339 = cv2.cvtColor(image_1339, cv2.COLOR_BGR2GRAY)
image_1340 = cv2.cvtColor(image_1340, cv2.COLOR_BGR2GRAY)

compare_images(sample_1, image_1337, "sample_1 vs. image_1337")
compare_images(sample_1, image_1338, "sample_1 vs. image_1338")
compare_images(sample_1, image_1339, "sample_1 vs. image_1339")
compare_images(sample_1, image_1340, "sample_1 vs. image_1340")