from skimage.io import imread
from skimage import img_as_float
import pylab
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import math

image = imread('parrots.jpg')
float_image = img_as_float(image)
pylab.imshow(float_image)
data_list = []
mean = []
median = []
for i in range(float_image.shape[0]):
    for j in range(float_image.shape[1]):
        data_dict = {}
        data_dict.update({"i":i, "j":j,"r":float_image[i][j][0],"g":float_image[i][j][1], "b":float_image[i][j][2]})
        data_list.append(data_dict)
data = pd.DataFrame(data_list)

X=data.iloc[:,2:5]
for cl in range(2,21):
    kmeans = KMeans(init='k-means++' , random_state=241, n_clusters=cl ).fit(X)
    y = kmeans.predict(X)
    data['cluster'] = y
    data["meanr"] = (data.groupby(['cluster'])['r'].transform(np.mean))
    data["meang"] = (data.groupby(['cluster'])['g'].transform(np.mean))
    data["meanb"] = (data.groupby(['cluster'])['b'].transform(np.mean))
    data["medianr"] = (data.groupby(['cluster'])['r'].transform(np.median))
    data["mediang"] = (data.groupby(['cluster'])['g'].transform(np.median))
    data["medianb"] = (data.groupby(['cluster'])['b'].transform(np.median))

##########metric PSNR########
    MSE_mean = (1/(3*float_image.shape[0]*float_image.shape[1]))*(sum((data["r"] - data["meanr"])**2)+sum((data["g"] - data["meang"])**2)+sum((data["b"] - data["meanb"])**2))
    MSE_median = (1/(3*float_image.shape[0]*float_image.shape[1]))*(sum((data["r"] - data["medianr"])**2)+sum((data["g"] - data["mediang"])**2)+sum((data["b"] - data["medianb"])**2))
    print(cl, MSE_mean, MSE_median)
    PSNR_mean = 20*math.log10(1/math.sqrt(MSE_mean))
    PSNR_median = 20*math.log10(1/math.sqrt(MSE_median))
    print(cl, PSNR_mean, PSNR_median)
plt.show()