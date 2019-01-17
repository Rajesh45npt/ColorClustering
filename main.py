from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import numpy as np



def centroid_histogram(clt):
    numLabels = np.arange(0,len(np.unique(clt.labels_))+1)
    hist, binedges = np.histogram(clt.labels_, bins=numLabels)
    print(hist)
    print(binedges)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startx=0

    for(percent, color) in zip(hist, centroids):
        color = color.reshape(1,1,3).astype("uint8")
        color = cv2.cvtColor(color,cv2.COLOR_HSV2RGB).reshape(3)
        print("New Color", color)
        endx = startx+percent*300
        cv2.rectangle(bar,(int(startx),0), (int(endx),50),
        color.astype("uint8").tolist(), -1)
        startx = endx
    return bar

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",required=True, help="path to the file")
ap.add_argument("-k", "--nclusters",required=True, help="Number of clusters", type=int)
args=vars(ap.parse_args())


img = cv2.imread(args["image"])
# cv2.imshow("Original Image", img)
# cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
img_flatten = img_hsv.reshape(-1,3)
print(img_flatten.shape)


clst = KMeans(n_clusters = args["nclusters"])
clst.fit(img_flatten)
print(clst.labels_)
hist = centroid_histogram(clst)
bar = plot_colors(hist, clst.cluster_centers_)
print(clst.cluster_centers_)


plt.figure("Original Image")
plt.axis("off")
plt.imshow(img)
plt.show()

plt.figure("Clustered colors")
plt.axis("off")
plt.imshow(bar)
plt.show()
print(clst.cluster_centers_.shape)
print(clst.cluster_centers_)

black_color_index = np.argmin(clst.cluster_centers_[:,2])
print(black_color_index)

print(len(clst.labels_), img.shape[0]*img.shape[1])
black_flag = clst.labels_ == black_color_index
black_flag = black_flag.reshape(img.shape[:2])
img_filtered = np.zeros(black_flag.shape, dtype="uint8")
img_filtered[black_flag] = 255

plt.figure("Black colored pixels")
plt.axis("off")
plt.imshow(img_filtered, cmap="gray")
plt.show()

gaussian_img = cv2.GaussianBlur(img_filtered,(3,3),3)
median_img = cv2.medianBlur(img_filtered,3)


documentfolder = args['image'].split(".jpg")[0]
# print(documentfolder)
cv2.imwrite(documentfolder+"_original_HSV.jpg",255-img_filtered)
cv2.imwrite(documentfolder+"_medianblur_HSV.jpg",255-median_img)
cv2.imwrite(documentfolder+"_gaussianblur_HSV.jpg",255-gaussian_img)

