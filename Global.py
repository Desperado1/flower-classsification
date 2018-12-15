# imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

# image size
fixed_size = tuple((500, 500))

#training data path
train_path = "C:\\Users\\himanshu\\Desktop\\flowers classification\\dataset\\train"

# no. of random forest trees
num_trees = 100

# bins for histogram
bins = 0

# train test split size
test_train = 0.10

# seed for representing same results
seed = 9 

# first feature descriptor : Hu Moments
def fd_hu_moments(image):
	# converting imge to greyscale using cvtColor 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# using HuMoments func. to extract Hu moments
	# argument is a flattened image moments 
	feature = cv2.HuMoments(cv2.moments(image)).flatten()
	return feature

# second feature descriptor : Haralick texture
def fd_haralick(image):
	# converting to greyscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# computing haralick texture feature vector
	haralick = mahotas.features.haralick(gray).mean(axis = 0)
	return haralick

# third feature descriptor : Color Histogram
def fd_histogram(image, mask = None):
	# converting imge to HSV using cvtColor 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# computing color histogram
	hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
	# normalize the histogram
	hist = cv2.normalize(hist, hist)
	return hist


# getting training labels
train_labels = os.listdir(train_path)

#sorting training labels
train_labels.sort()
print(train_labels)

# emoty list to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# number of images per class
image_per_class = 80

# loop over training data sub-folders
for training_name in train_labels:
    
    # joiimg the training path athe specific folders
    dir = os.path.join(train_path, training_name)
    
    current_label = training_name
    
    k = 1
    # loop over the images in each sub-folder
    for x in range(1, image_per_class+1):
        # getting inage file name
        file = dir + "/1 (" + str(x) + ").jpg"
        
        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, (500, 500))
        
        #
        # global feature extraction
        #
        
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        #fv_histogram = fd_histogram(image)

        # concatenate global features
        global_feature = np.hstack([fv_hu_moments, fv_haralick])
        
        # updating the list of labels and feature vector
        labels.append(current_label)
        global_features.append(global_feature)
        
        i += 1
        k += 1
    print("[status] processed folder: {}".format(current_label))
    j  += 1
    
print("[status] completed global feature extraction")

# getting the feature vector size
print("feature vector size {}".format(np.array(global_features).shape))

# training label size
print("training label {}".format(np.array(labels).shape))

# encoding target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print( "[STATUS] training labels encoded..")

# normalize feature vector
scaler = MinMaxScaler(feature_range = (0, 1))
rescaled_features = scaler.fit_transform(global_features)
print( "[STATUS] feature vector normalized..")

print( "target labels: {}".format(target))
print("target labels shape: {}".format(target.shape))


# save feature vector using HDF5
h5f_data = h5py.File('C:\\Users\\himanshu\\Desktop\\flowers classification\\output\\data.h5', 'w')
h5f_data.create_dataset('dataset_1', data = np.array(rescaled_features))

h5f_label = h5py.File('C:\\Users\\himanshu\\Desktop\\flowers classification\\output\\label.h5', 'w')
h5f_label.create_dataset('dataset_1', data = np.array(target))

h5f_data.close()
h5f_label.close()


print("[STATUS] end of training..")






