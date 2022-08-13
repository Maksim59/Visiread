import numpy as np
import tensorflow
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path = 'Alphabet'
pathlabels = "labels.csv"
count = 0
testRatio = 0.2
valRatio = 0.2
images_dim = (32,32,3)


myList = os.listdir(path)

print(("Total Classes Detected:"),len(myList))
print(len(myList))
print("Importing Classes")

images = []
classNo = []

noOfClasses = len(myList)

for x in range(0,noOfClasses):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+str(y))
        curImg = cv2.resize(curImg,(images_dim[0],images_dim[1]))
        images.append(curImg)
        classNo.append(count)
    print(count,end=" ")
    count +=1
print(" ")
print("Total Images in Image List", len(images))
print("Total IDS in classNo List = ", len(classNo))
images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)

#SPLIT#
X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size = testRatio)
X_train,X_validation,y_train, y_validation = train_test_split(X_train,y_train,test_size = valRatio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for x in range(0,noOfClasses):
    #print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

# plt.figure(figsize = (10,5))
# plt.bar(range(0,noOfClasses), numOfSamples)
# plt.title("Number of Images of Each Class")
# plt.xlabel("Class ID")
# plt.ylabel("Number of Images")
# plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[2])
# img = cv2.resize(img,(300,300))
# cv2.imshow("preProcessed",img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))

print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation= X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

print(X_train.shape)


dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)

y_train = to_categorical(y_train,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(images_dim[0],
                                                               images_dim[1],
                                                               1),activation='relu'
                                                                )))

    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Dense(noOfNode,activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation = 'softmax'))
    model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

model = myModel()
print(model.summary)
