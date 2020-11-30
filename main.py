# This is a sample Python script.
import matplotlib.pyplot as plt
import os
import numpy as np
import skimage.feature
import sklearn.neighbors as sn
import random

#dataset : 2 folders test and train
#           --> 7 emotions/folders withgreyscale image 48x48


def store_img(path):            # store images in a numpy array [[ img1 raster scan], ... [img n raster scan]]
    emotions=os.listdir(path)   #and his label [ 0 " Angry" ,0,......,6,6,6 "surprise" ]
    labels=[]
    imgs=[]
    for i in range(len(emotions)) : # for each emotion
        path_emotion=path+emotions[i]+'/'
        files=os.listdir(path_emotion)
        for file in files:          # for each image inside the folder
            img=plt.imread(path_emotion+file)
            img_lpb=skimage.feature.local_binary_pattern(img,8,24,method='default')
            img_lpb=np.reshape(img_lpb,2304)
            imgs.append(img_lpb)    #we save lpb images rasterscanned
            labels.append(i)        #and its label
    return labels,np.asarray(imgs)

def train(X,Y,k):         #train the k-nearest neigbors classifier
    classifier=sn.KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X,Y)
    return classifier

def test(classifier,indice,X,Y):        #test 1 image in the classifier [ return True if the classifier is right]
    X=np.reshape(X_test[indice], (1, -1))   #issues for size, find it on stackOverflow
    guess=classifier.predict(X)
    real=Y[indice]
    if (int(real)-int(guess[0]))==0:        #compare the label find by the algorithm and the real one.
        return True
    else :
        return False

def test_global(classifier,X,Y,n):      #test n images in the classifier [ return xx% of good classification]
    l = [random.randint(0, len(X)) for i in range(n)]
    score = 0
    for num in l :
        if test(classifier,num,X,Y):
            score+=1
    return score/n*100

if __name__ == '__main__':
    Y_train,X_train=store_img('archive/train/')
    Y_test,X_test=store_img('archive/test/')
    classi=train(X_train,Y_train,6)
    # print("train/angry number : ", labels.count(0))
    # print("train/disgust number : ", labels.count(1))
    # print("train/fear number : ", labels.count(2))
    # print("train/happy number : ", labels.count(3))
    # print("train/neutral number : ", labels.count(4))
    # print("train/sad number : ", labels.count(5))
    # print("train/surprise number : ", labels.count(6))
    print(test_global(classi,X_test,Y_test,60))





