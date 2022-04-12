from asyncio.windows_events import NULL
from itertools import count
import math
from turtle import distance
import numpy as np
from sklearn import datasets
import pandas as pd
import random
iris = datasets.load_iris(return_X_y=True)
iris_x = iris[0]



# K-MEANS ALGORITHM
class Datapoint:
    label = -1

    def __init__(self,v) :
        self.points = v
        
    def setLabel(self,n):
        self.label = n
    def getLabel(self):
        return self.label
    def __eq__(self,w) -> bool:
        for i in range(self.points):
            if not math.isclose(self.points[i],w,abs_tol=0.001):
                return False
    def __len__(self):
        return len(self.points)

data = [Datapoint(x) for x in iris_x]

class K_meansClassifier:
    centroids = []
    def __init__(self,k,dataset):
        self.k = k
        self.dataset = dataset
    
    def random_centroid(self):
        pass
    
    def init_centroids(self):
        for i in range(self.k):
           self.centroids.append(random.choice(self.dataset))
           self.centroids[i].label = i

    def distance(self,v,w):
        radicand = 0
        for i in range (len(v)):
           radicand += pow(v.points[i] - w.points[i],2)
        return math.sqrt(radicand)
    def getCentroids(self):
        return self.centroids


classifier = K_meansClassifier(3,data)
classifier.init_centroids()
centroids = classifier.getCentroids()
min_distance = 0.0
for i in range(len(data)):
    for j in range(len(centroids)):
        distance = classifier.distance(centroids[j],data[i])
        if j==0:
            min_distance = distance
            data[i].setLabel(centroids[j].getLabel())
        elif min_distance > distance:
            min_distance = distance
            data[i].setLabel(centroids[j].getLabel())
    

count_label_0 = 0
count_label_1 = 0
count_label_2 = 0

for i in range(len(data)):
    if data[i].getLabel() == 0:
        count_label_0 += 1  
    elif data[i].getLabel() == 1:
        count_label_1 += 1  
    elif data[i].getLabel() == 2:
        count_label_2 += 1 
print("label 0: "+str(count_label_0)+" label 1: "+str(count_label_1)+" label 2: "+str(count_label_2))
