"""

Cameron Fabbri

Simple neural network implementation for classifying images.
Simply provide the folder for which your images are stored in.

Folder structure should have images for each class in a seperate
folder. Example

images/
   cat/
      image1.jpg
      image2.jpg
      ...
   dog/
      ...

"""

import sys
import os
import cv2
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from sklearn.cross_validation import train_test_split
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from sklearn import preprocessing
import cPickle as pickle

SHAPE = (30,30)

"""
Converts text labels to numbers, i.e cat -> 0, dog -> 1, ...
"""
def convertLabels(label_list):
   
   num_labels = len(label_list)
   
   pre = preprocessing.LabelEncoder()

   label_list = pre.fit_transform(label_list)

   return label_list
   
def read_files(directory):
   s = 1
   feature_list = list()
   label_list   = list()
   num_classes = 0
   for root, dirs, files in os.walk(directory):
      for d in dirs:
         num_classes += 1
         images = os.listdir(root+d)
         for image in images:
            s += 1
            label_list.append(d)
            feature_list.append(extract_feature(root+d+"/"+image))

   label_list = convertLabels(label_list)
   print str(num_classes) + " classes"
   return np.asarray(feature_list), np.asarray(label_list), num_classes


def extract_feature(image_file):
   img = cv2.imread(image_file)
   img = cv2.resize(img, SHAPE, interpolation = cv2.INTER_CUBIC)
   img = img.flatten()
   img = img/np.mean(img)
   return img


if __name__ == "__main__":
   
   if len(sys.argv) < 2:
      print "Usage: python neural.py [image folder]"
      exit()

   image_folder = sys.argv[1]

   features, labels, num_classes = read_files(image_folder)

   X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=12)

   net = buildNetwork(SHAPE[0]*SHAPE[1]*3, 15000, num_classes, bias=True, outclass=SoftmaxLayer)

   train_ds = SupervisedDataSet(SHAPE[0]*SHAPE[1]*3, num_classes)
   test_ds  = SupervisedDataSet(SHAPE[0]*SHAPE[1]*3, num_classes)

   for feature, label in zip(X_train, y_train):
      train_ds.addSample(feature, label)
   
   for feature, label in zip(X_test, y_test):
      test_ds.addSample(feature, label)

   # checking for model
   if os.path.isfile("neural_model.pkl"):
      print "Using previous model..."
      trainer = pickle.load(open("neural_model.pkl", "rb"))
   else:
      print "Training"
      trainer = BackpropTrainer(net, train_ds, momentum=0.1, verbose=True, weightdecay=0.01)
      trainer.train()

      print "Saving model..."
      pickle.dump(trainer, open("neural_model.pkl", "wb"))


   correct_count = 0
   total_count   = 0

   print "Testing..."
   for feature, label in zip(X_test, y_test):

      prediction = net.activate(feature).argmax(axis=0)

      if prediction == label:
         correct_count += 1
      total_count += 1

   print
   print str((float(correct_count)/float(total_count))*100) + "% correct"
