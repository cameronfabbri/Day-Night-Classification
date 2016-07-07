"""

Cameron Fabbri

"""

import sys
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import time
import cPickle as pickle

SHAPE = (30, 30)

def read_files(directory):
   print "Reading files..."
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

   print str(num_classes) + " classes"
   return np.asarray(feature_list), np.asarray(label_list)

def extract_feature(image_file):
   img = cv2.imread(image_file)
   img = cv2.resize(img, SHAPE, interpolation = cv2.INTER_CUBIC)
   img = img.flatten()
   img = img/np.mean(img)
   return img
   

if __name__ == "__main__":
   if len(sys.argv) < 2:
      print "Usage: python extract_features.py [image_folder]"
      exit()

   # Directory containing subfolders with images in them.
   directory = sys.argv[1]

   # generating two numpy arrays for features and labels
   feature_array, label_array = read_files(directory)

   # Splitting the data into test and training splits
   X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.2, random_state=42)

   # checking for model
   if os.path.isfile("svm_model.pkl"):
      print "Using previous model..."
      svm = open("svm_model.pkl", "rb")
   else:
      print "Fitting"

      # Fitting model
      svm = SVC()
      svm.fit(X_train, y_train)

      print "Saving model..."
      pickle.dump(svm, open("svm_model.pkl", "wb"))

   print "Testing...\n"
  
   right = 0
   total = 0
   for x, y in zip(X_test, y_test):
      x = x.reshape(1, -1)
      prediction = svm.predict(x)[0]

      if y == prediction:
         right += 1
      total += 1

   accuracy = float(right)/float(total)*100
   print str(accuracy) + "% accuracy"
