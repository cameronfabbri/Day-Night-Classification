# Day and Night

This compares the difference of using a neural network and a support vector machine
for classifying images based on color. The exact application is differentiating
between pictures taken during the day and pictures taken during the night.


### Data

Both scripts assume that the data is all in a root folder, with each subfolder
(being a label) containing images. Ex.


images/<br>
   day/<br>
      image1.png<br>
      image2.png<br>
      ...<br>
   night/<br>
      image1.png<br>
      image2.png<br>
      ...<br>


### Feature computing

A complex feature extraction is not needed for classifying images based on color.
This uses the image's RGB matrix flattened and normalized, i.e each entry in
the flattened matrix is divided by the mean of all the entries.


### Comparison

The SVM takes a much shorter time to train, and does much better in testing. 
The SVM gets about 89% accuracy whereas the neural network gets about 78% accuracy,
both resizing the images to 30x30.


### Usage

`python neural.py images/`<br>
`python svm.py images/`<br>

`neural_model.pkl` and `svm_model.pkl` respectively will be saved to the
same directory and loaded automatically if it is detected.


