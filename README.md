# TensorFlow_image_classification
A basic convnet implementation using TensorFlow for classification task. 
Architecture:
Conv->Conv->Pool->Conv->Conv->Pool->Conv->Conv-Pool->fc1->fc2->150 softmax probabilities
Kernel size is 3x3 and batch normalization is performed after every conv operation.
Model only loads the current image batch into memory.
To run, have the true class of the image stored in its filename e.g. file#_class_label.jpg , see get_batch() in model.py
To train the model, run train.py after generating a split using split_data() in Utility.py
To test, add test images in Data/Test folder and run test.py
