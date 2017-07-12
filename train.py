import os
from model import image_classifier

if os.path.isdir('./Data/Train') and os.path.isdir('./Data/Val'):
    images = os.listdir('Data/Train')
    model = image_classifier(images, mode='train', resume=1)
    loss, inp_dict = model.build_training_graph()
    model.train(loss, inp_dict)
else:
    os.mkdir('./Data/Train')
    os.mkdir('./Data/Val')
    os.mkdir('./Data/Test')
    print "Please run download_data() from Utility.py"
