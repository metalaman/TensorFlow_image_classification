import os
from model import image_classifier

images = os.listdir('Data/Test')
model = image_classifier(images, mode='test')
test_batch, predicted_class = model.build_test_graph()
model.test(test_batch, predicted_class)
