import time
import numpy as np #Data is often stored as "Numpy Arrays"
import matplotlib.pyplot as plt #matplotlib.pyplot allows us to visualize results
import caffe #caffe is our deep learning framework, we'll learn a lot more about this later in this task.
%matplotlib inline

MODEL_JOB_DIR = '/dli/data/digits/20180301-185638-e918'  ## Remember to set this to be the job directory for your model
DATASET_JOB_DIR = '/dli/data/digits/20180222-165843-ada0'  ## Remember to set this to be the job directory for your dataset

MODEL_FILE = MODEL_JOB_DIR + '/deploy.prototxt'                 # This file contains the description of the network architecture
PRETRAINED = MODEL_JOB_DIR + '/snapshot_iter_735.caffemodel'    # This file contains the *weights* that were "learned" during training
MEAN_IMAGE = DATASET_JOB_DIR + '/mean.jpg'                      # This file contains the mean image of the entire dataset. Used to preprocess the data.

# Tell Caffe to use the GPU so it can take advantage of parallel processing.
# If you have a few hours, you're welcome to change gpu to cpu and see how much time it takes to deploy models in series.
caffe.set_mode_gpu()
# Initialize the Caffe model using the model trained in DIGITS
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# load the mean image from the file
mean_image = caffe.io.load_image(MEAN_IMAGE)
print("Ready to predict.")
# Choose a random image to test against
#RANDOM_IMAGE = str(np.random.randint(10))
IMAGE_FILE = '/dli/tasks/task5/task/images/LouieReady.png'
input_image= caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
plt.show()

# Load the input image into a numpy array and display it
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
plt.show()

# Calculate how many 256x256 grid squares are in the image
rows = input_image.shape[0] / 256
cols = input_image.shape[1] / 256

# Initialize an empty array for the detections
detections = np.zeros((rows, cols))

# Iterate over each grid square using the model to make a class prediction
start = time.time()
for i in range(0, rows):
    for j in range(0, cols):
        grid_square = input_image[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256]
        # subtract the mean image
        grid_square -= mean_image
        # make prediction
        prediction = net.predict([grid_square])
        detections[i, j] = prediction[0].argmax()
end = time.time()

# Display the predicted class for each grid square
plt.imshow(detections, interpolation=None)

# Display total time to perform inference
print
'Total inference time: ' + str(end - start) + ' seconds'

import caffe
import cv2
import sys


def deploy(img_path):
    MODEL_JOB_DIR = '/dli/data/digits/20190403-034002-0f93'
    ARCHITECTURE = MODEL_JOB_DIR + '/' + 'deploy.prototxt'
    WEIGHTS = MODEL_JOB_DIR + '/' + 'snapshot_iter_108.caffemodel'
    print("Filepath to Architecture = " + ARCHITECTURE)
    print("Filepath to weights = " + WEIGHTS)
    caffe.set_mode_gpu()

    # Initialize the Caffe model using the model trained in DIGITS. Which two files constitute your trained model?
    net = caffe.Classifier(ARCHITECTURE, WEIGHTS,
                           channel_swap=(2, 1, 0),
                           raw_scale=255)

    # Create an input that the network expects. This is different for each project, so don't worry about the exact steps, but find the dataset job directory to show you know that whatever preprocessing is done during training must also be done during deployment.
    input_image = caffe.io.load_image(img_path)
    input_image = cv2.resize(input_image, (256, 256))
    mean_image = caffe.io.load_image('/dli/data/digits/20190403-033640-947a/mean.jpg')
    input_image = input_image - mean_image

    # Make prediction. What is the function and the input to the function needed to make a prediction?
    prediction = net.predict([
                                 input_image])  ##REPLACE WITH THE FUNCTION THAT RETURNS THE OUTPUT OF THE NETWORK##([##REPLACE WITH THE INPUT TO THE FUNCTION##])

    # Create an output that is useful to a user. What is the condition that should return "whale" vs. "not whale"?
    if prediction.argmax() == 0:
        return "whale"
    else:
        return "not whale"


##Ignore this part
if __name__ == '__main__':
    print(deploy(sys.argv[1]))

