import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io
import cv2 as cv
import visualize
import model as modellib
import swapbody
import mask2polygons


ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

config = swapbody.SwapbodyConfig()
SWAPBODY_DIR = os.path.join(ROOT_DIR, "dataset")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


dataset = swapbody.SwapbodyDataset()
dataset.load_swapbody(SWAPBODY_DIR, "val")

# Must call before using the dataset
dataset.prepare()

# Set the trained weight path
weights_path = os.path.join(MODEL_DIR,"swapbody_azure_20200114T1152/mask_rcnn_swapbody_0043.h5")

"""Device to load the neural network on. Useful if you're training a model on the same machine, 
in which case use CPU and leave the GPU for training."""

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
TEST_MODE = "inference"

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR, config=config)

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


focal_length = 2940 # in pixels = f*k
swapbody_height = 2.5 # in meters

video_name = "outputtest2.mp4"

vidcap = cv.VideoCapture(video_name)
success,image = vidcap.read()
height, width = image.shape[0:2]
size = (width,height)
img_array = []


while success:

    results = model.detect([image], verbose=1)   
    success,image = vidcap.read()

    # Display results
    ax = get_ax(1)
    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")

    masks = r['masks']
    N = masks.shape[2] 
    mask_dict = {}

    for i in range(N):

        name = 'mask'+str(i)
        my_mask = masks[:, :, i]*255
        mask_dict[name] = mask2polygons.Mask(my_mask)
        polyg_img = mask_dict[name].final_polygons(image)       
        mask_dict[name].show_distance(image, swapbody_height, focal_length)
    
    img_array.append(image)
    success,image = vidcap.read()
    print('Reading a new frame: ', success)

out = cv.VideoWriter('project.mp4', 0x00000021, 15, (1280,360) size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()