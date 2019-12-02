"""
Usage: To train the model. 
Change the dataset directory or pre trained weight if needed.
"""

import os
import sys
import json
import numpy as np
import skimage.draw
from config import Config
import utils
import model as modellib


# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class SwapbodyConfig(Config):
    """Configuration for training on the swapbody  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "swapbody"

    # We use a GPU with 12GB memory, which can fit two images. 

    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 3  # Background + back + side

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class SwapbodyDataset(utils.Dataset):

    def load_swapbody(self, dataset_dir, subset):
        """Load a subset of the Swapbody dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have two class to add.
        self.add_class("swapbody", 1, "back")
        self.add_class("swapbody", 2, "side")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        data_list = os.listdir(dataset_dir)

        json_ids = [i for i in data_list if '.json' in i]

        # Add images
        for json_id in json_ids:
            # Testing for N images. N jpg files and N corresponding json files
            # assuming they have same name with different extension and same chronological order.
            json_file_path = os.path.join(dataset_dir, json_id)
            jsonfile = json.load(open(json_file_path))

            polygons = jsonfile['shapes']

            image_path = os.path.join(dataset_dir, jsonfile['imagePath'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "swapbody",
                image_id=jsonfile['imagePath'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Swapbody dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "swapbody":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        class_ids = []
        class_id = {'back':1, 'side':2, '__ignore__':0}

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them ACCORDING TO THE LABEL
            # BACK = 1, SIDE = 2

            points = p['points']
            x_points = [point[0] for point in points]
            y_points = [point[1] for point in points]
            row, column = skimage.draw.polygon(y_points, x_points)
            label = p['label']
            mask[row, column, i] = 1

            class_ids.append(class_id[label])


        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = SwapbodyDataset()
    dataset_train.load_swapbody(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SwapbodyDataset()
    dataset_val.load_swapbody(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # a pre-trained weights(like COCO), we don't need to train too long.
    # Also, no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect swapbodys.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/swapbody/dataset/",
                        help='Directory of the Swapbody dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()


    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = SwapbodyConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    train(model)
    