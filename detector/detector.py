"""
detector

Copyright (c) 2020 Nathan Cueto
Licensed under the MIT License (see LICENSE for details)
Written by Nathan Cueto

"""

import os
import sys
# import json
# import datetime
import numpy as np
import skimage.draw
# import imgaug
from PIL import Image

# Root directory of project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights
WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights.h5")

# class Detector():
def apply_cover(image, mask):
    """Apply cover over image. Based off of Mask-RCNN Balloon color splash function
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result covered image.
    """
    # Copy color pixels from the original color image where mask is set
    # green = np.array([[[0, 255, 0]]], dtype=np.uint8)
    print('apply_cover: shape of image is',image.shape)
    green = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
    green[:,:] = [0, 255, 0]
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        cover = np.where(mask, image, green).astype(np.uint8)
    else:
        # error case, return image
        cover = green.astype(np.uint8)
    return cover


def detect_and_cover(model, image_path=None):
    assert image_path
    # Image or video?
    # if image_path:
        # Run model detection and generate the color splash effect
    print("Running on {}".format(args.image))
    # Read image
    image = skimage.io.imread(args.image)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    splash = apply_cover(image, r['masks'])
    # Save output
    file_name = "cover_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)
    '''elif video_path: # TODO: video capabilities will come later
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()'''
    print("Saved to ", file_name)

# main only used for debugging here. Comment out in prod
if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect censor bars.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="just 'cover'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate argument
    if args.command == "cover":
        assert args.image

    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    if args.command == "cover":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'cover'".format(args.command))
