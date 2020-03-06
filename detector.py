"""
detector

Copyright (c) 2020 Nathan Cueto
Licensed under the MIT License (see LICENSE for details)
Written by Nathan Cueto

"""

import os
import sys
import json
# import datetime # not really useful so remove soon pls
import numpy as np
import skimage.draw
import imgaug # should augment this improt as well haha
# from PIL import Image

# Root directory of project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
sys.path.insert(1, 'samples/hentai/')
from hentai import HentaiConfig

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights
WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights.h5")

class Detector():
    # at startup, dont create model yet
    def __init__(self, weights_path):
        class InferenceConfig(HentaiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        self.config = InferenceConfig()

        self.weights_path = weights_path
        # counts how many non-png images, if >1 then warn user
        self.dcp_compat = 0
        # keep model loading to be done later, not now

    # Make sure this is called before using model weights
    def load_weights(self):
        self.model = modellib.MaskRCNN(mode="inference", config=self.config,
                                      model_dir=DEFAULT_LOGS_DIR)
        self.model.load_weights(self.weights_path, by_name=True)

    def apply_cover(self, image, mask):
        """Apply cover over image. Based off of Mask-RCNN Balloon color splash function
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]

        Returns result covered image.
        """
        # Copy color pixels from the original color image where mask is set
        # green = np.array([[[0, 255, 0]]], dtype=np.uint8)
        # print('apply_cover: shape of image is',image.shape)
        green = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
        green[:,:] = [0, 255, 0]
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) < 1)
            cover = np.where(mask, image, green).astype(np.uint8)
        else:
            # error case, return image
            cover = green.astype(np.uint8)
        return cover

    def get_non_png(self):
        return self.dcp_compat

    def video_create(self, image_path=None, dcp_path=''):
        assert image_path
        
        from cv2 import VideoCapture, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc
        # Video capture to get shapes and stats
        # Only supports 1 video at a time, but this can still get mp4 only
        
        vid_list = []
        for file in os.listdir(image_path):
            if file.endswith('mp4') or file.endswith('MP4'):
                vid_list.append(image_path + '/' + file)
        
        video_path = vid_list[0] # ONLY works with 1 video for now
        vcapture = VideoCapture(video_path)
        width = int(vcapture.get(CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(CAP_PROP_FPS)

        # Define codec and create video writer, video output is purely for debugging and educational purpose. Not used in decensoring.
        file_name = fname + "_out.avi"
        vwriter = VideoWriter(file_name,
                                    VideoWriter_fourcc(*'MJPG'),
                                    fps, (width, height))
        count = 0
        print("Beginning build. Do ensure only relevant images are in source directory")
        input_path = dcp_path + '/decensor_output/'
        img_list = []
        # output of the video detection should be in order anyway
        # os.chdir(input_path)
        # files = filter(os.path.isfile, os.listdir(input_path))
        # files = [os.path.join( f) for f in files]    
        # files.sort(key=lambda x: os.path.getmtime(x))
        # for file in files:
        for file in os.listdir(input_path):
            # TODO: check what other filetpyes supported
            if file.endswith('.png') or file.endswith('.PNG'):
                img_list.append(input_path  + file)
                print('adding image ', input_path  + file)
        for img in img_list:
            print("frame: ", count)
            # Read next image
            image = skimage.io.imread(img)
            # Add image to video writer, after flipping R and B value
            image = image[..., ::-1]
            vwriter.write(image)
            count += 1

        vwriter.release()
        print('video complete')

    # save path and orig video folder are both paths, but orig video folder is for original mosaics to be saved.
    # fname = filename.
    # image_path = path of input file, image or video
    def detect_and_cover(self, image_path=None, fname=None, save_path='', is_video=False, orig_video_folder=None):
        assert image_path
        assert fname # replace these with something better?
        
        if is_video: # TODO: video capabilities will come later
            # from cv2 import VideoCapture, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc
            import cv2
            # Video capture
            video_path = image_path
            vcapture = VideoCapture(video_path)
            width = int(vcapture.get(CAP_PROP_FRAME_WIDTH))
            height = int(vcapture.get(CAP_PROP_FRAME_HEIGHT))
            fps = vcapture.get(CAP_PROP_FPS)
    
            # Define codec and create video writer, video output is purely for debugging and educational purpose. Not used in decensoring.
            file_name = fname + "_mask.avi"
            vwriter = VideoWriter(file_name,
                                      VideoWriter_fourcc(*'MJPG'),
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
                    # save frame into decensor input original. Need to keep names persistent.
                    im_name = fname[:-4] # if we get this far, we definitely have a .mp4. Remove that, add count and .png ending
                    file_name = orig_video_folder + im_name + str(count).zfill(6) + '.png' # NOTE Should be adequite for having 10^6 frames, which is more than enough for even 30 mintues total.
                    
                    print('saving frame as ', file_name)
                    skimage.io.imsave(file_name, image)
                    # Detect objects
                    r = self.model.detect([image], verbose=0)[0]
                    # Color splash
                    splash = self.apply_cover(image, r['masks'])
                    
                    # save covered frame into input for decensoring path
                    file_name = save_path + im_name + str(count).zfill(7) + '.png'
                    print('saving covered frame as ', file_name)
                    skimage.io.imsave(file_name, splash)

                    # RGB -> BGR to save image to video
                    splash = splash[..., ::-1]
                    # Add image to video writer
                    vwriter.write(splash)
                    count += 1

            vwriter.release()
            print('video complete')
        else:
            print("Running on {}".format(image_path))
            # Read image
            image = skimage.io.imread(image_path)
            # Detect objects
            r = self.model.detect([image], verbose=0)[0]
            # Color splash
            cov = self.apply_cover(image, r['masks'])
            # Save output
            file_name = save_path + fname
            skimage.io.imsave(file_name, cov)
            print("Saved to ", file_name)

    # return 0 means all good
    # return 1 means jpgs found
    def run_on_folder(self, input_folder, output_folder, is_video=False, orig_video_folder=None):
        assert input_folder
        assert output_folder # replace with catches and popups

        if(is_video == True):
            # support for multiple videos if your computer can even handle that
            vid_list = []
            for file in os.listdir(input_folder):
                if file.endswith('mp4') or file.endswith('MP4'):
                    vid_list.append((input_folder + '/' + file, file))
            
            for vid_path, vid_name in vid_list:
                self.detect_and_cover(vid_path, vid_name, output_folder, is_video=True, orig_video_folder=orig_video_folder), 
        else:
            # obtain inputs from the input folder
            img_list = []
            for file in os.listdir(input_folder):
                # TODO: check what other filetpyes supported
                if file.endswith('.png') or file.endswith('.PNG'):
                    img_list.append((input_folder + '/' + file, file))
                elif file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg"):
                    # img_list.append((input_folder + '/' + file, file)) # Do not add jpgs. Conversion to png must happen first
                    self.dcp_compat += 1

            # save run detection with outputs to output folder
            for img_path, img_name in img_list:
                self.detect_and_cover(img_path, img_name, output_folder)
            # return 0



# main only used for debugging here. Comment out pls
'''if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Utilize Mask R-CNN to detect censor bars.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights.h5")
    parser.add_argument('--imagedir', required=True,
                        metavar="path to image folder",
                        help='Folder of images to apply mask coverage on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    args = parser.parse_args()

    weights_path = args.weights
    images_path = args.imagedir
    output_dir = "temp_out/"

    print('Initializing Detector class')
    detect_instance = Detector(weights_path=args.weights)
    print('loading weights')
    detect_instance.load_weights()
    print('running detect on in and out folder')
    detect_instance.run_on_folder(input_folder=images_path, output_folder=output_dir)

    print("Fin")'''
