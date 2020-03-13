import os
import keras
import pydload
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import numpy as np

class Detector():
    detection_model = None
    classes = [
        'BELLY',
        'BUTTOCKS',
        'F_BREAST',
        'F_GENITALIA',
        'M_GENITALIA',
        'M_BREAST',
    ]
    
    def __init__(self):
        '''
            model = Detector()
        '''
        url = 'https://github.com/bedapudi6788/NudeNet/releases/download/v0/detector_model'
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, '.NudeNet/')
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        
        model_path = os.path.join(model_folder, 'detector')

        if not os.path.exists(model_path):
            print('Downloading the checkpoint to', model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)

        Detector.detection_model = models.load_model(model_path, backbone_name='resnet101')
    
    def detect(self, img_path, min_prob=0.6):
        image = read_image_bgr(img_path)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = Detector.detection_model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        processed_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            if label == 0 or label == 1 or label == 2 or label == 5:
                continue
            box = box.astype(int).tolist()
            label = Detector.classes[label]
            processed_boxes.append(box)
			
        return processed_boxes


if __name__ == '__main__':
    m = Detector('/Users/bedapudi/Desktop/inference_resnet50_csv_14.h5')
    print(m.censor('/Users/bedapudi/Desktop/n2.jpg', out_path='a.jpg'))
