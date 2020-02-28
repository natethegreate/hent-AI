# Detecting censors with Mask R-CNN

Illustrated adult content created in Japan is required to be censored by law. Two common types of censoring involves censor bars and mosaic blurs. For us degenerated living outside of Japan, this means we are also subject to the bars and mosaics. There is a solution, [DeepCreamPy](https://github.com/deeppomf/DeepCreamPy) by deeppomf that can draw over the censors, given that you tell it where the censors are. That is a long and painstaking process, so I hope to automate that process with this project. This project will utilize deep learning and image segmentation, techniques typically used in autonomous vehicles and computer vision tasks. 

This is an implementation of Matterposrt's [Mask R-CNN](https://arxiv.org/abs/1703.06870), modified to my liking. 

NOTE: Despite the title, this project does not actually use AI.

SFW example of image segmentation:
![Instance Segmentation Sample](assets/street.png)

# Getting Started
You will need all the same requirements as matterport's Mask RCNN implementation, nothing more. Note that I am using tensorflow 1.5.0, tensorflow-gpu 1.9.0, and keras 2.2.0. I have not been able to get newer combinations stable.

* [inspect_model.ipynb](samples/coco/inspect_model.ipynb) This notebook is identical to the balloon notebook. I modified it to work with this project instead, and it is best used to inspect a model. For detailed logging, use Tensorboard (which should be installed if you have tensorflow)

* [inspect_weights.ipynb](samples/coco/inspect_weights.ipynb)
Same thing as above, except this notebook is used to validate the dataset. Also has cool information showing some of the quirks and features of MaskRcnn

I have only worked on Windows platforms, and had not been able to train or work on other instances like Google colab and Google Cloud. 


# The Dataset

Extended the existing Balloon class to support 3 classes: BG, bar, and mosaic. I have decided to not provide my dataset. Annotated with VGG annotator in .json format.

The color_splash function will be overwritten to instead return a full green mask over the returned rpn mask. This will ensure compatibility with the DeepCreamPy framework.

# The Model

I experimented with other pre-trained models, but ended transfer learning with the imagenet model. 

I have a prototype model with 45 epochs available [here](https://drive.google.com/open?id=1u8I-oRKxe8Mx8wENVkccliOvSj4MEr45). I will continue adding more trained models as I continue training.

Further configuration changes, and likely more training is inevitable, and I may choose to start fresh with a new model.


## Contributing
I only have a bare understanding of convolutional nueral networks and deep learning as a whole. Contributions and improvements to this repo are welcome.

## Requirements
I would reccomend running these on a virtual environment.
Python 3.5, TensorFlow 1.5, Keras 2.2, tensorflow-gpu 1.9.0, and other common packages listed in `requirements.txt`.


## Installation

* After cloning this repo, first install the requirements:

```
pip install -r requirements.txt
```

* Next, compile maskrcnn:

```
python setup.py install
```

* To train, run

```
python samples\hentai\hentai.py train --dataset=dataset_img/ --weights=path/to/weights
```
Alternatively, you can resume training using --weights=last



# Acknowledgements
Inspiration from [DeepCreamPy](https://github.com/deeppomf/DeepCreamPy)
Mask Rcnn implementation from [Matterport](https://github.com/matterport/Mask_RCNN)
Obtained weights from mattya's [chainer-DCGAN]( https://github.com/mattya/chainer-DCGAN)
Dataset annotated with [VGG annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via.html)
Dataset created with numerous doujins and hentai
