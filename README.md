# Mask R-CNN for Object Detection and Segmentation

This is an implementation of Matterposrt's [Mask R-CNN](https://arxiv.org/abs/1703.06870), which uses Python 3, Keras 2.2, and TensorFlow 1.5, based on FPN and Resnet 101 backbone.

Unrelated image
![Instance Segmentation Sample](assets/street.png)

# Getting Started
Not yet

* [inspect_model.ipynb](samples/coco/inspect_model.ipynb) This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline. Mostly unchanged from Matterports implementation, and is kept here for educational purposes.

* [inspect_weights.ipynb](samples/coco/inspect_weights.ipynb)
This notebooks inspects the weights of a trained model and looks for anomalies and odd patterns, and is mostly unchanged from Matterports implementation.


# The Dataset

In summary, to train the model on your own dataset you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

See examples in `samples/shapes/train_shapes.ipynb`, `samples/coco/coco.py`, `samples/balloon/balloon.py`, and `samples/nucleus/nucleus.py`.


## Contributing
I only have a bare understanding of convolutional nueral networks and deep learning as a whole. Contributions and improvements to this repo are welcome.

## Requirements
I would reccomend running these on a virtual environment.
Python 3.5, TensorFlow 1.5, Keras 2.2 (Not sure which one I have but I had to revert to some earlier version) and other common packages listed in `requirements.txt`.


## Installation
WIP

# Acknowledgements
Inspiration from [DeepCreamPy](https://github.com/deeppomf/DeepCreamPy)
Mask Rcnn implementation from [Matterport](https://github.com/matterport/Mask_RCNN)
Obtained weights from mattya's [chainer-DCGAN]( https://github.com/mattya/chainer-DCGAN)
Dataset annotated with [VGG annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via.html)
Dataset created with numerous doujins and hentai
