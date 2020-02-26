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

Extended the existing Balloon class to support 3 classes: BG, bar, and mosaic. I have decided to not provide my dataset.

# The Model

I have a model with 45 epochs available [here](https://drive.google.com/open?id=1u8I-oRKxe8Mx8wENVkccliOvSj4MEr45).

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
