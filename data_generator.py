# Feb 2020 - hent-AI
# Script for data and annotation generation
# import sys
import math
import random
from os import listdir, system

'''
points: points of polygon
annotation file: name of csv or json for vgg annotations
'''
def write_annotation(points, annotation_file):
    # something csv or json idk


# sourced from https://medium.com/@richardpricejones/drawing-a-rectangle-with-a-angle-using-opencv-c9284eae3380
'''
x0, y0: center, must be chosen randomly within lewd region by parent caller
width, height: should be generated randomly, width > height, by parent caller.
angle: angle
img: image, loaded via opencv probably
file_name: file name of img, used for annotation. md5 pls
'''
def draw_angled_rec(x0, y0, width, height, angle, img, file_name, annotation_file):

    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5

    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    points = [pt0, pt1, pt2, pt3]

    ## Call function here to write annotations to file_name using the 4 points above as a polygon
    write_annotation(points, annotation_file)

    ## Random color function - Want multiple shades of dark-grey to black, and white to super light grey
    # random_color = 
    cv2.fillConvexPoly(img, points, random_color)

'''
data_gen- Generate bar censors, or a sequence of bar censors in lewd region
min/max x and y: Boundaries of lewd region
'''
def data_gen(min_x, max_x, min_y, max_y):
    # Choose generation mode: random individual bars, line sequence of bars, or bars that are close/merging with each other. Can add more modes or remove modes idk
    mode = random.choice('individual', 'sequence', 'merged')
    if mode=='individual': 
        ## get random number of bars

        ## get random angle, dimensitons and center for each bar
        ## call draw_angled rect on bars
    elif mode=='sequence': 
        ## choose 2 points, for a line. This is the 'shaft'
        ## choose random number of bars

        ## get random angle, dimensions and center for each bar
        ## place bars on the shaft
    elif mode=='merged': 
        ## choose random number of bars

        ## choose center of a bar
        ## put remaining bars nearby this bar






if __name__ == "__main__":
    for img in os.listdir('.'):
        if img.endswith('.png') or img.endswith('.PNG') or img.endswith('.jpg') or img.endswith('.JPG'):
            print('Generating bars for',img)
            ## Run NudeNet or something to get lewd region
            min_x, max_x, min_y, max_y = #
            ## Generate data with this
            datagen(min_x, max_x, min_y, max_y)
    pass
