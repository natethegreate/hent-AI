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
    print('do something')

# random bar color, black or white with ranges. TODO: Maybe add slight tone variation as well?
def rand_color():
    bar_mode = random.choice('white', 'black')
    if(bar_mode == 'white'):
        r = random.randint(235,255) # for now, all same color so no slight color tones.
        g = r
        b = r
    elif(bar_mode == 'black'):
        r = random.randint(0,60)
        g = r
        b = r
    return r, g, b


# sourced from https://medium.com/@richardpricejones/drawing-a-rectangle-with-a-angle-using-opencv-c9284eae3380
'''
x0, y0: center, must be chosen randomly within lewd region by parent caller
width, height: should be generated randomly, width > height, by parent caller.
angle: angle in degrees
img: image, loaded via opencv probably
file_name: file name of img, used for annotation. md5 pls
'''
def draw_angled_rec(x0, y0, width, height, angle, img, annotation_file):

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
    random_color = rand_color()
    cv2.fillConvexPoly(img, points, random_color)

'''
data_gen- Generate bar censors, or a sequence of bar censors in lewd region
min/max x and y: Boundaries of lewd region
'''
def data_gen(img, min_x, max_x, min_y, max_y):
    filename = 'generated_bar_anno'
    # Choose generation mode: random individual bars, line sequence of bars, or bars that are close/merging with each other. Can add more modes or remove modes idk
    # mode = random.choice('individual', 'sequence', 'merged')
    mode = random.choice('individual', 'merged') # not sure how to approach sequence yet
    if mode=='individual': 
        ## get random number of bars
        num_bars = random.randint(1, 3)

        ## get random angle, dimensitons and center for each bar
        ## call draw_angled rect on bars
        for bar in num_bars:
            bar_angle = random.randint(0, 180)
            bar_height = random.randint(8, 30)
            bar_width = random.randint(bar_height*2, bar_height*6) # this dimension MUST be much longer
            bar_x = random.randint(min_x, max_x)
            bar_y = random.randint(min_y, max_y)
            draw_angled_rec(bar_x, bar_y, bar_width, bar_height, bar_angle, img, filename)
    elif mode=='sequence': 
        ## choose 2 points, for a line. This is the 'shaft'
        ## choose random number of bars
        num_bars = random.randint(1, 5)

        ## get random angle, dimensions and center for each bar
        ## place bars on the shaft
    elif mode=='merged': 
        ## choose random number of bars
        num_bars = random.randint(2, 5)

        ## choose center of leftmost bar, every bar will add on to the right
        center_x = random.randint(min_x, max_x)
        center_y = random.randint(min_y, max_y)
        ## put remaining bars nearby this bar to the right
        for bar in num_bars:
            bar_angle = random.randint(0, 180)
            bar_height = random.randint(8, 30)
            bar_width = random.randint(bar_height*2, bar_height*6) # this dimension MUST be much longer
            bar_x = center_x + random.randint(25, 60) # 25 to 60 pixels away from center
            bar_y = center_y + random.randint(25, 60)
            if(bar_x > max_x): # make sure added bars not outside lewd zone, or not outside image
                bar_x = max_x
            if(bar_y > max_y):
                bar_y = max_y
            draw_angled_rec(bar_x, bar_y, bar_width, bar_height, bar_angle, img, filename)





if __name__ == "__main__":
    for img in os.listdir('.'):
        if img.endswith('.png') or img.endswith('.PNG') or img.endswith('.jpg') or img.endswith('.JPG'):
            print('Generating bars for',img)
            ## Run NudeNet or something to get lewd region
            min_x, max_x, min_y, max_y = 0, 10, 0, 10 # temp numbers 
            ## Generate data with this
            data_gen(img, min_x, max_x, min_y, max_y)
    pass
