#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import math
import csv
import numpy as np
import random
import os
import glob
from PIL import Image
from nudenet import NudeDetector    #from NudeNet_edited import Detector
detector = NudeDetector()    #detector = Detector()

#You can change those folder paths
rootdir = "./decensor_input"
outdir = "./decensor_input_original"
os.makedirs(rootdir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)

rgbvals = 255, 255, 255

files = glob.glob(rootdir + '/**/*.png', recursive=True)
files_jpg = glob.glob(rootdir + '/**/*.jpg', recursive=True)
files.extend(files_jpg)
err_files=[]

def rand_color():
    #color variation on 0, 1, or 2 of the 3 values. 
    variation2 = random.randrange(0,1)
    color_var = 0
    if(random.random() >=.5): # half chance for white 
        r = random.randrange(239,255)  
        g = r
        b = r
        var_amnt = random.randrange(1,25) # ~half chance of >16, which is guaranteed overflow, so variation doesnt happen too much.
        if r + var_amnt > 255:
            var_amnt = 0 # cancel variation in case of overflow. 
            # print('canceled')
            return r, g, b # Early cancellation
        if variation2 == 1: # Case where we vary 2 of the 3 values
            color_var = random.randrange(0,2)   # r & g, g & b, or r & b
            # print('variation')
            if color_var == 0:
                r += var_amnt
                g += var_amnt
            elif color_var == 1:
                b += var_amnt
                g += var_amnt
            elif color_var == 2:
                r += var_amnt
                b += var_amnt
        else:               # case where we vary only 1 of the 3 values
            color_var = random.randrange(0,2)   # r, g, or b
            # print('variation')
            if color_var == 0:
                r += var_amnt
            elif color_var == 1:
                g += var_amnt
            elif color_var == 2:
                b += var_amnt
        return r, g, b
    else: #half chance for black
        r = random.randrange(0, 50)
        g = r
        b = r
        var_amnt = random.randrange(10,70) # same idea, part of range will guarantee no variation
        if r + var_amnt > 50:
            var_amnt = 0
            print('canceled')
            return r, g, b # Early cancellation
        if variation2 == 1: # Case where we vary 2 of the 3 values
            color_var = random.randrange(0,2) # r & g, g & b, or r & b

            if color_var == 0:
                r += var_amnt
                g += var_amnt
            elif color_var == 1:
                b += var_amnt
                g += var_amnt
            elif color_var == 2:
                r += var_amnt
                b += var_amnt
        else:
            color_var = random.randrange(0,2) # r, g, or b
            if color_var == 0:
                r += var_amnt
            elif color_var == 1:
                g += var_amnt
            elif color_var == 2:
                b += var_amnt
        return r, g, b
    return 0,255,255 # bug color

def draw_angled_rec(x0, y0, width, height, angle, img, color):
    points = []
    points2 = []
    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    #print(str(b) + ", " + str(a) + " - cos, sin. Angle - " + str(_angle))    #DEBUG

    # draw points with slightly smaller dimenstions, width and height difference due to different scaling
    height_s = height -3
    width_s = width -3
    # also decreas the scale of b in the x calculation
    x1 = [int(x0 - a * height_s - b * width_s), int(y0 + b * height_s - a * width_s)]
    y1 = [int(x0 + a * height_s - b * width_s), int(y0 - b * height_s - a * width_s)]
    x2 = [int(2 * x0 - x1[0]), int(2 * y0 - x1[1])]
    y2 = [int(2 * x0 - y1[0]), int(2 * y0 - y1[1])]
    points = np.array((x1, y1, x2, y2))

    # original size
    x1s = [int(x0 - a * height - b * width), int(y0 + b * height - a * width)]
    y1s = [int(x0 + a * height - b * width), int(y0 - b * height - a * width)]
    x2s = [int(2 * x0 - x1s[0]), int(2 * y0 - x1s[1])]
    y2s = [int(2 * x0 - y1s[0]), int(2 * y0 - y1s[1])]
    points2 = np.array((x1s, y1s, x2s, y2s))

    # print(points)
    # print(points2)
    ## Random color function - Want multiple shades of dark-grey to black, and white to super light grey
    r, g, b = color
    cv2.fillConvexPoly(img, points, color=(r, g, b))
    # send original points
    return(points2)

#Working with files
with open('example.csv', 'w', newline='', encoding='utf-8') as f_output:     #CSV
    csv_output = csv.writer(f_output, quoting=csv.QUOTE_NONE, quotechar="", delimiter=",", escapechar=' ')     #CSV
    csv_output.writerow(['filename','file_size','file_attributes','region_count','region_id','region_shape_attributes','region_attributes'])     #CSV
    for f in files:
        try:
            while True:
                print("Working on " + f)
                img_C = Image.open(f).convert("RGB")
                x, y = img_C.size
                card = np.array(Image.new("RGB", (x, y), (rgbvals)))
                img_C = np.array(img_C) 
                img_rgb = img_C[:, :, ::-1].copy() 
                
                color = rand_color()

                detection = detector.detect(f)
                label=['F_GENITALIA', 'M_GENITALIA']#
                all_regions = [i['box'] for i in detection if i['label'] in label]#
                if(all_regions == []):
                    # skip entire detection, avoid saving 
                    print('skipping image with failed nudenet detection')
                    break
                print(all_regions)#
                
                points = []
                comp_array = []
                for region in all_regions:
                    min_x, min_y, max_x, max_y = region 

                    len_x = max_x-min_x
                    len_y = max_y-min_y
                    #thickness 3-15% from long side
                    #wideness 30-75% from short side
                    #score - 15-30% from area
                    #angle - +-15* from axis
                    area = len_x*len_y    #area of nudenet zone
                    score = random.triangular(area*0.15, area*0.3)    #maximal area for rectangles
                    i=0
                    while score >= area*0.03:
                        if len_x >= len_y:    #decide the longest side
                            # print("vertical bar")
                            thickness = random.triangular(len_x*0.03, len_x*0.15)    #thickness of the bar
                            wideness = random.triangular(len_y*0.3, len_y*0.75)    #wideness of the bar
                            angle = 0    #axis
                            bar_x = int(random.uniform(min_x, max_x))    #random bar_x
                            bar_y = int(random.triangular(min_y, max_y))#, min_y+(max_y-min_y)/2-wideness/2))    #random bar_y
                            #print(bar_x, bar_y)
                            comp_area = list(range(bar_x, bar_x+int(len_x*0.1),1))
                        else:
                            # print("horisontal bar")
                            thickness = random.triangular(len_y*0.03, len_y*0.15)    #thickness of the bar
                            wideness = random.triangular(len_x*0.3, len_x*0.75)    #wideness of the bar
                            angle = 90    #axis
                            bar_x = int(random.triangular(min_x, max_x))#, min_x+(max_x-min_x)/2-wideness/2))    #random bar_x
                            bar_y = int(random.uniform(min_y, max_y))    #random bar_y
                            #print(bar_x, bar_y)
                            comp_area = list(range(bar_y, bar_y+int(len_y*0.1),1))
                        if thickness*wideness <= score + area*0.02:
                            rotate = random.randint(angle-15, angle+15)    #random angle within 15% from axis
                            #print(rotate)
                            if rotate < 0:
                                rotate += 360
                            if not any(check in comp_area for check in comp_array):
                                comp_array = comp_array + comp_area
                                points.append(draw_angled_rec(bar_x, bar_y, thickness, wideness, rotate, img_rgb, color))
                                score -= thickness*wideness    #subtract last rectangle from maximal area for rectangles
                            else:    #recursion prevention
                                i += 1
                                if i == 25:
                                    print(str(score) + " of area left")
                                    break
                    #print(points)

                    output1x = []
                    output1y = []
                    for idx,conturJ in enumerate(points):
                        outputX = []
                        outputY = []
                        for idx2,temp in enumerate(conturJ):
                            xt, yt = temp
                            outputX.append(xt)
                            outputY.append(yt)
                        output1x.append(outputX)
                        output1y.append(outputY)
                    NudeNet_regions = zip(output1x, output1y)

                #Save file
                f=f.replace(rootdir, outdir, 1)
                os.makedirs(os.path.dirname(f), exist_ok=True)
                cv2.imwrite('temp_out.png', img_rgb)     #still a hack for non-unicode names
                os.replace('temp_out.png', f)

                for idx,_ in enumerate(NudeNet_regions):
                    csv_output.writerow([os.path.basename(f), os.stat(f).st_size, '"{}"', len(output1x), idx, '"{""name"":""polygon""','""all_points_x"":' + str(output1x[idx]), '""all_points_y"":' + str(output1y[idx]) + '}"', '"{}"'])     #CSV
                break
        except Exception as Exception:
            err_files.append(os.path.basename(f) + ": " + str(Exception))
            pass

#Error list    
if err_files:
    print("\n" + "NudeNet failed: ") 
    for f in err_files:
        print(f)
