#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import random
import math
import csv
import numpy as np
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
err_files=[]

def draw_angled_rec(x0, y0, width, height, angle, img):
    points = []
    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5

    x1 = [int(x0 - a * height - b * width),
           int(y0 + b * height - a * width)]
    y1 = [int(x0 + a * height - b * width),
           int(y0 - b * height - a * width)]
    x2 = [int(2 * x0 - x1[0]), int(2 * y0 - x1[1])]
    y2 = [int(2 * x0 - y1[0]), int(2 * y0 - y1[1])]

    points = np.array((x1, y1, x2, y2))

    ## Call function here to write annotations to file_name using the 4 points above as a polygon
    #write_annotation(points, annotation_file)

    ## Random color function - Want multiple shades of dark-grey to black, and white to super light grey
    #random_color = rand_color()
    cv2.fillConvexPoly(img, points, 0, 255, 0)
    return(points)

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

                detection = detector.detect(f)
                label=['F_GENITALIA', 'M_GENITALIA']#
                all_regions = [i['box'] for i in detection if i['label'] in label]#
                print(all_regions)#

                points = []
                for region in all_regions:
                    min_x, min_y, max_x, max_y = region 

                    len_x = max_x-min_x
                    len_y = max_y-min_y
                    #thickness 3-15%
                    #wideness 30-75%
                    #area 12-33%
                    #angle - +-15* from axis
                    area = len_x*len_y
                    score = random.uniform(area*0.12, area*0.33)
                    while score >= area*0.03:
                        if len_x >= len_y:
                            thickness = random.uniform(len_x*0.03, len_x*0.15)
                            wideness = random.uniform(len_y*0.3, len_y*0.75)
                            angle = 90
                        else:
                            thickness = random.uniform(len_y*0.3, len_y*0.75)
                            wideness = random.uniform(len_x*0.03, len_x*0.15)
                            angle = 0
                        if thickness*wideness <= score + area*0.02:
                            rotate = random.randint(angle-15, angle+15)
                            bar_x = random.randint(min_x, max_x)
                            bar_y = random.randint(min_y, max_y)
                            points.append(draw_angled_rec(bar_x, bar_y, thickness, wideness, rotate, img_rgb))
                            score -= thickness*wideness
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
