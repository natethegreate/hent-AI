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

files = glob.glob(rootdir + '/**/*.png', recursive=True)
err_files=[]

def pixelate(image, ratio, mosaic_kernel):
    # Get input size
    height, width, _ = image.shape
    # Desired "pixelated" size
    h, w = (mosaic_kernel, int(mosaic_kernel*ratio))
    # Resize image to "pixelated" size
    temp = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    # Initialize output image
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

with open('example.csv', 'w', newline='', encoding='utf-8') as f_output:     #CSV
    csv_output = csv.writer(f_output, quoting=csv.QUOTE_NONE, quotechar="", delimiter=",", escapechar=' ')     #CSV
    csv_output.writerow(['filename','file_size','file_attributes','region_count','region_id','region_shape_attributes','region_attributes'])     #CSV
    for f in files:
        try:
            while True:
                print("Working on " + f)
                img_C = Image.open(f).convert("RGB")
                x, y = img_C.size
                img_C = np.array(img_C) 
                image = img_C[:, :, ::-1].copy() 
                

                detection = detector.detect(f)
                label=['F_GENITALIA', 'M_GENITALIA']#
                all_regions = [i['box'] for i in detection if i['label'] in label]#
                print(all_regions)#
                mosaic_kernel = int(random.triangular(3, 35, 16))    #mosaic resolution
                if random.random() <= 0.5:    #probability for ajasting to ratio
                    calculate = True
                else:
                    calculate = False
                    ratio = 1
                points = []
                for region in all_regions:
                    min_x, min_y, max_x, max_y = region
                    
                    if calculate:
                        ratio = (max_x-min_x)/(max_y-min_y)
                    center = (int((max_x+min_x)*0.5), int((max_y+min_y)*0.5))
                    #print(center)
                    len_x = max_x-min_x
                    len_y = max_y-min_y
                    thickness = random.triangular(len_x*0.4, len_x, len_x*0.9)
                    wideness = random.triangular(len_y*0.4, len_y, len_y*0.9)
                    min_x = int(center[0] - thickness*0.5)
                    min_y = int(center[1] - wideness*0.5)
                    max_x = int(center[0] + thickness*0.5)
                    max_y = int(center[1] + wideness*0.5)
                    ROI = image[min_y:max_y, min_x:max_x]
                    pixelated_ROI = pixelate(ROI, ratio, mosaic_kernel)
                    image[min_y:max_y, min_x:max_x] = pixelated_ROI
                    points.append(np.array(((min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y))))
                    
                output1x = []
                output1y = []
                for conturJ in points:
                    outputX = []
                    outputY = []
                    it = iter(conturJ.flatten())
                    for x in it:
                        outputX.append(x)
                        outputY.append(next(it))
                    output1x.append(outputX)
                    output1y.append(outputY)
                NudeNet_regions = zip(output1x, output1y)

                #Save file
                f=f.replace(rootdir, outdir, 1)
                os.makedirs(os.path.dirname(f), exist_ok=True)
                cv2.imwrite('temp_out.png', image)     #still a hack for non-unicode names
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