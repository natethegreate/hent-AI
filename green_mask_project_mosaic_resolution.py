import cv2
import numpy as np
import os
import glob
from scipy.signal import argrelextrema
from PIL import Image

# Conversion from file based script to individual image usage
def get_mosaic_res(root_img=None):
    # assert root_img
    #You can change those folder paths
    # os.makedirs(root_img, exist_ok=True)

    # files = glob.glob(rootdir + '/**/*.png', recursive=True)
    # files_jpg = glob.glob(rootdir + '/**/*.jpg', recursive=True)
    # files.extend(files_jpg)
    f = root_img # use input image path
    #-----------------------Logic-----------------------
    GBlur = 5
    CannyTr1 = 20
    CannyTr2 = 100
    LowRange = 2
    HighRange = 24
    DetectionTr = 0.32

    pattern = [None] * (HighRange+2)
    for masksize in range(HighRange+2, LowRange+1, -1):
        maskimg = 2+masksize+masksize-1+2
        screen = (maskimg, maskimg)
        img = Image.new('RGB', screen, (255,255,255))
        pix = img.load()
        for i in range(2,maskimg,masksize-1):
            for j in range(2,maskimg,masksize-1):
                for k in range(0,maskimg):
                    pix[i, k] = (0,0,0)
                    pix[k, j] = (0,0,0)
        pattern[masksize-2] = img

    #Working with files
    # for f in files:
    #-----------------------Files-----------------------
    img_C = Image.fromarray(f).convert("RGBA")
    x, y = img_C.size
    card = Image.new("RGBA", (x, y), (255, 255, 255, 0))
    cvI = Image.alpha_composite(card, img_C)
    cvI = np.array(cvI)
    img_rgb = cv2.cvtColor(cvI, cv2.COLOR_BGRA2RGBA)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.Canny(img_gray,CannyTr1,CannyTr2)
    img_gray = 255-img_gray
    img_gray = cv2.GaussianBlur(img_gray,(GBlur,GBlur),0)
    
    #-----------------------Detection-----------------------
    resolutions = [-1] * (HighRange+2)
    for masksize in range(HighRange+2, LowRange+1, -1):
        template = cv2.cvtColor(np.array(pattern[masksize-2]), cv2.COLOR_BGR2GRAY)
        w, h = pattern[masksize-2].size[::-1]
    
        img_detection = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where(img_detection >= DetectionTr)
        rects = 0
        for pt in zip(*loc[::-1]):
            rects += 1    #increase rectangle count of single resolution
#            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)     #DEBUG To see regions on image
        resolutions[masksize-1] = rects

    resolutions.append(0)
#    print(resolutions)    #DEBUG Resolutions array
    extremaMIN = argrelextrema(np.array(resolutions), np.less, axis=0)[0]
    extremaMIN = np.insert(extremaMIN,0,LowRange)
    extremaMIN = np.append(extremaMIN,HighRange+2)

    Extremas = []
    for i, ExtGroup in enumerate(extremaMIN[:-1]):
        Extremas.append((ExtGroup, resolutions[extremaMIN[i]:extremaMIN[i+1]+1]))

    ExtremasSum = []
    BigExtrema = [0,0,[0,0]]
    for i, _ in enumerate(Extremas):
        ExtremasSum.append(sum(Extremas[i][1]))
        if BigExtrema[0] <= sum(Extremas[i][1])+int(sum(Extremas[i][1])*0.05):    #5% precedency for smaller resolution
            BigExtrema = [sum(Extremas[i][1]),Extremas[i][0],Extremas[i][1]]    
    MosaicResolutionOfImage = BigExtrema[1]+BigExtrema[2].index(max(BigExtrema[2]))    #Output
    if MosaicResolutionOfImage == 0:    #If nothing found - set resolution as smallest
        MosaicResolutionOfImage = HighRange+1
    # print('Mosaic Resolution of "' + os.path.basename(f) + '" is: ' + str(MosaicResolutionOfImage))    #The Resolution of Mosaiced Image
    return MosaicResolutionOfImage
    
    #DEBUG Show image
#    cv2.imshow('image',img_rgb)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
