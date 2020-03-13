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
from NudeNet_edited import Detector
detector = Detector()

#You can change those folder paths
rootdir = "./decensor_input"
outdir = "./decensor_input_original"
os.makedirs(rootdir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)

rgbvals = 255, 255, 255

files = glob.glob(rootdir + '/**/*.png', recursive=False)
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
	#points = np.array(points)

	## Call function here to write annotations to file_name using the 4 points above as a polygon
	#write_annotation(points, annotation_file)

	## Random color function - Want multiple shades of dark-grey to black, and white to super light grey
	#random_color = rand_color()
	cv2.fillConvexPoly(img, points, 0, 255, 0)
	return(points)

#Working with files
with open('example.csv', 'w', newline='', encoding='utf-8') as f_output:     #CSV
	csv_output = csv.writer(f_output, quoting=csv.QUOTE_NONE, quotechar="", delimiter=",", escapechar=' ')     #CSV
	csv_output.writerow(['filename','file_size','file_attributes','region_count','region_id','region_shape_attributes','region_attributes'])	 #CSV
	for f in files:
		try:
			while True:
				print("Working on " + f)
				img_C = Image.open(f).convert("RGB")
				x, y = img_C.size
				card = np.array(Image.new("RGB", (x, y), (rgbvals)))
				img_C = np.array(img_C) 
				img_rgb = img_C[:, :, ::-1].copy() 
				#####
				all_regions = detector.detect(f)
				points = []
	#			outputXY = []
				for region in all_regions:
					min_x, min_y, max_x, max_y = region 
					## Generate data with this
				
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

					print(points)
					output1x = []
					output1y = []
					for idx,conturJ in enumerate(points):
						outputX = []
						outputY = []
						for idx2,temp in enumerate(conturJ):
						#print(points[0])
						#print(points[1])
							xt, yt = temp
							outputX.append(xt)
							outputY.append(yt)
						#print(temp)
						output1x.append(outputX)
						output1y.append(outputY)
						

					outputXY = zip(output1x, output1y)
			#		outputXY_list = list(outputXY)
			#		output2Y = [output1y]
					#print(output2y)
						

						#outputX.append(temp[idx-1][0])
						#outputY.append(temp[idx-1][1])
							#outputX.append(temp[0])
							#outputY.append(temp[0])
							#for idx,temp in enumerate(points):
								#outputX.append(temp[0][0])
								#outputY.append(temp[0][1])
						#print(outputX, outputY)
						
					#####
				
				#Save file
				f=f.replace(rootdir, outdir, 1)
				os.makedirs(os.path.dirname(f), exist_ok=True)
				cv2.imwrite('temp_out.png', img_rgb)	 #still a hack for non-unicode names
				os.replace('temp_out.png', f)
				

				#for idx,temp in enumerate(points):
					#outputX.append(temp[0][0])
					#outputY.append(temp[0][1])
				#outputX,outputY = outputXY
				for idx,tempX in enumerate(outputXY):
					for idx2, tempX in enumerate(output1x):
						print(output1x[idx2], output1y[idx2])
					csv_output.writerow([os.path.basename(f), os.stat(f).st_size, '"{}"', len(output1x), idx, '"{""name"":""polygon""','""all_points_x"":' + str(output1x[idx]), '""all_points_y"":' + str(output1y[idx]) + '}"', '"{}"'])     #CSV
			#		print(tempX)
			#		for idx, tempY in enumerate(output1y):
						
					#outputX,outputY = temp[0]
			#			print(tempY[0])
		#			print(tempY[0])
		#			for idx2, temp2 in enumerate(temp):
						#outputX,outputY = temp2
		#				print(temp2[idx])
					
#				for idx,(x1,y1,x2,y2) in enumerate(points):
						#cv2.rectangle(img_rgb,(x1,y1),(x2,y2),(0,255,0),1)
						#rectNum=list(zip(*loc[::-1]))	 #CSV
						#cv2.rectangle(img_rgb, outputBoxes, (0,255,0), 1)
#					csv_output.writerow([os.path.basename(f), os.stat(f).st_size, '"{}"', len(region), idx, '"{""name"":""rect""','""x"":' + str(x1), '""y"":' + str(y1), '""width"":' + str(x2-x1), '""height"":' + str(y2-y1) + '}"', '"{}"'])	 #CSV
#					csv_output.writerow([os.path.basename(f), os.stat(f).st_size, '"{}"', len(region), idx, '"{""name"":""polygon""','""all_points_x"":' + str(outputX), '""all_points_y"":' + str(outputY) + '}"', '"{}"'])     #CSV
				#print(outputX)

				#Change path to save folder
				#f=f.replace(rootdir, outdir, 1)


				break
		except Exception as Exception:
			err_files.append(os.path.basename(f) + ": " + str(Exception))
			pass

#Error list	
if err_files:
	print("\n" + "Could not mask those files: ") 
	for f in err_files:
		print(f)