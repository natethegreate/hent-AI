import cv2
import math
import csv
import numpy as np
import random
import os
import glob
from PIL import Image
from nudenet import NudeDetector	#from NudeNet_edited import Detector
detector = NudeDetector()	#detector = Detector()

#You can change those folder paths
rootdir = "./decensor_input"
outdir_mosaics = "./decensor_input_mosaics"
outdir_bars = "./decensor_input_bars"
os.makedirs(rootdir, exist_ok=True)
os.makedirs(outdir_mosaics, exist_ok=True)
os.makedirs(outdir_bars, exist_ok=True)

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
		else:			   # case where we vary only 1 of the 3 values
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

def pixelate(image, ratio, mosaic_kernel, interp):
	# Get input size
	height, width, _ = image.shape
	# Desired "pixelated" size
	h, w = (mosaic_kernel, int(mosaic_kernel*ratio))
	# Resize image to "pixelated" size
	temp = cv2.resize(image, (w, h), interpolation=interp)    #cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_NEAREST, cv2.INTER_LINEAR
	# Initialize output image
	return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

''' draw angled rectangle function
 x0,y0: center point of rectangle
 height, width, angle, color: rectangle properties
 img, img_x, img_y: source image and its dimensions
 returns: will return np array of points, or None type
'''
def draw_angled_rec(x0, y0, width, height, angle, img, color, img_x, img_y, mode, q):
	points = []
	points2 = []
	quantity = q
	while quantity > 0:
		if '_wing' in mode:
			if mode == 'horizontal_wing':
				mu = abs((angle-90)/90)
			else:
				mu = abs((angle)/90)
			sigma = 0.5 - mu
			angl_mod = 10*random.gauss(mu, sigma)
			#print(angl_mod)
			if quantity == 2:
				height = height*0.5*(abs(math.sin(angle))+abs(math.cos(angle)))
				#print(height)
				card = np.array(Image.new('RGB', (img_x, img_y), (0, 0, 0)))
				_angle = (angle+angl_mod-2*(angle-90)) * math.pi / 180.0
			else:
				_angle = (angle+angl_mod) * math.pi / 180.0
		else:
			_angle = angle * math.pi / 180.0
		b = math.cos(_angle) * 0.5
		a = math.sin(_angle) * 0.5
		#print(str(b) + ", " + str(a) + " - cos, sin. Angle - " + str(_angle))	#DEBUG

		# draw points with slightly smaller dimenstions, width and height difference due to different scaling
		height_s = height - 3
		width_s = width - 3
		# also decrease the scale of b in the x calculation
		bl = [int(x0 - a * height_s - b * width_s), int(y0 + b * height_s - a * width_s)]
		ul = [int(x0 + a * height_s - b * width_s), int(y0 - b * height_s - a * width_s)]
		ur = [int(2 * x0 - bl[0]), int(2 * y0 - bl[1])]
		br = [int(2 * x0 - ul[0]), int(2 * y0 - ul[1])]

		# original size
		bls = [int(x0 - a * height - b * width), int(y0 + b * height - a * width)]
		uls = [int(x0 + a * height - b * width), int(y0 - b * height - a * width)]
		urs = [int(2 * x0 - bls[0]), int(2 * y0 - bls[1])]
		brs = [int(2 * x0 - uls[0]), int(2 * y0 - uls[1])]
		
		angl_devider = random.triangular(0.35, 0.5) # ==/3.0~/2.0

		if (mode, quantity) == ('horizontal_wing', 1):
			bl = (int(bl[0] - height*math.sin(_angle)*angl_devider), bl[1])
			ul = (int(ul[0] - height*math.sin(_angle)*angl_devider), ul[1])
			br = (int(br[0] - height*math.sin(_angle)*angl_devider), br[1])
			ur = (int(ur[0] - height*math.sin(_angle)*angl_devider), ur[1])
			
			bls = (int(bls[0] - height*math.sin(_angle)*angl_devider), bls[1])
			uls = (int(uls[0] - height*math.sin(_angle)*angl_devider), uls[1])
			brs = (int(brs[0] - height*math.sin(_angle)*angl_devider), brs[1])
			urs = (int(urs[0] - height*math.sin(_angle)*angl_devider), urs[1])
		elif (mode, quantity) == ('vertical_wing', 1):
			bl = (bl[0], int(bl[1] - height*math.cos(_angle)*angl_devider))
			ul = (ul[0], int(ul[1] - height*math.cos(_angle)*angl_devider))
			br = (br[0], int(br[1] - height*math.cos(_angle)*angl_devider))
			ur = (ur[0], int(ur[1] - height*math.cos(_angle)*angl_devider))
			
			bls = (bls[0], int(bls[1] - height*math.cos(_angle)*angl_devider))
			uls = (uls[0], int(uls[1] - height*math.cos(_angle)*angl_devider))
			brs = (brs[0], int(brs[1] - height*math.cos(_angle)*angl_devider))
			urs = (urs[0], int(urs[1] - height*math.cos(_angle)*angl_devider))
		elif (mode, quantity) == ('horizontal_wing', 2):
			bl = (int(bl[0] + height*math.sin(_angle)*angl_devider), bl[1])
			ul = (int(ul[0] + height*math.sin(_angle)*angl_devider), ul[1])
			br = (int(br[0] + height*math.sin(_angle)*angl_devider), br[1])
			ur = (int(ur[0] + height*math.sin(_angle)*angl_devider), ur[1])
			
			bls = (int(bls[0] + height*math.sin(_angle)*angl_devider), bls[1])
			uls = (int(uls[0] + height*math.sin(_angle)*angl_devider), uls[1])
			brs = (int(brs[0] + height*math.sin(_angle)*angl_devider), brs[1])
			urs = (int(urs[0] + height*math.sin(_angle)*angl_devider), urs[1])
		elif (mode, quantity) == ('vertical_wing', 2):
			bl = (bl[0], int(bl[1] - height*math.cos(_angle)*angl_devider))
			ul = (ul[0], int(ul[1] - height*math.cos(_angle)*angl_devider))
			br = (br[0], int(br[1] - height*math.cos(_angle)*angl_devider))
			ur = (ur[0], int(ur[1] - height*math.cos(_angle)*angl_devider))
			
			bls = (bls[0], int(bls[1] - height*math.cos(_angle)*angl_devider))
			uls = (uls[0], int(uls[1] - height*math.cos(_angle)*angl_devider))
			brs = (brs[0], int(brs[1] - height*math.cos(_angle)*angl_devider))
			urs = (urs[0], int(urs[1] - height*math.cos(_angle)*angl_devider))
		
		points = np.array((bl, ul, ur, br))
		points2 = np.array((bls, uls, urs, brs))
		
		# verify rectangle is within borders
		for pnt in points2:
			if pnt[0] < 0 or pnt[0] > img_x:
				return []
			if pnt[1] < 0 or pnt[1] > img_y:
				return []
	## Random color function - Want multiple shades of dark-grey to black, and white to super light grey
		r, g, b = color
		cv2.fillConvexPoly(img, points, color=(r, g, b), lineType=cv2.LINE_AA) 
		if q == 2:
			cv2.fillConvexPoly(card, points2, color=(255, 255, 255))
		quantity -= 1
		
	if q == 2:
		card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
		conturs, _ = cv2.findContours(card,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	#cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_TC89_KCOS
		#print(conturs[0])
		return(conturs[0])
		
		

		
		
		
	#img[y0,x0]=0,0,255
	#cv2.imwrite('temp_out.png', img)
	# send original points
	#print(points)
	#print(points2)
	return(points2)

#Working with files
with open('example.csv', 'w', newline='', encoding='utf-8') as f_output:	 #CSV
	csv_output = csv.writer(f_output, quoting=csv.QUOTE_NONE, quotechar="", delimiter=",", escapechar=' ')	 #CSV
	csv_output.writerow(['filename','file_size','file_attributes','region_count','region_id','region_shape_attributes','region_attributes'])	 #CSV
	for f in files:
		try:
			while True:
				print("Working on " + f)
				img_C = Image.open(f).convert("RGB")
				x, y = img_C.size
				img_C = np.array(img_C) 
				image = img_C[:, :, ::-1].copy()
				img_rgb = img_C[:, :, ::-1].copy()

				color = rand_color()
				
				detection = detector.detect(f)
				label=['F_GENITALIA', 'M_GENITALIA']#
				all_regions = [i['box'] for i in detection if i['label'] in label]#

				if all_regions == []:
					# skip entire detection, avoid saving 
					#os.remove(f)    #to remove file from input
					print('skipping image with failed nudenet detection')
					break
				print(all_regions)#
				interp = random.choices([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_AREA], cum_weights=[1, 1, 1, 7])[0]    #randomize the interpolation
				#print(interp)
				#interp = cv2.INTER_NEAREST    #cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_NEAREST, cv2.INTER_LINEAR
				#mosaic_kernel = int(random.triangular(8, 50, 32))    #mosaic resolution
				mosaic_kernel = int(random.triangular(int(min(x*0.01, y*0.01)), int(min(x*0.2, y*0.2)), int(min(x*0.0625, y*0.0625))))    #mosaic resolution 0.5%~33% with 
				#print(int(min(x, y)/mosaic_kernel))
				if random.random() <= 0.75:    #probability for ajasting to ratio
					calculate = True
					#print('calculate')
				else:
					calculate = False
					ratio = 1
				if calculate:
					ratio = x/y
				
				pixelated_ROI = pixelate(image, ratio, mosaic_kernel, interp)
				
				points = []
				for region in all_regions:
					min_x, min_y, max_x, max_y = region
					center = (int((max_x+min_x)*0.5), int((max_y+min_y)*0.5))
					#print(center)
					len_x = max_x-min_x
					len_y = max_y-min_y
					thickness = random.triangular(len_x*0.4, len_x, len_x*0.9)
					wideness = random.triangular(len_y*0.4, len_y, len_y*0.9)
					min_x = int(center[0] - thickness*0.5)+2
					min_y = int(center[1] - wideness*0.5)+2
					max_x = int(center[0] + thickness*0.5)-2
					max_y = int(center[1] + wideness*0.5)-2
					image[min_y:max_y, min_x:max_x] = pixelated_ROI[min_y:max_y, min_x:max_x]
					points.append(np.array(((min_x-2, min_y-2), (min_x-2, max_y+2), (max_x+2, max_y+2), (max_x+2, min_y-2))))
					
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
				f=f.replace(rootdir, outdir_mosaics, 1)
				os.makedirs(os.path.dirname(f), exist_ok=True)
				cv2.imwrite('temp_out.png', image)	 #still a hack for non-unicode names
				os.replace('temp_out.png', f)

				for idx,_ in enumerate(NudeNet_regions):
					csv_output.writerow([os.path.basename(f), os.stat(f).st_size, '"{}"', len(output1x), idx, '"{""name"":""polygon""','""all_points_x"":' + str(output1x[idx]), '""all_points_y"":' + str(output1y[idx]) + '}"', '"{""censor"":""bar""}"'])	 #CSV
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
					area = len_x*len_y	#area of nudenet zone
					score = random.triangular(area*0.15, area*0.3)	#maximal area for rectangles
					i=0
					while score >= area*0.03:
						if len_x >= len_y:	#decide the longest side
							# print("vertical bar")
							mode = 'vertical'
							thickness = random.triangular(len_x*0.03, len_x*0.15)	#thickness of the bar
							wideness = random.triangular(len_y*0.3, len_y*0.75)	#wideness of the bar
							angle = 0	#axis
							bar_x = int(random.uniform(min_x, max_x))	#random bar_x
							bar_y = int(random.triangular(min_y, max_y))#, min_y+(max_y-min_y)/2-wideness/2))	#random bar_y
							#print(bar_x, bar_y)
							comp_area = list(range(bar_x, bar_x+int(len_x*0.1),1))
						else:
							# print("horizontal bar")
							mode = 'horizontal'
							thickness = random.triangular(len_y*0.03, len_y*0.15)	#thickness of the bar
							wideness = random.triangular(len_x*0.3, len_x*0.75)	#wideness of the bar
							angle = 90	#axis
							bar_x = int(random.triangular(min_x, max_x))#, min_x+(max_x-min_x)/2-wideness/2))	#random bar_x
							bar_y = int(random.uniform(min_y, max_y))	#random bar_y
							#print(bar_x, bar_y)
							comp_area = list(range(bar_y, bar_y+int(len_y*0.1),1))
						if thickness*wideness <= score + area*0.02:
							rotate = random.randint(angle-15, angle+15)	#random angle within 15% from axis
							#print(rotate)
							if rotate < 0:
								rotate += 360
							if not any(check in comp_area for check in comp_array):
								comp_array = comp_array + comp_area
								quantity = 1
								if random.random() >= 0.8:	#20% probability of wings
									mode += '_wing'
									quantity = 2
									rotate = random.randint(angle-45, angle+45)	#angle between wings
									if rotate < 0:
										rotate += 360
								rect_points = draw_angled_rec(bar_x, bar_y, thickness, wideness, rotate, img_rgb, color, x, y, mode, quantity)
								if len(rect_points) != 0:
									points.append(rect_points)
								
								else:
									print("skipping out of bounds rect spawn")
									continue # in case of no rectangle drawn, simply go to next iteration
								score -= thickness*wideness	#subtract last rectangle from maximal area for rectangles
							else:	#recursion prevention
								i += 1
								if i == 30:
									print(str(score/area*100) + " of area left")
									break
					#print(points)

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
				f=f.replace(outdir_mosaics, outdir_bars, 1)
				os.makedirs(os.path.dirname(f), exist_ok=True)
				cv2.imwrite('temp_out.png', img_rgb)	 #still a hack for non-unicode names
				os.replace('temp_out.png', f)

				for idx,_ in enumerate(NudeNet_regions):
					csv_output.writerow([os.path.basename(f), os.stat(f).st_size, '"{}"', len(output1x), idx, '"{""name"":""polygon""','""all_points_x"":' + str(output1x[idx]), '""all_points_y"":' + str(output1y[idx]) + '}"', '"{""censor"":""mosaic""}"'])	 #CSV
				break

		except Exception as Exception:
			err_files.append(os.path.basename(f) + ": " + str(Exception))
			pass

#Error list	
if err_files:
	print("\n" + "NudeNet failed: ") 
	for f in err_files:
		print(f)