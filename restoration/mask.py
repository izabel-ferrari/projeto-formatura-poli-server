import numpy as np
import cv2

color_black = 0
color_white = 255
fill = -1

def get_ridge_mask(img):
	scale = .5
	inv = cv2.bitwise_not(img)
	ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create(scale = scale)
	ridges = ridge_filter.getRidgeFilteredImage(inv)
	max_value = np.amax(inv) - 130
	ret,thresh = cv2.threshold(ridges,max_value,255,cv2.THRESH_BINARY)
	mask = transform_contours(thresh, "fill", 50, img)[0]
	mask = thresh | mask

	return mask


def get_mask_by_type(img, type):
	# type is white (default) or black

	# Create HSV and grayscale formats
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Define variables used in masking
	value_range = 50
	saturation_range = 80
	if (type == "black"):
		min_value = np.amin(gray)
		max_value = min_value + value_range
	else:
		max_value = np.amax(gray)
		min_value = max_value - value_range
	min_color = np.array([color_black, color_black, min_value])
	max_color = np.array([color_white, saturation_range, max_value])

	# Create initial region mask
	reg = cv2.inRange(hsv, min_color, max_color)

	# Define morphologial transformation kernel
	kernel = np.ones((3,3),np.uint8)

	# Create edges and dilate to get better results
	edges = cv2.Canny(gray, 100, 150)
	edges = cv2.dilate(edges, kernel)
	# kernel = np.ones((5,5),np.uint8)
	edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	# Intersect region and edges
	mask = reg & edges

	return mask

def transform_contours(mask, operation, areaThreshold, img):
	blank = np.zeros(mask.shape, np.uint8)
	blank2 = np.copy(mask)
	blank3 = np.copy(blank)
	im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for i, contour in enumerate(contours):
		if(cv2.contourArea(contour) < areaThreshold):
			if(operation == "retangulate"):
				retangulate_contour(blank,contour)
			elif(operation == "fill"):
				fill_contours(blank, contours, i, hierarchy, img)
		else:
			fill_contours(blank2, contours, i, hierarchy, img)
			fill_contours(blank3, contours, i, hierarchy, img)
	return blank,blank2

def retangulate_contour(img, contour):
	leftmost = contour[contour[:,:,0].argmin()][0][0]
	rightmost = contour[contour[:,:,0].argmax()][0][0]
	topmost = contour[contour[:,:,1].argmin()][0][1]
	bottommost = contour[contour[:,:,1].argmax()][0][1]
	topleft = (leftmost, topmost)
	bottomright = (rightmost, bottommost)
	cv2.rectangle(img, topleft, bottomright, color_white, fill)

def fill_contours(img, contours, index, hierarchy, origImage):
	lineType = 1
	maxLevel = 1
	contourMask = np.zeros(img.shape,np.uint8)
	cv2.drawContours(contourMask,contours,index,255,-1)
	pixelpoints = np.transpose(np.nonzero(contourMask))
	region = origImage[contourMask==255]
	average = np.average(region)
	if(average > 200):
		cv2.drawContours(img, contours, index, color_white, fill, lineType, hierarchy, maxLevel)

def get_mask(img, damageType = "white"):
	mask = np.zeros(img.shape[:2], np.uint8)
	if(damageType == "white"):
		mask = get_mask_by_type(img, "white")
	if(damageType == "black"):
		mask = get_mask_by_type(img, "black")
	if(damageType == "both"):
		mask += get_mask_by_type(img, "white")
		mask += get_mask_by_type(img, "black")
	kernel = np.ones((3, 3),np.uint8)
	mask = cv2.dilate(mask, kernel)
	mask = mask | transform_contours(mask, "fill", max_image_area(mask), img)[0]
	return mask

def remove_eyes_from_mask(face_mask, true_eyes):
	angle = 0
	startAngle = 0
	endAngle = 360
	try:
		for (x, y, width, height) in true_eyes:
			size = (int(width / 2), int(height / 4))
			center = (int(x + 0.5 * width), int(y + 0.5 * height))
			cv2.ellipse(face_mask, center, size, angle, startAngle, endAngle, color_black, fill)
	except:
		pass

def remove_face_from_mask(mask, x, y, w, h):
	mask[y:y+h, x:x+w] = color_black

def remove_border_from_mask(mask):
	mask[0, 0:] = color_black
	mask[-1, 0:] = color_black
	mask[0:, 0] = color_black
	mask[0:, -1] = color_black

def get_rect_mask(img):
	mask = get_mask(img)
	mask, mixed = transform_contours(mask, "retangulate", max_image_area(mask), img)
	return mask, mixed

def max_image_area(img):
	return 0.003*img.shape[0]*img.shape[1]
