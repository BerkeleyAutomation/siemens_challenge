import cv2
import numpy as np
import json
from os import listdir, path
import re
import xml.etree.ElementTree as et
from xml.dom import minidom

import IPython

import shapely.geometry as geometry
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay

import math

SEG_LABELS = {"screwdriver":0, "tape":1, "tube":2, "scrap":3, "hammer":4, "wrench":5, "background":6}
# def find_red_markers(image):
# 	b = image[:,:,0]
# 	g = image[:,:,1]
# 	r = image[:,:,2]

# 	image_copy = np.copy(image)
# 	# b[bkgr_copy[:,:,0] > 10] = 0
# 	r[image_copy[:,:,0] > 10] = 0
# 	# g[bkgr_copy[:,:,0] > 10] = 0
# 	# g[bkgr_copy[:,:,1] > 10] = 0
# 	# b[bkgr_copy[:,:,1] > 10] = 0
# 	r[image_copy[:,:,1] > 10] = 0
# 	# b[bkgr_copy[:,:,2] < 245] = 0
# 	# g[bkgr_copy[:,:,2] < 245] = 0
# 	r[image_copy[:,:,2] < 245] = 0

# 	ret, mask = cv2.threshold(r, 128, 255, 0)
# 	contours, hierarchy = cv2.findContours(mask, 1, 1)
# 	areas = [cv2.contourArea(cnt) for cnt in contours]
# 	# print(areas)
# 	moments = [[cv2.moments(cnt), cnt] for cnt in contours if cv2.contourArea(cnt) > 0]
# 	cv2.imwrite("debug.png", mask)
# 	centroids = [(int(M[0]['m10']/M[0]['m00']), int(M[0]['m01']/M[0]['m00'])) for M in moments]
# 	print(len(centroids))

# 	assert len(centroids) == 16
# 	return centroids

# def find_homography(centroids1, centroids2):
# 	h, status = cv2.findHomography(np.array(centroids1), np.array(centroids2))
# 	return h




def alpha_shape(points, alpha):
	"""
	Compute the alpha shape (concave hull) of a set
	of points.
	@param points: Iterable container of points.
	@param alpha: alpha value to influence the
		gooeyness of the border. Smaller numbers
		don't fall inward as much as larger numbers.
		Too large, and you lose everything!
	"""
	if len(points) < 4:
		# When you have a triangle, there is no sense
		# in computing an alpha shape.
		return geometry.MultiPoint(list(points)).convex_hull
	def add_edge(edges, edge_points, coords, i, j):
		"""
		Add a line between the i-th and j-th points,
		if not in the list already
		"""
		if (i, j) in edges or (j, i) in edges:
			# already added
			return
		edges.add( (i, j) )
		edge_points.append(coords[ [i, j] ])
	coords = np.array([point[0] for point in points])
	# print(coords)
	tri = Delaunay(coords)
	edges = set()
	edge_points = []
	# loop over triangles:
	# ia, ib, ic = indices of corner points of the
	# triangle
	for ia, ib, ic in tri.vertices:
		pa = coords[ia]
		pb = coords[ib]
		pc = coords[ic]
		# Lengths of sides of triangle
		a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
		b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
		c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
		# Semiperimeter of triangle
		s = (a + b + c)/2.0
		# Area of triangle by Heron's formula
		area = math.sqrt(s*(s-a)*(s-b)*(s-c))
		circum_r = a*b*c/(4.0*area)
		# Here's the radius filter.
		#print circum_r
		if circum_r < 1.0/alpha:
			add_edge(edges, edge_points, coords, ia, ib)
			add_edge(edges, edge_points, coords, ib, ic)
			add_edge(edges, edge_points, coords, ic, ia)
	m = geometry.MultiLineString(edge_points)
	triangles = list(polygonize(m))
	return unary_union(triangles), edge_points

def read_img(img_path):
	return cv2.imread(img_path, 0)

def extract_mask(gray):
	ret, mask = cv2.threshold(gray, 2, 255, 0)
	kernel = np.ones((3, 3), np.uint8)
	# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	contours, hierarchy = cv2.findContours(mask, 1, 1)
	areas = [cv2.contourArea(cnt) for cnt in contours]
	# print(areas)
	moments = [[cv2.moments(cnt), cnt] for cnt in contours if cv2.contourArea(cnt) > 0]
	centroids = [(int(M[0]['m10']/M[0]['m00']), int(M[0]['m01']/M[0]['m00'])) for M in moments]

	for i in range(len(moments)):
		if centroids[i][0] < 10 or mask.shape[1] - centroids[i][0] < 10 or centroids[i][1] < 10 or mask.shape[0] - centroids[i][1] < 10 or cv2.contourArea(moments[i][1]) < 200:
			cv2.drawContours(mask, [moments[i][1]], 0, (0, 0, 0), -1)
		else:
			cv2.drawContours(mask, [moments[i][1]], 0, (255, 255, 255), -1)

	cv2.imwrite("test.png", mask)
	return mask

def find_curr_img(prev_img, curr_img):
	# rows, cols, channels = curr_img.shape
	# shift_x = 0
	# shift_y = 0
	# shift_r = 0
	# min_diff = np.sum(cv2.subtract(curr_img, prev_img))
	# for i in range(10):
	# 	for j in range(10):
	# 		for r in range(6):
	# 			if r > 2:
	# 				M = cv2.getRotationMatrix2D((cols/2,rows/2),(r - 3) * 10,1)
	# 			else:
	# 				M = cv2.getRotationMatrix2D((cols/2,rows/2),(3 - r) * 10,0)
	# 			shift_prev = cv2.warpAffine(prev_img,M,(cols,rows))
	# 			m = np.float32([[1, 0, i-5], [0, 1, j-5]])
	# 			shift_prev = cv2.warpAffine(shift_prev, m, (cols, rows))
	# 			diff = cv2.subtract(curr_img, shift_prev)
	# 			if np.sum(diff) < min_diff:
	# 				shift_x = i - 5
	# 				shift_y = j - 5
	# 				shift_r = (r - 3) * 10
	# 				min_diff = np.sum(diff)
	# print(shift_x, shift_y, shift_r)
	# kernel = np.ones((3, 3), np.uint8)
	# if shift_r > 0: 
	# 	M = cv2.getRotationMatrix2D((cols/2,rows/2),shift_r,1)
	# else:
	# 	M = cv2.getRotationMatrix2D((cols/2,rows/2),abs(shift_r),0)
	# shift_prev = cv2.warpAffine(prev_img,M,(cols,rows))
	# m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
	# shift_prev = cv2.warpAffine(prev_img, m, (cols, rows))
	diff = cv2.absdiff(curr_img, prev_img)
	cv2.imwrite("debug_diff.png", diff)

	# return diff, trans_prev_img
	return diff, prev_img

def reduce_noice(mask):

	kernel = np.ones((3, 3), np.uint8)
	# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	img, contours, hierarchy = cv2.findContours(mask, 1, 1)
	if len(contours) == 0:
		return mask

	areas = [cv2.contourArea(cnt) for cnt in contours]
	max_area = max(areas)

	for i in range(len(contours)):
		if areas[i] < 20:
			cv2.drawContours(mask, [contours[i]], 0, (0, 0, 0), -1)
		else:
			cv2.drawContours(mask, [contours[i]], 0, (255, 255, 255), -1)

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	img, contours, hierarchy = cv2.findContours(mask, 1, 1)

	if len(contours) == 0:
		return mask

	areas = [cv2.contourArea(cnt) for cnt in contours]
	max_area = max(areas)

	for i in range(len(contours)):
		if areas[i] < 40:
			cv2.drawContours(mask, [contours[i]], 0, (0, 0, 0), -1)
		else:
			cv2.drawContours(mask, [contours[i]], 0, (255, 255, 255), -1)

	mask = cv2.dilate(mask, kernel)

	if len(contours) > 1:
		# kernel = np.ones((20, 20), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		pts = []

		for cnt in contours:
			pts.extend(cnt)
			M = cv2.moments(cnt)
			# print(M)
			if M['m00'] == 0.0:
				continue
			pts.append(np.array([[int(M['m10']/M['m00']), int(M['m01']/M['m00'])]]))


		concave_hull, edge_points = alpha_shape(pts, 0.1)
		coords = geometry.mapping(concave_hull)["coordinates"]
		for coord_list in coords:
			coord_list = np.array(coord_list, dtype=np.int32) 
			cv2.drawContours(mask, [coord_list], 0, (255, 255, 255), -1)

	img, contours, hierarchy = cv2.findContours(mask, 1, 1)

	# cv2.drawContours(mask, [concave_hull], 0, (255, 255, 255), -1)
	if len(contours) == 0:
		return mask

	areas = [cv2.contourArea(cnt) for cnt in contours]
	max_area = max(areas)

	for i in range(len(contours)):
		if areas[i] < max_area - 0.001:
			cv2.drawContours(mask, [contours[i]], 0, (0, 0, 0), -1)
		else:
			cv2.drawContours(mask, [contours[i]], 0, (255, 255, 255), -1)

	# cv2.imwrite("test.png", mask)
	return mask

def find_contour_and_bounding_box(mask):
	contours, hierarchy = cv2.findContours(mask, 1, 1)
	if len(contours) != 1:
		return None, None
	x, y, w, h = cv2.boundingRect(contours[0])
	contours = [pt[0].tolist() for pt in contours[0]]

	return contours, [x, y, x + w, y + h]


def find_item_masks(folder_path):
	masks = []
	with open(folder_path+"/"+"labels.json") as f:
		lst = json.load(f)
	# print(lst)
	item_num = len(lst)

	for i in range(item_num):
		curr_img = cv2.imread(folder_path+"/rgb_"+str(i)+".png")
		if i == 0:
			prev_img = cv2.imread(folder_path+"/rgb_background.png")
		else:
			prev_img = cv2.imread(folder_path+"/rgb_"+str(i-1)+".png")
		
		diff, trans_prev_img = find_curr_img(prev_img, curr_img)
		cv2.imwrite(folder_path+"/debug_diff_"+str(i-1)+".png", diff)

		cv2.imwrite(folder_path+"/debug_"+str(i-1)+".png", trans_prev_img)
		
		ret, diff = cv2.threshold(diff, 20, 255, 0)

		cv2.imwrite(folder_path+"/debug_diff_filted_"+str(i-1)+".png", diff)
		
		gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
		
		ret, mask = cv2.threshold(gray, 2, 255, 0)
		cv2.imwrite(folder_path+"/debug_mask_"+str(i-1)+".png", mask)
		#TODO: turn RGB to binary image
		masks.append(reduce_noice(mask))
	# print(len(masks))

	for i in range(item_num):
		cv2.imwrite(folder_path+"/"+lst[i]+".png", masks[i])

def draw_masks(folder_path):
	with open(folder_path+"/"+"labels.json") as f:
		lst = json.load(f)
	item_num = len(lst)
	# print(folder_path+"/rgb_"+str(item_num - 1)+".png")
	img = cv2.imread(folder_path+"/rgb_"+str(item_num - 1)+".png", 1)
	img2 = cv2.imread(folder_path+"/rgb_"+str(item_num - 1)+".png", 1)
	# print(img)
	diff, trans_prev_img = find_curr_img(cv2.imread(folder_path+"/rgb_"+str(item_num - 1)+".png"), cv2.imread(folder_path+"/rgb_background.png"))
	cv2.imwrite("test_diff.png", diff)
	ret, diff = cv2.threshold(diff, 10, 255, 0)
	cv2.imwrite("test_diff2.png", diff)
	gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
	pure_img = extract_mask(gray)

	count = 1

	blank = np.zeros(img.shape, img.dtype)
	blank2 = np.zeros(pure_img.shape, pure_img.dtype)
	blank3 = np.zeros(pure_img.shape, pure_img.dtype)
	for filename in listdir(folder_path):
		if filename.split(".")[1] == "png" and filename.split("_")[0] != 'rgb' and filename.split("_")[0] != 'depth' and filename.split("_")[0] != 'debug':

			class_label = re.split('(\d+)', filename)[0]

			mask = cv2.imread(folder_path+"/"+filename, 0)
			seg, bb = find_contour_and_bounding_box(mask)
			if not bb is None:
				test_bb = cv2.rectangle(img2,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),2)

			color = np.zeros(img.shape, img.dtype)
			color[:,:] = (count % 3 * 127,(count // 3) % 3 * 127, (count // 9) % 3 * 127)
			colorMask = cv2.bitwise_and(color, color, mask=mask)

			pure = np.zeros(pure_img.shape, pure_img.dtype)
			# pure[:,:] = 6 - SEG_LABELS[class_label]
			pure[:,:] = 255
			pureMask = cv2.bitwise_and(pure, pure, mask=mask)
			cv2.addWeighted(pureMask, 1, blank2, 1, 0, blank2)

			seg = np.zeros(pure_img.shape, pure_img.dtype)
			seg[:,:] = 6 - SEG_LABELS[class_label]
			segMask = cv2.bitwise_and(seg, seg, mask=mask)
			cv2.addWeighted(segMask, 1, blank3, 1, 0, blank3)

			cv2.addWeighted(colorMask, 1, blank, 1, 0, blank)
			# cv2.imwrite(folder_path+"/masked_"+filename, colorMask)
			count += 1
	cv2.imwrite(folder_path+"/masked_imgs.png", blank2)
	cv2.addWeighted(blank, 1, img, 0.5, 0, img)
	cv2.imwrite(folder_path+"/compare_imgs2.png", img)
	abs_diff = cv2.absdiff(blank2, pure_img)
	print(abs_diff.sum() / 255)
	# return img, abs_diff.sum() / 255, cv2.subtract(6, blank2)
	return img, abs_diff.sum() / 255, cv2.subtract(6, blank3), img2


def create_segment_label(folder_base, label_index):
	folder_path = folder_base + str(label_index)
	with open(folder_path+"/"+"labels.json") as f:
		lst = json.load(f)	

	item_num = len(lst)

	# set up segmentation json file
	seg_label = {
		"shapes":[],
		"lineColor": [
			0,
			255,
			0,
			128
		],
		"fillColor": [
			255,
			0,
			0,
			128
		], 
		"imagePath": "rgb_"+str(item_num-1)+".png"
	}

	# set up bounding box xml
	bbox_root = et.Element("annotate")

	et.SubElement(bbox_root, "folder").text = folder_path

	et.SubElement(bbox_root, "filename").text = path.abspath("rgb_"+str(item_num-1)+".png")

	source = et.SubElement(bbox_root, "source")
	et.SubElement(source, "database").text = "Unknown"

	size = et.SubElement(bbox_root, "size")
	et.SubElement(size, "width").text = "640"
	et.SubElement(size, "height").text = "480"
	et.SubElement(size, "depth").text = "3"

	et.SubElement(bbox_root, "segmented").text = "0"

	for filename in listdir(folder_path):
		if filename.split(".")[1] == "png" and filename.split("_")[0] != 'rgb' and filename.split("_")[0] != 'depth' and filename.split("_")[0] != 'masked' and filename.split("_")[0] != 'compare' and filename.split("_")[0] != 'debug':
			mask = cv2.imread(folder_path+"/"+filename, 0)
			seg, bb = find_contour_and_bounding_box(mask)

			# test_bb = cv2.rectangle(cv2.imread(folder_path+"/"+filename, 1),(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),2)
			# cv2.imwrite(folder_path+"/bb_"+filename, test_bb)

			class_label = re.split('(\d+)', filename)[0]

			if seg is None:
				continue

			# json add item
			single_item = {"label": class_label, "line_color": None, "fill_color": None, "points": seg}
			seg_label["shapes"].append(single_item)

			# xml add item
			obj_info = et.SubElement(bbox_root, "object")

			et.SubElement(obj_info, "name").text = class_label
			et.SubElement(obj_info, "pose").text = "Unspecified"
			et.SubElement(obj_info, "truncated").text = "0"
			et.SubElement(obj_info, "difficult").text = "0"

			bbox = et.SubElement(obj_info, "bndbox")
			et.SubElement(bbox, "xmin").text = str(bb[0])
			et.SubElement(bbox, "ymin").text = str(bb[1])
			et.SubElement(bbox, "xmax").text = str(bb[2])
			et.SubElement(bbox, "ymax").text = str(bb[3])


	# write to json
	with open(folder_path+"/"+"rgb_"+str(item_num-1)+".json", "w") as out:
		json.dump(seg_label, out, indent=4)

	# write to json
	with open(folder_base+"seg_label_"+str(label_index)+".json", "w") as out:
		json.dump(seg_label, out, indent=4)

	# write to xml tree
	# tree = et.ElementTree(bbox_root)
	# tree.write(folder_path+"/"+"rgb_"+str(item_num-1)+".xml")
	xmlstr = minidom.parseString(et.tostring(bbox_root)).toprettyxml(indent="   ")
	with open(folder_path+"/"+"rgb_"+str(item_num-1)+".xml", "w") as f:
		f.write(xmlstr)

	with open(folder_base+"box_label_"+str(label_index)+".xml", "w") as f:
		f.write(xmlstr)
	

	# tree = et.ElementTree(bbox_root)
	# tree.write(folder_base+"box_label_"+str(label_index)+".xml")

