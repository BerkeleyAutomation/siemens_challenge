from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image, ImageTk
import ttk

import numpy as np
import cv2
import IPython
from skimage.draw import polygon

import tpc.config.config_tpc as cfg
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

from tpc.perception.cluster_registration import draw_point
import pickle



def set_crop(img_path="debug/imgs/new_setup_crop/crop_sample.png"):
    points = []
    # see https://stackoverflow.com/questions/5501192/how-to-display-picture-and-get-mouse-click-coordinate-on-it
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    init_dir = "/home/autolab/Workspaces/michael_working/tpc/src/tpc/debug_imgs/"
    File = askopenfilename(parent=root, initialdir=init_dir,title='Choose an image.')
    img = ImageTk.PhotoImage(Image.open(File))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        points.append([event.x,event.y])
        print([event.x,event.y])
    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()
    pickle.dump(points, open("src/tpc/config.crop.p", "wb"))

def crop_img(img, use_preset=False, arc=True, viz=False, simple=False, bycoords=None):
    """ Crops image polygonally
    to white working area (WA)
    Parameters
    ----------
    img : :obj:`numpy.ndarray`
    viz : boolean
        if true, displays the crop
        polygon on the original image
    Returns
    -------
    :obj:`BinaryImage`
        Image with only WA in white
    """
    img = np.copy(img)
    #threshold to WA and flip colors
    mask_shape = (img.shape[0], img.shape[1])

    if simple:
        scale_factors = [10, 10]
        to_chop = [mask_shape[0]/10, mask_shape[0]/5, mask_shape[1]/10, mask_shape[1]/5]
        #create mask
        focus_mask = np.zeros(mask_shape, dtype=np.uint8)
        focus_mask[to_chop[0]:-to_chop[1],to_chop[2]:-to_chop[3]] = 255
        return BinaryImage(focus_mask.astype("uint8"))
    if not (bycoords is None):
        focus_mask = np.zeros(mask_shape, dtype=np.uint8)
        focus_mask[bycoords[0]:bycoords[1],bycoords[2]:bycoords[3]] = 255
        return BinaryImage(focus_mask.astype("uint8"))

    #this case only works for white background
    if not use_preset:
        thresh_img = np.zeros(mask_shape).astype(np.uint8)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for rnum, r in enumerate(hsv_img):
            for cnum, c in enumerate(r):
                sat = c[1]
                is_white = sat < 0.1 * cfg.SAT_RANGE
                if is_white:
                    thresh_img[rnum][cnum] = 0
                else:
                    thresh_img[rnum][cnum] = 255
        # return BinaryImage(thresh_img.astype(np.uint8))

        #remove noise by blurring and re-thresholding, then flip colors
        blur_amt = 20.0
        thresh_img = cv2.GaussianBlur(thresh_img, (21, 21), blur_amt)
        thresh_img[np.where(thresh_img > 3)] = 255
        thresh_img = 255 - thresh_img

        # return BinaryImage(thresh_img.astype(np.uint8))

        points = [b[::-1] for b in BinaryImage(thresh_img).nonzero_pixels()]

        #get 4 corner vertices
        lower_left = min(points, key=lambda p:p[0] + p[1])
        upper_right = min(points, key=lambda p:-p[0] - p[1])
        lower_right = min(points, key=lambda p:-p[0] + p[1])
        upper_left = min(points, key=lambda p: p[0] - p[1])
        points = np.array([lower_left, lower_right, upper_right, upper_left])

        #blurring makes region smaller, so re-enlarge
        mean = np.mean(points, axis=0).astype("int32")
        points -= mean
        norms = np.apply_along_axis(np.linalg.norm, 1, points)
        points = (points * (1 + blur_amt/1.4/norms.reshape(-1,1))).astype("int32")
        points += mean

        #compute rectangle by stretching to outer points
        low_y = min(lower_left[1], lower_right[1])
        low_x = min(lower_left[0], upper_left[0])
        high_x = max(lower_right[0], upper_right[0])
        high_y = max(upper_left[1], upper_right[1])
    else:
        points = pickle.load(open("src/tpc/config.crop.p", "rb"))
        #get 4 corner vertices
        lower_left = min(points, key=lambda p:p[0] + p[1])
        upper_right = min(points, key=lambda p:-p[0] - p[1])
        lower_right = min(points, key=lambda p:-p[0] + p[1])
        upper_left = min(points, key=lambda p: p[0] - p[1])
        points = np.array([lower_left, lower_right, upper_right, upper_left])
        #compute rectangle by stretching to outer points
        low_y = min(lower_left[1], lower_right[1])
        low_x = min(lower_left[0], upper_left[0])
        high_x = max(lower_right[0], upper_right[0])
        high_y = max(upper_left[1], upper_right[1])

    if arc:
        #compute arc (invisible to camera)
        upper_left = [low_x, high_y]
        mid_left = [low_x + 10, low_y * 7.0/12.0 + high_y * 5.0/12.0]
        arc_left = [low_x * 3.0/4.0 + high_x * 1.0/4.0, low_y * 3.0/4.0 + high_y * 1.0/4.0]
        arc_mid = [(low_x + high_x)/2.0, low_y * 5.0/6.0 + high_y * 1.0/6.0]
        arc_right = [low_x * 1.0/4.0 + high_x * 3.0/4.0, low_y * 3.0/4.0 + high_y * 1.0/4.0]
        mid_right = [high_x - 10, low_y * 7.0/12.0 + high_y * 5.0/12.0]
        upper_right = [high_x, high_y]

        points = np.array([upper_left, mid_left, arc_left, arc_mid, arc_right, mid_right, upper_right])
    else:
        points = np.array([upper_left, lower_left, lower_right, upper_right])

    # for p in points:
    #     img = draw_point(img, p[::-1])

    #create mask
    focus_mask = np.zeros(mask_shape, dtype=np.uint8)
    rr, cc = polygon(points[:,1], points[:,0])
    focus_mask[rr,cc] = 255
    if viz:
        ctrs = points.reshape((points.shape[0], 1, points.shape[1]))
        cv2.drawContours(img, [ctrs], -1, [0, 255, 0], 3)
        cv2.imwrite("crop.png", img)
        cv2.imwrite("mask.png", focus_mask)
    return BinaryImage(focus_mask.astype("uint8"))

if __name__ == "__main__":
    set_crop()
