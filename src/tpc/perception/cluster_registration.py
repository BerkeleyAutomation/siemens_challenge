import numpy as np
import cv2
import IPython
from tpc.perception.connected_components import get_cluster_info
from tpc.perception.groups import Group
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

import tpc.config.config_tpc as cfg
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

def run_connected_components(img, viz=False):
    """ Generates mask for
    each cluster of objects
    Parameters
    ----------
    img :obj:`ColorImage`
        color image masked to
        white working area
    viz : boolean
        if true, displays proposed grasps
    Returns
    -------
    :obj:list of `Group`
    """
    fg = img.subtract_background()
    # fg = img.foreground_mask(cfg.COLOR_TOL, ignore_black=True)
    if viz:
        cv2.imwrite("debug_imgs/mask.png", fg.data)

    groups = get_cluster_info(fg)

    if viz:
        display_grasps(img, groups)

    return groups

def draw_point(img, point):
    box_color = (255, 0, 0)
    box_size = 5
    img[int(point[0] - box_size):int(point[0] + box_size),
        int(point[1] - box_size):int(point[1] + box_size)] = box_color
    return img

def display_grasps(img, groups,name="debug_imgs/grasps", as_tuple = False):
    """ Displays the proposed grasps
    Parameters
    ----------
    img :obj:`ColorImage`
        color image masked to
        white working area
    center_masses :list of `Group`
    Returns
    -------
    :obj:`numpy.ndarray`
        visualization image
    """
    if len(groups) > 0:
        box_color = (255, 0, 0)
        line_color = box_color[::-1]
        img_data = np.copy(img.data)
        for i in range(len(groups)):
            if as_tuple:
                cm = groups[0]
                d = groups[1]
            else:
                cm = groups[i].cm
                d = groups[i].dir

            img_data = draw_point(img_data, cm)

            p0 = tuple((cm - d * cfg.LINE_SIZE/2)[::-1].astype('int32'))
            p1 = tuple((cm + d * cfg.LINE_SIZE/2)[::-1].astype('int32'))
            cv2.line(img_data, p0, p1, line_color, 2)
        #BGR to RGB
        rgb = np.fliplr(img_data.reshape(-1,3)).reshape(img_data.shape)
        plt.imshow(rgb)
        plt.axis('off')
        plt.savefig(name + ".png")
        if cfg.QUERY:
            plt.show()


def dist_mod(m, a, b):
    """
    returns shortest distance between a and b under mod m
    """
    diff = abs(a-b)
    if diff < m/2:
        return diff
    else:
        return m - diff

def get_hsv_hist(img):
    """ Separates image into bins by HSV and creates histograms
    Parameters
    ----------
    img :obj:`ColorImage`
        color image masked to relevant area
    Returns
    -------
    :obj:tuple of
        :obj:dict
            hsv value to count
        :obj:dict
            hsv value to list of pixel coordinates
    """
    hsv = cv2.cvtColor(img.data, cv2.COLOR_BGR2HSV)

    hue_counts = {}
    hue_pixels = {}

    for rnum, row in enumerate(hsv):
        for cnum, pix in enumerate(row):
            hue = pix[0]
            val = pix[2]
            sat = pix[1]
            #ignore white
            if not (sat < cfg.WHITE_FACTOR * cfg.SAT_RANGE):
                #black is its own bin
                if val < cfg.BLACK_FACTOR * cfg.VALUE_RANGE:
                    bin_hue = -1
                else:
                    non_black_hues = cfg.HUE_VALUES.keys()
                    bin_hue = min(non_black_hues, key = lambda h:
                        dist_mod(cfg.HUE_RANGE, hue, h))

                if bin_hue not in hue_counts:
                    hue_counts[bin_hue] = 0
                    hue_pixels[bin_hue] = []
                hue_counts[bin_hue] += 1
                hue_pixels[bin_hue].append([rnum, cnum])

    return hue_counts, hue_pixels

def view_hsv(img):
    """ Separates image into bins by HSV and creates visualization
    Parameters
    ----------
    img :obj:`ColorImage`
    Returns
    -------
    :obj:`ColorImage`
    """
    _, pixels = get_hsv_hist(img)

    view = np.zeros(img.data.shape)
    for hsv_col in pixels.keys():
        points = pixels[hsv_col]
        for p in points:
            #black case
            if hsv_col == -1:
                view[p[0]][p[1]] = [0, 255, 255]
            else:
                view[p[0]][p[1]] = [hsv_col, 128, 128]
    view = view.astype(np.uint8)
    col = cv2.cvtColor(view, cv2.COLOR_HSV2BGR)
    return ColorImage(col)

def hsv_classify(img):
    """ Classifies the lego by HSV
    Parameters
    ----------
    img :obj:`ColorImage`
        mask of lego
    Returns
    -------
    :integer
        class number from 0 to 7
    """
    hue_to_count, _ = get_hsv_hist(img)
    dominant_hue = max(hue_to_count, key=hue_to_count.get)

    all_hues = cfg.ALL_HUE_VALUES.keys()
    all_hues.sort()

    class_num = all_hues.index(dominant_hue)
    return class_num

def class_num_to_name(class_num):
    """ Gets the color name for the index
    Parameters
    ----------
    class_num :integer
        class number from 0 to 7
    Returns
    -------
    :obj:string
        color name
    """
    all_hues = cfg.ALL_HUE_VALUES.keys()
    all_hues.sort()
    hue = all_hues[class_num]
    color_name = cfg.ALL_HUE_VALUES[hue]
    return color_name

def is_valid_grasp(point, focus_mask):
    """ Checks that the point does not overlap
    another object in the cluster
    Parameters
    ----------
    point :obj:list of `numpy.ndarray`
        1x2 point
    focus_mask :obj:`BinaryImage`
        mask of object cluster
    Returns
    -------
    boolean
    """
    ymid = int(point[0])
    xmid = int(point[1])
    d = cfg.CHECK_RANGE

    check_box = focus_mask.data[ymid - d:ymid + d, xmid - d:xmid + d]
    num_nonzero = np.sum(check_box > 0)

    fraction_nonzero = (num_nonzero * 1.0)/((2 * d)**2)
    return fraction_nonzero < 0.2

def color_to_binary(img):
    """
    Parameters
    ----------
    img :obj:`ColorImage`
        mask of object cluster
    Returns
    -------
    :obj:`BinaryImage`
    """
    return BinaryImage(np.sum(255 * (img.data > 0), axis=2).astype(np.uint8))

def grasps_within_pile(color_mask):
    """ Finds any grasps within the pile
    Parameters
    ----------
    color_mask :obj:`ColorImage`
        mask of object cluster
    Returns
    -------
    :obj:list of `Group`
    """
    hue_counts, hue_pixels = get_hsv_hist(color_mask)

    individual_masks = []

    #color to binary
    focus_mask = color_to_binary(color_mask)

    #segment by hsv
    for block_color in hue_counts.keys():
        #same threshold values for number of objects
        if hue_counts[block_color] > cfg.SIZE_TOL:
            valid_pix = hue_pixels[block_color]
            obj_mask = focus_mask.mask_by_ind(np.array(valid_pix))
            individual_masks.append(obj_mask)

    if len(individual_masks) > 0:
        obj_focus_mask = individual_masks[0]
        for im in individual_masks[1:]:
            obj_focus_mask += im

    #for each hsv block, again separate by connectivity
    all_groups = []
    for i, obj_mask in enumerate(individual_masks):
        groups = get_cluster_info(obj_mask)

        for group in groups:
            #matches endpoints of line in visualization
            cm = group.cm
            d = group.dir

            grasp_top = cm + d * cfg.LINE_SIZE/2
            grasp_bot = cm - d * cfg.LINE_SIZE/2
            if is_valid_grasp(grasp_top, obj_focus_mask) and is_valid_grasp(grasp_bot, obj_focus_mask):
                all_groups.append(group)

    return all_groups
