from tpc.perception.groups import Group
import tpc.config.config_tpc as cfg
from tpc.perception.crop import crop_img
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')
import numpy as np
from tpc.perception.connected_components import get_cluster_info, merge_groups
import IPython

"""
New metric for whether to ask for help
For segmentation mask, compute minimum distance to another segmentation max
Take maximum of above, and check if less than threshold t
(basically, ask for help if all objects are within t of at least 1 other object)
cond(argmax_i argmin_j metric(i, j)) == forall i, therexists j cond(metric(i, j))
"""

def find_isolated_objects_by_overlap(bboxes):
    valid_bboxes = []
    for curr_ind in range(len(bboxes)):
        curr_bbox = bboxes[curr_ind]
        overlap = False
        for test_ind in range(len(bboxes)):
            if curr_ind != test_ind:
                test_bbox = bboxes[test_ind]
                if curr_bbox.test_overlap(test_bbox):
                    overlap = True
                    break
        if not overlap:
            valid_bboxes.append(curr_bbox)
    return valid_bboxes

def find_isolated_objects_by_distance(bboxes, col_img):
    if len(bboxes) == 1:
        return bboxes

    groups = [box.to_group(col_img.data, col_img) for box in bboxes]
    valid_bboxes = []
    for curr_ind in range(len(groups)):
        curr_group = groups[curr_ind]
        distances = [curr_group.cm_dist(groups[i]) for i in range(len(groups)) if i != curr_ind]
        if min(distances) >= cfg.ISOLATED_TOL:
            valid_bboxes.append(bboxes[curr_ind])
    return valid_bboxes

#effective, but too hard to figure out which class labels correspond to which groups
# def find_almost_isolated_objects(bboxes, col_img):
#     fg_imgs = [box.to_mask(col_img.data, col_img) for box in bboxes]
#     if len(fg_imgs) > 0:
#         total_mask = fg_imgs[0][0]
#         for im in fg_imgs[1:]:
#             total_mask += im[0]
#         groups = get_cluster_info(total_mask, tol = 3)

#         valid_groups = []
#         for g in groups:
#             if not g.was_merged:
#                 valid_groups.append(g)
#         return valid_groups
#     else:
#         return []

def select_first_obj(bboxes):
    bottom_left_bbox =  min(bboxes, key = lambda box: box.xmin - box.ymin)

    return bottom_left_bbox

def format_net_bboxes(net_output, shape, maskrcnn=False):
    #first filter by confidence
    if maskrcnn:
        scores = net_output['scores'].round(2)
        classes = net_output['class_ids']
        boxes = net_output['rois']
    else:
        scores = net_output['detection_scores'].round(2)
        classes = net_output['detection_classes']
        boxes = net_output['detection_boxes']
    num_valid = 0
    while num_valid < len(scores) and scores[num_valid] > cfg.CONFIDENCE_THRESH:
        num_valid += 1
    filtered_output = []

    for i in range(num_valid):
        points = [boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]]
        box = Bbox(points, classes[i], scores[i])
        if not maskrcnn:
            box.scale_from_net(shape)
        box.convert_label_from_net()
        filtered_output.append(box)
        #to_add = box.filter_far_boxes()
        #if to_add:
            #filtered_output.append(box)

    return filtered_output

def draw_boxes(bboxes, img):
    img = np.copy(img)
    for b in bboxes:
        img = b.draw(img)
    return img

class Bbox:
    """
    class for object bounding boxes
    """
    def __init__(self, points, label, confidence=None):
        #points should have format [xmin, ymin, xmax, ymax]
        #label should be an integer
        self.xmin = points[0]
        self.ymin = points[1]
        self.xmax = points[2]
        self.ymax = points[3]
        self.label = label
        self.points = points
        self.prob = confidence

    def test_overlap(self, other):
        if self.xmin > other.xmax or self.xmax < other.xmin:
            return False
        if self.ymin > other.ymax or self.ymax < other.ymin:
            return False
        return True

    def to_mask(self, c_img, col_img, tol=cfg.COLOR_TOL):
        obj_mask = crop_img(c_img, bycoords = [self.ymin, self.ymax, self.xmin, self.xmax])
        obj_workspace_img = col_img.mask_binary(obj_mask)
        # fg = obj_workspace_img.foreground_mask(cfg.COLOR_TOL, ignore_black=True)
        fg = obj_workspace_img.foreground_mask(tol, ignore_black=True)
        return fg, obj_workspace_img

    def to_group(self, c_img, col_img):
        fg, obj_w = self.to_mask(c_img, col_img)
        # cv2.imwrite("debug_imgs/test.png", obj_w.data)
        # cv2.imwrite("debug_imgs/test2.png", fg.data)
        groups = get_cluster_info(fg)
        curr_tol = cfg.COLOR_TOL
        while len(groups) == 0 and curr_tol > 10:
            curr_tol -= 5
            #retry with lower tolerance- probably white object
            fg, obj_w = self.to_mask(c_img, col_img, tol=curr_tol)
            groups = get_cluster_info(fg)

        if len(groups) == 0:
            raise ValueError("no object within bounding box")

        return groups[0]

    def scale_from_net(self, shape):
        h, w, dim = shape
        self.xmin = int(self.xmin * w)
        self.xmax = int(self.xmax * w)
        self.ymin = int(self.ymin * h)
        self.ymax = int(self.ymax * h)

    def convert_label_from_net(self):
        name = cfg.net_labels[self.label]
        self.label = cfg.labels.index(name)

    def draw(self, img):
        color = (255, 0, 0)
        new_img = np.copy(img)
        new_img[self.ymin:self.ymax, self.xmin:self.xmax] = color
        width = 3
        xlo, xhi = self.xmin + width, self.xmax - width
        ylo, yhi = self.ymin + width, self.ymax - width
        new_img[ylo:yhi,xlo:xhi] = img[ylo:yhi,xlo:xhi]
        return new_img

    def filter_far_boxes(self):
        if self.ymax < 120:
            return False
        return True
