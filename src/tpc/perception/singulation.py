# To run this test script run python src/tpc/perception/singulation.py
# from the root of the repo

import matplotlib.pyplot as plt
import skimage as sk
from skimage.draw import polygon
import sys
import numpy as np
from sklearn.decomposition import PCA
import cv2
import IPython

import tpc.config.config_tpc as cfg
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

class Singulation():
    def __init__(self, img, focus_mask, obj_masks, goal_p=None, waypoints=None, gripper_angle=None):
        """
        Parameters
        ----------
        img :obj:`ColorImage`
            original image
        focus_mask :obj:`BinaryImage`
            crop of workspace
        obj_masks :list:obj:`BinaryImage`
            list of crops of object clusters
        """
        self.img = img.copy()
        self.focus_mask = focus_mask.copy()
        self.workspace_img = self.img.mask_binary(self.focus_mask)

        #singulate smallest pile first
        self.obj_masks = [o.copy() for o in obj_masks]
        self.obj_masks.sort(key=lambda m:len(m.nonzero_pixels()))

        #run computations in advance
        self.goal_p = goal_p
        self.waypoints = waypoints
        self.gripper_angle = gripper_angle
        if self.goal_p is None or self.waypoints is None or self.gripper_angle is None:
            self.obj_mask = self.obj_masks[0]
            self.other_obj_masks = self.obj_masks[1:]
            self.compute_singulation()

    def display_border(self, border):
        """ helper to display the border between the segments

        Parameters
        ----------
        border : :obj:`numpy.ndarray`
            nx2 array of n border pixels
        """
        plt.imshow(self.img.data)
        plt.axis('off')
        plt.plot(border[:,1], border[:,0], 'o')
        plt.show()

    def display_segments(self, segmented):
        """ helper to display the border between the segments

        Parameters
        ----------
        segmented : :obj:`SegmentedImage`
            the segments to be displayed
        """
        for i in range(segmented.num_segments):
            brick = self.img.mask_binary(segmented.segment_mask(i))
            plt.imshow(brick.data)
            plt.show()

    def get_border(self):
        """ Splits the object pile into two segments, and
        returns the pixels that border the two segments to be
        used as the pushing direction

        Returns
        -------
        :obj:`numpy.ndarray`
            nx2 array of n border pixels
        """
        bricks = self.img.mask_binary(self.obj_mask)
        segmented = bricks.segment_kmeans(.1, 2)
        border = segmented.border_pixels()

        # self.display_border(border)
        # self.display_segments(segmented)

        #rare case- 2 clusters aren't touching
        if len(border) == 0:
            print("no boundary points")
            border = self.obj_mask.nonzero_pixels()

        return border

    def get_goal_pixel(self):
        """ Finds the goal pixel, which is the point
        in the workspace furthest from all walls and
        other object clusters

        Returns
        -------
        :obj:`numpy.ndarray`
            1x2 array of the goal pixel
        """
        occupied_space = self.focus_mask.inverse()
        new_mask = occupied_space.data
        for i in range(len(self.other_obj_masks)):
            new_mask += self.other_obj_masks[i].data
            # occupied_space += self.other_obj_masks[i]
        new_mask += self.obj_mask.data
        # occupied_space += self.obj_mask
        occupied_space = BinaryImage(new_mask.astype(np.uint8))
        goal_pixel = occupied_space.most_free_pixel()
        return goal_pixel

    def get_direction(self, border):
        """ Finds the direction in which to push the
        pile to separate it the most using the direction
        of the border pixels

        Parameters
        ----------
        border :obj:`numpy.ndarray`
            nx2 array of n border pixels
        Returns
        -------
        :obj:`numpy.ndarray`
            1x2 vector pointing in the direction in which the
            robot should singulate
        :obj: int
            distance the robot should push to go through the
            entire pile
        """
        #optimal pushing direction
        pca = PCA(2)
        pca.fit(border)
        border_dir = pca.components_[0]
        border_dir /= np.linalg.norm(border_dir)
        max_distance = pca.explained_variance_[0]
        distance = .4*max_distance

        push_dir = border_dir

        return push_dir, distance

    def get_border_endpoints(self, mean, direction):
        """ Finds the endpoints of the first push along the borer

        Parameters
        ----------
        mean :obj:`numpy.ndarray`
            1x2 vector, mean of border pixels
        direction :obj:`numpy.ndarray`
            1x2 vector pointing in the direction in which the
            robot should singulate
        """
        #start_p = self.obj_mask.closest_zero_pixel(mean, -1*direction)
        #end_p = self.obj_mask.closest_zero_pixel(mean, direction)
        start_p = self.obj_mask.closest_nonzero_pixel(mean, -1*direction)
        end_p = self.obj_mask.closest_nonzero_pixel(mean, direction)

        #should be closer to goal pixel after border push than before it
        if np.linalg.norm(self.goal_p - start_p) < np.linalg.norm(self.goal_p - end_p):
            start_p, end_p = end_p, start_p

        #small adjustments (empirically does better)
        start_p += (start_p - end_p)/(np.linalg.norm(start_p-end_p)+1e-8) * cfg.SINGULATE_START_FACTOR
        end_p = start_p * (1-cfg.SINGULATE_END_FACTOR) + end_p * cfg.SINGULATE_END_FACTOR

        return start_p, end_p

    def get_goal_waypoint(self, mean):
        """ Finds the endpoints of the first push along the border

        Parameters
        ----------
        mean :obj:`numpy.ndarray`
            1x2 vector, mean of border pixels
        """
        if self.goal_p is None:
            self.goal_p = self.get_goal_pixel()
        return self.goal_p

        # goal_dir = self.goal_p - mean
        # goal_dir = goal_dir / np.linalg.norm(goal_dir)
        # towards_goal = self.obj_mask.closest_zero_pixel(mean, goal_dir, w=40)

        # #don't want to push too far if goal pixel is inside object cluster
        # closer_goal = min([self.goal_p, towards_goal], key = lambda p: np.linalg.norm(p - mean))

        # return closer_goal

    def compute_singulation(self):
        """ Finds the direction in which the robot should push
        the pile to separate it the most, and the gripper angle
        which will keep objects farthest from other piles
        """
        self.goal_p = self.get_goal_pixel()

        border = self.get_border()
        direction, _ = self.get_direction(border)
        mean = np.mean(border, axis=0)

        start_p, end_p = self.get_border_endpoints(mean, direction)
        towards_goal_p = self.get_goal_waypoint(mean)
        self.waypoints = [start_p, end_p, towards_goal_p]

        self.gripper_angle = 0

    def get_singulation(self):
        """
        Returns
        -------
        :list:obj:`numpy.ndarray`
            list of 1x2 vectors representing the waypoints of the singulation
        float
            angle of gripper, aligned towards free space
        :obj: `numpy.ndarray`
            1x2 vector representing the goal pixel
        """
        if self.waypoints is None or self.gripper_angle is None or self.goal_p is None:
            self.compute_singulation()

        return self.waypoints, self.gripper_angle, self.goal_p

    def display_singulation(self, name="debug_imgs/singulate"):
        """
        saves visualization of singulation trajectories
        """
        if self.waypoints is None or self.gripper_angle is None or self.goal_p is None:
            self.compute_singulation()

        plt.figure()
        ax = plt.axes()
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]
            end = self.waypoints[i+1]
            ax.arrow(
                start[1], start[0],
                end[1] - start[1], end[0] - start[0],
                head_width = 10, head_length = 10
            )

        #BGR to RGB
        image = self.workspace_img
        rgb = np.fliplr(image.data.reshape(-1,3)).reshape(image.data.shape)
        plt.imshow(rgb)

        plt.plot(self.goal_p[1], self.goal_p[0], 'bo')
        plt.axis('off')
        plt.savefig(name + ".png")
        if cfg.QUERY:
            plt.show()
