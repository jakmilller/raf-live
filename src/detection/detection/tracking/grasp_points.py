#!/usr/bin/env python3

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


class GraspAnalyzer:
    def __init__(self, node=None):
        """
        Initialize grasp analyzer
        
        Args:
            node: ROS node instance for logging and camera info (optional)
        """
        self.node = node
    
    def analyze_grasp(self, mask, depth_image, single_bite, current_item=""):
        """
        Analyze grasp points and calculate grip value, width points, and angle
        
        Args:
            mask: Binary segmentation mask
            depth_image: Depth image from camera
            single_bite: Whether this is a single bite item
            current_item: Name of current food item
            
        Returns:
            dict: Dictionary containing:
                - grip_value: Calculated grip value (0-1)
                - width_p1: First width point coordinates
                - width_p2: Second width point coordinates  
                - food_angle: Food angle in degrees
                - centroid: Mask centroid coordinates
                - success: Whether analysis was successful
        """
        result = {
            'grip_value': None,
            'width_p1': None,
            'width_p2': None,
            'food_angle': None,
            'centroid': None,
            'success': False
        }
        
        try:
            # Calculate food width and grip points
            centroid, grip_val, width_p1, width_p2 = self._calculate_food_width(mask, depth_image, single_bite)
            
            # Calculate food angle using PCA
            food_angle = self._get_food_angle_pca(mask)
            
            result.update({
                'grip_value': grip_val,
                'width_p1': width_p1,
                'width_p2': width_p2,
                'food_angle': food_angle,
                'centroid': centroid,
                'success': True
            })
            
            return result
            
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Error in grasp analysis: {str(e)}")
            return result
    
    def _get_mask_centroid(self, mask):
        """Find the centroid of a binary mask"""
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None 
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    
    def _calculate_food_width(self, mask, depth_image, single_bite):
        """Calculate food width and grip points exactly like perception node"""
        # Convert mask to proper format for findContours
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroid = self._get_mask_centroid(mask)

        if not contours:
            if self.node:
                self.node.get_logger().error("No contours found in mask")
            return None, None, None

        if centroid is None:
            if self.node:
                self.node.get_logger().error("Could not calculate centroid")
            return None, None, None

        largest_contour = max(contours, key=cv2.contourArea)

        # Check if contour has enough points for minAreaRect
        if len(largest_contour) < 5:
            if self.node:
                self.node.get_logger().error("Contour has insufficient points for minAreaRect")
            return None, None, None

        # get a rotated rectangle around the segmentation
        rect = cv2.minAreaRect(largest_contour)
        if rect is None:
            if self.node:
                self.node.get_logger().error("minAreaRect returned None")
            return None, None, None

        # get the box points of the rectangle and convert to integers
        box = cv2.boxPoints(rect)
        if box is None or len(box) != 4:
            if self.node:
                self.node.get_logger().error("boxPoints returned invalid data")
            return None, None, None

        box = np.intp(box)

        try:
            if np.linalg.norm(box[0]-box[1]) < np.linalg.norm(box[1]-box[2]):
                # grab points for width calculation
                p1 = (box[1]+box[2])/2
                p2 = (box[3]+box[0])/2

                if not single_bite:
                    # lower point has the higher pixel y value
                    if box[1][1]>box[2][1]:
                        lower1 = box[1]
                    else:
                        lower1 = box[2]

                    if box[3][1]>box[0][1]:
                        lower2 = box[3]
                    else:
                        lower2 = box[0]
                    
                    # divide it more to get the lower half for grasping multibite foods
                    p1 = (p1+lower1)/2
                    p2 = (p2+lower2)/2
                    centroid = tuple(map(int, (p1+p2)/2))

            else:
                p1 = (box[0]+box[1])/2
                p2 = (box[2]+box[3])/2

                if not single_bite:
                    # lower point has the higher pixel y value
                    if box[0][1]>box[1][1]:
                        lower1 = box[0]
                    else:
                        lower1 = box[1]

                    if box[2][1]>box[3][1]:
                        lower2 = box[2]
                    else:
                        lower2 = box[3]
                    
                    # divide it more to get the lower half for grasping multibite foods
                    p1 = (p1+lower1)/2
                    p2 = (p2+lower2)/2
                    centroid = tuple(map(int, (p1+p2)/2))

            # Convert midpoints to integers and find the nearest point on the mask
            width_p1 = self._proj_pix2mask(tuple(map(int, p1)), mask)
            width_p2 = self._proj_pix2mask(tuple(map(int, p2)), mask)

            if width_p1 is None or width_p2 is None:
                if self.node:
                    self.node.get_logger().error("Could not project points to mask")
                return None, None, None, None

            # get the coordinates relative to RealSense of width points
            rs_width_p1, success1 = self._pixel_to_rs_frame(width_p1[0], width_p1[1], depth_image)
            rs_width_p2, success2 = self._pixel_to_rs_frame(width_p2[0], width_p2[1], depth_image)

            if not success1 or not success2 or rs_width_p1 is None or rs_width_p2 is None:
                if self.node:
                    self.node.get_logger().warn("Could not convert width points to RealSense coordinates")
                grip_val = None
            else: 
                # get true distances of points from each other (ignore depth for accuracy)
                rs_width_p1_2d = rs_width_p1[:2]
                rs_width_p2_2d = rs_width_p2[:2]
    
                # Calculate the Euclidean distance between points
                width = np.linalg.norm(rs_width_p1_2d - rs_width_p2_2d)
                width_mm = width*1000
    
                # cubic regression function mapping gripper width to grip value
                grip_val = -0.0000246123*width_mm**3 + 0.00342851*width_mm**2 -0.667919*width_mm +80.44066
    
                # the grip value will move the gripper to the exact width of the food, but you'll want the insurance if it being a little wider so it grasps successfully
                grip_val = grip_val-4
    
                # make sure it doesn't break fingers
                if grip_val > 80:
                    grip_val = 80
                elif grip_val < 0:
                    grip_val = 0
    
                grip_val = round(grip_val)/100

            return centroid, grip_val, width_p1, width_p2

        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Error in _calculate_food_width: {str(e)}")
            return None, None, None, None
    
    def _get_food_angle_pca(self, mask):
        """Calculate food angle using PCA"""
        ys, xs = np.where(mask > 0)
        points = np.column_stack((xs, ys))

        if points.shape[0] < 2:
            return 0.0  # default angle

        # Mean center the data
        mean = np.mean(points, axis=0)
        centered = points - mean

        # Compute covariance and eigenvectors
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Choose the major axis (eigenvector with largest eigenvalue)
        major_axis = eigvecs[:, np.argmax(eigvals)]

        angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

        # wrap angle to correct range for servoing
        if -180 <= angle <= -45:
            angle += 180

        # make vertical 0 degrees
        angle -= 90

        return angle
    
    def _pixel_to_rs_frame(self, pixel_x, pixel_y, depth_image):
        """Convert pixel coordinates to 3D coordinates relative to RealSense camera"""
        if not self.node or not hasattr(self.node, 'camera_info') or self.node.camera_info is None:
            return None, False
            
        # Camera intrinsics
        fx = self.node.camera_info.k[0]
        fy = self.node.camera_info.k[4] 
        cx = self.node.camera_info.k[2]
        cy = self.node.camera_info.k[5]
        
        # Get depth value at pixel
        depth = depth_image[int(pixel_y), int(pixel_x)] / 1000.0  # Convert mm to m

        if depth <= 0:
            return None, False
            
        # Convert to 3D camera coordinates
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z]), True
    
    def _proj_pix2mask(self, px, mask):
        """Project pixel to nearest mask point"""
        ys, xs = np.where(mask > 0)
        if not len(ys):
            return px
        mask_pixels = np.vstack((xs,ys)).T
        neigh = NearestNeighbors()
        neigh.fit(mask_pixels)
        dists, idxs = neigh.kneighbors(np.array(px).reshape(1,-1), 1, return_distance=True)
        projected_px = mask_pixels[idxs.squeeze()]
        return projected_px