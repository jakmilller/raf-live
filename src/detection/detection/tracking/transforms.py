#!/usr/bin/env python3

import rclpy
import rclpy.duration
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point, Vector3, PointStamped
from cv_bridge import CvBridge
import numpy as np


class CoordinateTransforms:
    def __init__(self, node):
        """
        Initialize coordinate transforms helper
        
        Args:
            node: ROS node instance with tf_buffer, bridge, camera info, etc.
        """
        self.node = node
        self.bridge = CvBridge()
    
    def get_finger_midpoint_in_end_effector_frame(self):
        """Get the finger midpoint position in the end effector frame"""
        try:
            # Get finger pad positions in the end-effector frame
            right_finger_transform = self.node.tf_buffer.lookup_transform(
                'end_effector_link', 'right_inner_finger_pad', rclpy.time.Time())
            left_finger_transform = self.node.tf_buffer.lookup_transform(
                'end_effector_link', 'left_inner_finger_pad', rclpy.time.Time())
                
            # Calculate midpoint of finger pads in the end-effector frame
            right_pos = right_finger_transform.transform.translation
            left_pos = left_finger_transform.transform.translation
            finger_midpoint = Point()
            finger_midpoint.x = (right_pos.x + left_pos.x) / 2.0
            finger_midpoint.y = (right_pos.y + left_pos.y) / 2.0
            finger_midpoint.z = (right_pos.z + left_pos.z) / 2.0
            # Add half the 2f 140 finger pad length to the z coordinate
            finger_midpoint.z += 0.03  # Adjust based on your gripper's finger length
            
            return finger_midpoint
            
        except Exception as e:
            self.node.get_logger().warn(f"Could not get finger midpoint: {e}", throttle_duration_sec=2.0)
            return None
    
    
    def calculate_position_vector_from_mask(self, pixel_x, pixel_y, segmentation_mask, distance):
        """
        Calculate position vector using mask-averaged depth
        
        Args:
            pixel_x: Target pixel x coordinate
            pixel_y: Target pixel y coordinate
            segmentation_mask: Binary segmentation mask
            distance: How far in the z-direction the target should be from the end of the finger (m)
            
        Returns:
            Vector3: Position vector from finger to target, None if failed
        """
        if self.node.camera_info is None or self.node.latest_depth_image is None:
            return None
        
        # get the pixel indices of the segmentation
        object_pixels = segmentation_mask == 255
        
        try:
            # Get depth at pixel
            depth_image = self.bridge.imgmsg_to_cv2(self.node.latest_depth_image, desired_encoding='passthrough')
            mask_depths = depth_image[object_pixels]
            mask_depths = mask_depths[mask_depths > 0]  # Filter out invalid depths
    
            if len(mask_depths) == 0:
                return None
                
            # use the average mask depth instead of a single point
            avg_depth = np.mean(mask_depths) / 1000.0  # Convert mm to m
            
            # Convert pixel to 3D point in camera frame
            target_x = (pixel_x - self.node.cx) * avg_depth / self.node.fx
            target_y = (pixel_y - self.node.cy) * avg_depth / self.node.fy
            target_z = avg_depth
            
            # Transform target position to end effector frame
            target_stamped = PointStamped()
            target_stamped.header.frame_id = 'realsense_link'
            target_stamped.header.stamp = self.node.get_clock().now().to_msg()
            target_stamped.point.x = target_x
            target_stamped.point.y = target_y
            target_stamped.point.z = target_z
            
            # Transform to end effector frame
            target_in_effector = self.node.tf_buffer.transform(
                target_stamped, 'end_effector_link', timeout=rclpy.duration.Duration(seconds=0.5))
            
            # Get finger midpoint
            finger_midpoint = self.get_finger_midpoint_in_end_effector_frame()

            if finger_midpoint is None:
                return None
            
            # add distance to z coordinate of finger midpoint
            finger_midpoint.z += distance
            
            # Calculate vector from finger to target
            vector = Vector3()
            vector.x = target_in_effector.point.x - finger_midpoint.x
            vector.y = target_in_effector.point.y - finger_midpoint.y
            vector.z = target_in_effector.point.z - finger_midpoint.z
            
            return vector
            
        except Exception as e:
            self.node.get_logger().error(f"Error calculating position vector: {e}")
            return None