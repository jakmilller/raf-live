#!/usr/bin/env python3

import rclpy
import rclpy.duration
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import numpy as np


class RVizVisualizer:
    def __init__(self, node):
        """
        Initialize RViz visualizer
        
        Args:
            node: ROS node instance that has the publishers and other required attributes
        """
        self.node = node
        self.bridge = CvBridge()
        
        # Create publishers
        self.target_point_pub = node.create_publisher(Marker, '/target_point', 10)
        self.gripper_point_pub = node.create_publisher(Marker, '/gripper_point', 10)
        self.vis_vector_pub = node.create_publisher(MarkerArray, '/vis_vector', 10)
    
    def publish_markers(self, cx, cy, position_vector):
        """
        Publish RViz visualization markers
        
        Args:
            cx: Center x pixel coordinate
            cy: Center y pixel coordinate  
            position_vector: Vector3 position vector
        """
        try:
            current_time = self.node.get_clock().now().to_msg()
            
            # Get finger midpoint
            finger_midpoint = self._get_finger_midpoint_in_end_effector_frame()
            if finger_midpoint is None:
                return
            
            # Get food position in end effector frame
            food_in_effector = self._get_food_position_in_end_effector_frame(cx, cy, current_time)
            if food_in_effector is None:
                return
            
            # Publish all markers
            self._publish_target_point_marker(food_in_effector, current_time)
            self._publish_gripper_point_marker(finger_midpoint, current_time)
            self._publish_vector_arrow_marker(finger_midpoint, food_in_effector, current_time)
            
        except Exception as e:
            self.node.get_logger().error(f"Error publishing RViz markers: {e}")
    
    def _get_finger_midpoint_in_end_effector_frame(self):
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
    
    def _get_food_position_in_end_effector_frame(self, cx, cy, current_time):
        """Get food position transformed to end effector frame"""
        try:
            # Get depth at pixel
            depth_image = self.bridge.imgmsg_to_cv2(self.node.latest_depth_image, desired_encoding='passthrough')
            depth_mm = depth_image[cy, cx]
            depth_m = depth_mm / 1000.0
            
            if depth_m <= 0:
                return None
            
            # Convert pixel to 3D point and transform to end effector frame
            food_x = (cx - self.node.cx) * depth_m / self.node.fx
            food_y = (cy - self.node.cy) * depth_m / self.node.fy
            food_z = depth_m
            
            food_stamped = PointStamped()
            food_stamped.header.frame_id = 'realsense_link'
            food_stamped.header.stamp = current_time
            food_stamped.point.x = food_x
            food_stamped.point.y = food_y
            food_stamped.point.z = food_z
            
            food_in_effector = self.node.tf_buffer.transform(
                food_stamped, 'end_effector_link', timeout=rclpy.duration.Duration(seconds=0.5))
            
            return food_in_effector
            
        except Exception as e:
            self.node.get_logger().error(f"Error getting food position: {e}")
            return None
    
    def _publish_target_point_marker(self, food_in_effector, current_time):
        """Publish target point marker (food centroid) - Green sphere"""
        target_marker = Marker()
        target_marker.header.frame_id = "end_effector_link"
        target_marker.header.stamp = current_time
        target_marker.ns = "target_point"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position = food_in_effector.point
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.01
        target_marker.scale.y = 0.01
        target_marker.scale.z = 0.01
        target_marker.color.r = 0.0
        target_marker.color.g = 1.0
        target_marker.color.b = 0.0
        target_marker.color.a = 1.0
        self.target_point_pub.publish(target_marker)
    
    def _publish_gripper_point_marker(self, finger_midpoint, current_time):
        """Publish gripper point marker (finger midpoint) - Red sphere"""
        gripper_marker = Marker()
        gripper_marker.header.frame_id = "end_effector_link"
        gripper_marker.header.stamp = current_time
        gripper_marker.ns = "gripper_point"
        gripper_marker.id = 0
        gripper_marker.type = Marker.SPHERE
        gripper_marker.action = Marker.ADD
        gripper_marker.pose.position = finger_midpoint
        gripper_marker.pose.orientation.w = 1.0
        gripper_marker.scale.x = 0.01
        gripper_marker.scale.y = 0.01
        gripper_marker.scale.z = 0.01
        gripper_marker.color.r = 1.0
        gripper_marker.color.g = 0.0
        gripper_marker.color.b = 0.0
        gripper_marker.color.a = 1.0
        self.gripper_point_pub.publish(gripper_marker)
    
    def _publish_vector_arrow_marker(self, finger_midpoint, food_in_effector, current_time):
        """Publish vector arrow (from finger to food) - Blue arrow"""
        marker_array = MarkerArray()
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "end_effector_link"
        arrow_marker.header.stamp = current_time
        arrow_marker.ns = "vis_vector"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # Arrow points FROM finger midpoint TO food
        arrow_marker.points = [finger_midpoint, food_in_effector.point]
        
        arrow_marker.scale.x = 0.005  # shaft width
        arrow_marker.scale.y = 0.01   # head width
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 0.0
        arrow_marker.color.b = 1.0
        arrow_marker.color.a = 0.8
        
        marker_array.markers.append(arrow_marker)
        self.vis_vector_pub.publish(marker_array)