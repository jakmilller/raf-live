#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import time

class AutonomousChecker(Node):
    def __init__(self, config):
        super().__init__('autonomous_checker')
        
        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.camera_info = None
        
        # Load autonomous config parameters
        self.check_pixel_x = config['feeding']['autonomous']['check_pixel_x']
        self.check_pixel_y = config['feeding']['autonomous']['check_pixel_y']
        self.consecutive_checks = config['feeding']['autonomous']['consecutive_checks']
        self.check_interval = config['feeding']['autonomous']['check_interval']
        
        # Subscribers for camera data
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', 
            self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', 
            self.camera_info_callback, 10)
        
        self.get_logger().info('Autonomous checker initialized')
    
    def depth_callback(self, msg):
        self.latest_depth_image = msg
    
    def camera_info_callback(self, msg):
        self.camera_info = msg
    
    def get_depth_at_pixel(self, x, y):
        """Get depth value at specific pixel coordinates"""
        if self.latest_depth_image is None:
            self.get_logger().warn("No depth image available")
            return None
            
        try:
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
            
            # Ensure pixel coordinates are within image bounds
            height, width = depth_image.shape
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            # Get depth value (convert from mm to meters)
            depth_mm = depth_image[y, x]
            depth_m = depth_mm / 1000.0
            
            return depth_m
        except Exception as e:
            self.get_logger().error(f"Error getting depth at pixel: {str(e)}")
            return None
    
    def is_object_grasped(self):
        """
        Check if object is grasped based on depth at the fixed pickup pixel coordinates
        These coordinates represent where food appears when picked up by the gripper
        """
        depth = self.get_depth_at_pixel(self.check_pixel_x, self.check_pixel_y)
        
        if depth is None:
            return False
        
        self.get_logger().info(f"Food depth: ({self.check_pixel_x}, {self.check_pixel_y}): {depth:.3f}m")
        
        # Check if depth indicates object is picked up (within expected range for grasped food)
        if depth<0.25 and depth>0.15:
            return True
        else:
            return False
    
    def is_object_removed(self):
        """
        Check if object has been removed/eaten from gripper
        If no object is detected at the pickup pixel, it's been removed
        """
        depth = self.get_depth_at_pixel(self.check_pixel_x, self.check_pixel_y)
        
        if depth is None:
            self.get_logger().info("No valid depth found at pixel")
            return False
        
        self.get_logger().info(f"Food depth:({self.check_pixel_x}, {self.check_pixel_y}): {depth:.3f}m")
        
        # If depth is outside the pickup range, object is likely removed
        if depth > 0.2:
            return True
        else:
            return False
    
    def check_object_grasped(self, timeout=None):
        """
        Continuously check if object is grasped at the fixed pickup pixel coordinates
        
        Args:
            timeout: Maximum time to wait for grasp confirmation (uses config default if None)
            
        Returns:
            bool: True if object is confirmed grasped, False if timeout or failed
        """
        if timeout is None:
            timeout = 30
            
        self.get_logger().info("Detecting if food pickup was successful...")
        
        consecutive_grasped_count = 0
        consecutive_not_grasped_count = 0
        start_time = time.time()
        
        while rclpy.ok() and (time.time() - start_time) < timeout:
            grasped = self.is_object_grasped()
            
            if grasped:
                consecutive_grasped_count += 1
                consecutive_not_grasped_count = 0
                # self.get_logger().info(f"Object grasped: True (count: {consecutive_grasped_count})")
                
                if consecutive_grasped_count >= self.consecutive_checks:
                    self.get_logger().info("Detected successful pickup")
                    return True
            else:
                consecutive_not_grasped_count += 1
                consecutive_grasped_count = 0
                # self.get_logger().info(f"Object grasped: False (count: {consecutive_not_grasped_count})")
                
                if consecutive_not_grasped_count >= self.consecutive_checks * 2:
                    self.get_logger().warn("Detected unsuccessful pickup")
                    return False
            
            time.sleep(self.check_interval)
        
        self.get_logger().warn("Timeout waiting for grasp confirmation")
        return False
    
    def check_object_removed(self, timeout=None):
        """
        Continuously check if object has been removed from the fixed pickup pixel coordinates
        
        Args:
            timeout: Maximum time to wait for removal confirmation (uses config default if None)
            
        Returns:
            bool: True if object is confirmed removed, False if timeout
        """
        if timeout is None:
            timeout = 60
            
        self.get_logger().info("Checking if food has been removed...")
        
        consecutive_removed_count = 0
        start_time = time.time()
        
        while rclpy.ok() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec = 30)
            removed = self.is_object_removed()
            
            if removed:
                consecutive_removed_count += 1
                self.get_logger().info(f"Object removed: True (count: {consecutive_removed_count})")
                
                if consecutive_removed_count >= self.consecutive_checks:
                    self.get_logger().info("Food has been removed")
                    return True
            else:
                consecutive_removed_count = 0
                self.get_logger().info("Food is still in gripper...")
            
            time.sleep(self.check_interval)
        
        self.get_logger().info("Timeout waiting for removal - assuming user finished")
        return True 