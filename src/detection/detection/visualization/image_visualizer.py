#!/usr/bin/env python3

import cv2
import numpy as np
import os
import time
from sensor_msgs.msg import CompressedImage


class ImageVisualizer:
    def __init__(self, node=None):
        """
        Initialize image visualizer
        
        Args:
            node: ROS node instance (optional, for publishing compressed images)
        """
        self.node = node
        
        # Create save directories
        self.save_dir = os.path.expanduser('~/raf-live/pics/gemini_detection')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.dinox_save_dir = os.path.expanduser('~/raf-live/pics/dinox_detection')
        os.makedirs(self.dinox_save_dir, exist_ok=True)
        
        # Setup compressed image publisher if node provided
        if self.node:
            self.processed_image_pub = self.node.create_publisher(
                CompressedImage, '/processed_image', 10)
    
    def save_debug_image(self, frame, detection_input, detection_type="gemini"):
        """
        Save debug image with detected coordinates or bounding box
        
        Args:
            frame: Input image frame
            detection_input: Either (x, y) point for gemini or [x1, y1, x2, y2] bbox for dinox
            detection_type: "gemini" or "dinox"
        """
        try:
            height, width = frame.shape[:2]
            debug_frame = frame.copy()
            
            if detection_type == "dinox":
                # Draw bounding box for DINOX detection
                bbox = detection_input
                cv2.rectangle(debug_frame, (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
                
                # Save to dinox_detection directory
                timestamp = int(time.time())
                debug_path = os.path.join(self.dinox_save_dir, f"dinox_detection_{timestamp}.jpg")
            else:
                # Draw point for Gemini detection
                point = detection_input
                cv2.circle(debug_frame, point, 10, (0, 255, 0), -1)
                cv2.putText(debug_frame, f"({point[0]}, {point[1]})", 
                           (point[0] + 15, point[1] - 15), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                
                # Save to gemini_detection directory
                timestamp = int(time.time())
                debug_path = os.path.join(self.save_dir, f"gemini_detection_{timestamp}.jpg")
            
            cv2.imwrite(debug_path, debug_frame)
            if self.node:
                self.node.get_logger().info(f"Debug image saved to {debug_path}")
                
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Failed to save debug image: {e}")
    
    def draw_grasp_visualization(self, image, centroid, width_p1=None, width_p2=None, 
                                food_angle=None, current_item="", single_bite=True):
        """
        Draw grasp visualization on the image exactly like perception node
        
        Args:
            image: Input image
            centroid: (x, y) centroid coordinates
            width_p1: First width point coordinates (optional)
            width_p2: Second width point coordinates (optional)
            food_angle: Food angle in degrees (optional)
            current_item: Name of current food item
            single_bite: Whether this is a single bite item
            
        Returns:
            Visualization image with overlays
        """
        vis_image = image.copy()
        
        # Draw centroid
        cv2.circle(vis_image, centroid, 5, (255, 0, 0), -1)
        
        # Draw width points if provided
        if width_p1 is not None and width_p2 is not None:
            # Draw width points as circles
            cv2.circle(vis_image, tuple(width_p1), 3, (255, 255, 0), -1)  # Cyan circles
            cv2.circle(vis_image, tuple(width_p2), 3, (255, 255, 0), -1)
            
            # Draw line connecting width points
            cv2.line(vis_image, tuple(width_p1), tuple(width_p2), (255, 255, 0), 2)  # Cyan line
            
        # Add text information
        # info_text = [
        #     f"Detected Item: {current_item}",
        #     f"Food Angle: {food_angle:.2f} deg" if food_angle is not None else "Food Angle: N/A",
        #     f"Single Bite: {single_bite}"
        # ]
        
        # for i, text in enumerate(info_text):
        #     cv2.putText(vis_image, text, (10, 40 + i * 40), 
        #                 cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        #     cv2.putText(vis_image, text, (10, 40 + i * 40), 
        #                 cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            
        return vis_image
    
    def create_tracking_visualization(self, frame, mask, centroid, width_p1=None, width_p2=None,
                                    food_angle=None, current_item="", single_bite=True):
        """
        Create complete tracking visualization with mask overlay
        
        Args:
            frame: Input image frame
            mask: Segmentation mask
            centroid: (x, y) centroid coordinates
            width_p1: First width point coordinates (optional)
            width_p2: Second width point coordinates (optional) 
            food_angle: Food angle in degrees (optional)
            current_item: Name of current food item
            single_bite: Whether this is a single bite item
            
        Returns:
            Complete visualization image
        """
        # Create visualization with grasp points
        vis_image = self.draw_grasp_visualization(
            frame, centroid, width_p1, width_p2, food_angle, current_item, single_bite)
        
        # Apply mask overlay like in perception node
        height, width = frame.shape[:2]
        
        # Ensure mask is in the right format
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
            
        # Convert to 3-channel for overlay
        if len(mask.shape) == 2:
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            mask_3d = mask
            
        # Apply weighted overlay
        vis_image = cv2.addWeighted(vis_image, 1, mask_3d, 0.5, 0)
        
        return vis_image
    
    def publish_segmented_image(self, vis_image):
        """
        Publish segmented image as compressed image message
        
        Args:
            vis_image: Visualization image to publish
        """
        if not self.node or not hasattr(self, 'processed_image_pub'):
            return
            
        try:
            msg = CompressedImage()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', vis_image)[1]).tobytes()
            self.processed_image_pub.publish(msg)
        except Exception as e:
            self.node.get_logger().error(f"Failed to publish processed image: {e}")
    
    def show_image(self, image, window_name='Food Detection'):
        """
        Display image using OpenCV (non-blocking)
        
        Args:
            image: Image to display
            window_name: Name of the display window
        """
        cv2.imshow(window_name, image)
        cv2.waitKey(1)  # Non-blocking