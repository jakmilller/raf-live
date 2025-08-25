#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import yaml
import time
import tf2_ros
from dotenv import load_dotenv

# Import the detection and tracking modules
from visualization import ImageVisualizer
from tracking import SAM2Tracker, GraspAnalyzer, CoordinateTransforms
from detectors import GeminiDetector, DinoxDetector


class FoodDetectionNode(Node):
    def __init__(self):
        super().__init__('food_detection_node')
        
        # Load environment and config
        load_dotenv(os.path.expanduser('~/raf-live/.env'))
        config_path = os.path.expanduser('~/raf-live/config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # ROS setup
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        self.camera_info = None
        
        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Initialize detector
        self.detection_model = self.config['feeding']['detection_model']
        self.prompt = self._load_prompt()
        
        if self.detection_model == 'dinox':
            self.detector = DinoxDetector(
                node=self,
                dinox_api_key=os.getenv('dinox_api_key'),
                openai_api_key=os.getenv('openai_api_key'),
                prompt=self.prompt
            )
        else:
            self.detector = GeminiDetector(
                node=self,
                prompt=self.prompt
            )
        
        # Initialize tracking and analysis
        self.sam2_tracker = SAM2Tracker(self)
        self.coordinate_transforms = CoordinateTransforms(self)
        self.grasp_analyzer = GraspAnalyzer(self)
        self.image_viz = ImageVisualizer(self)
        
        # ROS setup - camera parameters
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        
        # State
        self.detection_active = False
        self.tracking_initialized = False
        self.distance_from_target = self.config['feeding']['dist_from_food']
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        # Control subscriber - simplified interface
        self.start_detection_sub = self.create_subscription(
            Bool, '/start_food_detection', self.start_detection_callback, 10)
        
        # Publishers - only what orchestrator needs
        self.position_vector_pub = self.create_publisher(Vector3, '/position_vector', 1)
        self.food_angle_pub = self.create_publisher(Float64, '/food_angle', 1)
        self.grip_value_pub = self.create_publisher(Float64, '/grip_value', 1)
        self.food_height_pub = self.create_publisher(Float64, '/food_height', 1)
        self.segmented_image_pub = self.create_publisher(CompressedImage, '/segmented_image', 10)
        self.detection_ready_pub = self.create_publisher(Bool, '/food_detection_ready', 1)
        
        # Processing timer (only active when detection is on)
        self.timer = None
        
        self.get_logger().info('Simple Food Detection Node initialized')
    
    def _load_prompt(self):
        """Load detection prompt"""
        if self.detection_model == 'dinox':
            prompt_file = os.path.expanduser('~/raf-deploy/src/perception/prompts/identification.txt')
        else:
            prompt_file = os.path.expanduser('~/raf-deploy/src/perception/prompts/gemini_identification.txt')
        
        try:
            with open(prompt_file, 'r') as f:
                return f.read().strip()
        except:
            return "Identify food items in this image."
    
    def color_callback(self, msg):
        self.latest_color_image = msg
    
    def depth_callback(self, msg):
        self.latest_depth_image = msg
    
    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
    
    def start_detection_callback(self, msg):
        """Simple on/off control"""
        if msg.data and not self.detection_active:
            self.get_logger().info("Starting food detection")
            self.detection_active = True
            self.tracking_initialized = False
            self.sam2_tracker.reset_tracking()
            self.grasp_analyzer.reset_food_height_calculation()
            
            # Start processing timer
            if self.timer:
                self.timer.cancel()
            self.timer = self.create_timer(0.1, self.process_frame)
            
        elif not msg.data and self.detection_active:
            self.get_logger().info("Stopping food detection")
            self.detection_active = False
            self.tracking_initialized = False
            self.sam2_tracker.reset_tracking()
            
            # Stop processing timer
            if self.timer:
                self.timer.cancel()
                self.timer = None
            
            # Publish zero vector to stop servoing
            self._publish_zero_vector()
            
            # Signal detection not ready
            self.detection_ready_pub.publish(Bool(data=False))
            
            # Clear segmented image
            self._clear_segmented_image()
    
    def process_frame(self):
        """Main processing loop - simplified"""
        if not self.detection_active or self.latest_color_image is None:
            return
        
        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            
            if not self.sam2_tracker.is_tracking_active():
                # Detection phase - still looking for food
                detection_result = self.detector.detect_food(frame)
                if detection_result is not None:
                    # Save debug image
                    self.image_viz.save_debug_image(frame, detection_result, self.detection_model)
                    
                    # Initialize tracking
                    success = self.sam2_tracker.initialize_tracking(frame, detection_result, self.detection_model)
                    if success:
                        # Get current item info and immediately start publishing
                        current_item = self.detector.get_current_item()
                        single_bite = self.detector.is_single_bite()
                        self.get_logger().info(f"Tracking started for: {current_item} (single_bite: {single_bite})")
                        
                        # Mark as ready and start publishing vectors
                        self.tracking_initialized = True
                        self.detection_ready_pub.publish(Bool(data=True))
                        self.get_logger().info("Food detection ready - tracking initialized!")
                        
                        self._update_tracking_and_publish(frame, current_item, single_bite)
                else:
                    # Still looking for food - don't publish anything yet
                    pass
            else:
                # Continue tracking - we're ready and publishing vectors
                current_item = self.detector.get_current_item()
                single_bite = self.detector.is_single_bite()
                self._update_tracking_and_publish(frame, current_item, single_bite)
                
        except Exception as e:
            self.get_logger().error(f'Error in process_frame: {e}')
            if self.tracking_initialized:
                self._publish_zero_vector()
    
    def _update_tracking_and_publish(self, frame, current_item, single_bite):
        """Update tracking and publish results"""
        try:
            # Update tracking
            mask_2d, _ = self.sam2_tracker.update_tracking(frame)
            
            if mask_2d is None:
                if self.tracking_initialized:
                    self._publish_zero_vector()
                return
            
            # Get depth image
            if self.latest_depth_image is None:
                self.get_logger().error("No depth image available")
                if self.tracking_initialized:
                    self._publish_zero_vector()
                return
            
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
            
            # Analyze grasp
            grasp_info = self.grasp_analyzer.analyze_grasp(mask_2d, depth_image, single_bite, current_item)
            
            if not grasp_info['success']:
                self.get_logger().warn("Grasp analysis failed")
                if self.tracking_initialized:
                    self._publish_zero_vector()
                return
            
            centroid = grasp_info['centroid']
            
            # Publish grip value and food height (only once per detection cycle)
            if grasp_info['grip_value'] is not None:
                self.grip_value_pub.publish(Float64(data=grasp_info['grip_value']))
            
            if grasp_info['food_height'] is not None:
                self.food_height_pub.publish(Float64(data=grasp_info['food_height']))
            
            # Publish food angle
            if grasp_info['food_angle'] is not None:
                self.food_angle_pub.publish(Float64(data=grasp_info['food_angle']))
            
            # Calculate and publish position vector
            position_vector = self.coordinate_transforms.calculate_position_vector_from_mask(
                centroid[0], centroid[1], mask_2d, self.distance_from_target)
            
            if position_vector is not None:
                self.position_vector_pub.publish(position_vector)
            else:
                if self.tracking_initialized:
                    self._publish_zero_vector()
            
            # Create and publish visualization
            vis_image = self.image_viz.create_tracking_visualization(
                frame, mask_2d, centroid, grasp_info['width_p1'], grasp_info['width_p2'],
                grasp_info['food_angle'], current_item, single_bite)
            self.image_viz.publish_segmented_image(vis_image)
            
        except Exception as e:
            self.get_logger().error(f"Error in tracking update: {e}")
            if self.tracking_initialized:
                self._publish_zero_vector()
    
    def _publish_zero_vector(self):
        """Publish zero position vector"""
        zero_vector = Vector3(x=0.0, y=0.0, z=0.0)
        self.position_vector_pub.publish(zero_vector)
    
    def _clear_segmented_image(self):
        """Clear the segmented image display"""
        try:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = b''
            self.segmented_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to clear segmented image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FoodDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()