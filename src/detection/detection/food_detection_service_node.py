#!/home/mcrr-lab/deploy-env/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, String, Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import tempfile
import os
import base64
import requests
import torch
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from sam2.build_sam import build_sam2_camera_predictor
import rclpy.duration
import yaml
import random
from sklearn.neighbors import NearestNeighbors
import math

# Import the service interface (reuse the same one)
from raf_interfaces.srv import StartFaceServoing
from visualization import RVizVisualizer, ImageVisualizer
from tracking import SAM2Tracker, GraspAnalyzer, CoordinateTransforms
from detectors import GeminiDetector, DinoxDetector

class FoodDetectionServiceNode(Node):
    def __init__(self):
        super().__init__('food_detection_service_node')
        
        # Load environment variables
        load_dotenv(os.path.expanduser('~/raf-live/.env'))
        self.dinox_api_key = os.getenv('dinox_api_key')
        self.openai_api_key = os.getenv('openai_api_key')
        genai.configure(api_key=os.getenv('google_api_key'))

        config_path = os.path.expanduser('~/raf-live/config.yaml')
        
        # load config variables
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.detection_model = self.config['feeding']['detection_model']
        self.get_logger().info(f"Using detection model: {self.detection_model}")
        self.prompt = self.load_prompt()

        # Initialize models based on config
        if self.detection_model == 'dinox':
            self.detector = DinoxDetector(
                node=self,
                dinox_api_key=self.dinox_api_key,
                openai_api_key=self.openai_api_key,
                prompt=self.prompt
            )
        else:
            self.detector = GeminiDetector(
                node=self,
                prompt=self.prompt,
                current_food_target=None
            )

        self.current_item = ""
        self.single_bite = True

        self.food_height_pub = self.create_publisher(Float64, '/food_height', 1)
        
        # Initialize SAM2 and coordinate transforms
        self.sam2_tracker = SAM2Tracker(self)
        self.coordinate_transforms = CoordinateTransforms(self)
        
        # ROS setup
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Camera data
        self.latest_color_image = None
        self.latest_depth_image = None
        self.camera_info = None
        
        # Service state - SIMPLIFIED
        self.service_active = False
        self.current_gains = Vector3()
        self.distance_from_target = 0.015  # Default 1.5cm
        
        # Tracking state
        self.grasp_analyzer = GraspAnalyzer(self)
        
        # Detection and grasp state
        self.grip_value = None
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        # Subscribe to food acquired signal
        self.food_acquired_sub = self.create_subscription(
            Bool, '/food_acquired', self.food_acquired_callback, 10)
        
        # Publishers
        self.position_vector_pub = self.create_publisher(Vector3, '/position_vector', 1)
        self.food_angle_pub = self.create_publisher(Float64, '/food_angle', 10)
        self.grip_val_pub = self.create_publisher(Float64, '/grip_value', 10)
        self.segmented_image_pub = self.create_publisher(CompressedImage, '/segmented_image', 10)
        self.detection_ready_pub = self.create_publisher(Bool, '/detection_ready', 1)
        
        # Publishers for servoing node configuration
        self.twist_gains_pub = self.create_publisher(Vector3, '/twist_gains', 10)
        self.min_distance_pub = self.create_publisher(Float64, '/min_distance', 10)
        
        # RViz marker publishers
        self.rviz_viz = RVizVisualizer(self)
        self.image_viz = ImageVisualizer(self)
        
        # Processing timer
        self.timer = None
        
        # State
        self.current_food_target = None
        
        # Create save directories
        self.save_dir = os.path.expanduser('~/raf-live/pics/gemini_detection')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.dinox_save_dir = os.path.expanduser('~/raf-live/pics/dinox_detection')
        os.makedirs(self.dinox_save_dir, exist_ok=True)
        
        # Create the service
        self.food_servoing_service = self.create_service(
            StartFaceServoing,  # Reusing the same service interface
            'start_food_servoing', 
            self.start_food_servoing_callback
        )
        self.get_logger().info('Food Detection Service Node initialized')
    
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
    
    def food_acquired_callback(self, msg):
        """Handle food acquired signal to end service"""
        if msg.data and self.service_active:
            self.get_logger().info('Received food acquired signal - ending food detection service')
            # publish 0 food angle so robot doesn't twist
            self.food_angle_pub.publish(Float64(data=0.0))
            self.service_active = False
            self.sam2_tracker.reset_tracking()
            if self.timer:
                self.timer.cancel()
                self.timer = None
            # Publish final zero vector
            self.publish_zero_vector()

    def start_food_servoing_callback(self, request, response):
        """Service callback to start food servoing"""
        self.get_logger().info('Food servoing service called')
        
        if self.service_active:
            response.success = False
            response.message = "Food servoing already active"
            return response
        
        # Store service parameters
        self.current_gains.x = request.gain_planar
        self.current_gains.y = request.gain_planar  # Same for both x and y
        self.current_gains.z = request.gain_depth
        self.distance_from_target = request.target_distance
        
        # Publish gains and min distance to servoing node
        self.twist_gains_pub.publish(self.current_gains)

        self.grasp_analyzer.reset_food_height_calculation()
        
        # Activate service
        self.service_active = True
        
        # Start processing timer
        if self.timer:
            self.timer.cancel()
        self.timer = self.create_timer(0.1, self.process_frame)
        
        self.get_logger().info(f'Started food servoing with gains: planar={request.gain_planar}, depth={request.gain_depth}, target_distance={request.target_distance}')
        
        response.success = True
        response.message = "Food servoing started successfully"
        return response

    def publish_zero_vector(self):
        """Publish zero position vector"""
        zero_vector = Vector3()
        zero_vector.x = 0.0
        zero_vector.y = 0.0
        zero_vector.z = 0.0
        self.position_vector_pub.publish(zero_vector)

    def load_prompt(self):
        """Load prompt from package prompts directory"""
        if self.detection_model == 'dinox':
            prompt_file = os.path.expanduser('~/raf-deploy/src/perception/prompts/identification.txt')
            self.get_logger().info(f'Loading ChatGPT identification prompt for DINOX stack')
        else:
            prompt_file = os.path.expanduser('~/raf-deploy/src/perception/prompts/gemini_identification.txt')
            self.get_logger().info(f'Loading Gemini detection prompt')

        try:
            with open(prompt_file, 'r') as f:
                loaded_prompt = f.read().strip()
            return loaded_prompt
        except:
            if self.detection_model == 'dinox':
                return "Identify all food items in this image. List them separated by commas."
            else:
                return "Identify the food item you see. Return coordinates if found."

    def _update_tracking_and_publish(self, frame):
        """Update tracking and publish results"""
        try:
            # Update tracking
            mask_2d, _ = self.sam2_tracker.update_tracking(frame)

            self.detection_ready_pub.publish(Bool(data=True))

            if mask_2d is None:
                self.publish_zero_vector()
                return

            # Get depth image
            if self.latest_depth_image is None:
                self.get_logger().error("No depth image available")
                return

            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')

            # Analyze grasp
            grasp_info = self.grasp_analyzer.analyze_grasp(mask_2d, depth_image, self.single_bite, self.current_item)

            centroid = grasp_info['centroid']

            if not grasp_info['success']:
                self.get_logger().warn("Grasp analysis failed")
                self.publish_zero_vector()
                return

            # Publish grip value if available
            if grasp_info['grip_value'] is not None:
                grip_msg = Float64()
                grip_msg.data = grasp_info['grip_value']
                self.grip_val_pub.publish(grip_msg)
                self.grip_value = grasp_info['grip_value']

            if grasp_info['food_height'] is not None:
                height_msg = Float64()
                height_msg.data = grasp_info['food_height']
                self.food_height_pub.publish(height_msg)

            # Publish food angle
            if grasp_info['food_angle'] is not None:
                self.food_angle_pub.publish(Float64(data=grasp_info['food_angle']))

            # Create visualization
            vis_image = self.image_viz.create_tracking_visualization(
                frame, mask_2d, centroid, grasp_info['width_p1'], grasp_info['width_p2'], 
                grasp_info['food_angle'], self.current_item, self.single_bite)

            # Publish segmented image
            self.image_viz.publish_segmented_image(vis_image)
            # self.image_viz.show_image(vis_image, 'Food Detection with Pose')

            # Convert to position vector
            position_vector = self.calculate_position_vector(centroid[0], centroid[1], mask_2d, self.distance_from_target)
            
            if position_vector is not None:
                # Always publish position vector - let orchestrator control when to use it
                self.position_vector_pub.publish(position_vector)
                
                # Publish RViz markers for visualization
                self.rviz_viz.publish_markers(centroid[0], centroid[1], position_vector)
            else:
                self.publish_zero_vector()

        except Exception as e:
            self.get_logger().error(f"Error in tracking update: {e}")
            self.publish_zero_vector()

    def get_finger_midpoint_in_end_effector_frame(self):
        """Get the finger midpoint position in the end effector frame"""
        return self.coordinate_transforms.get_finger_midpoint_in_end_effector_frame()
    
    def calculate_position_vector(self, pixel_x, pixel_y, segmentation_mask, distance):
        """Calculate position vector from finger midpoint to food centroid"""
        return self.coordinate_transforms.calculate_position_vector_from_mask(pixel_x, pixel_y, segmentation_mask, distance)
            
    def process_frame(self):
        """Main processing loop - simplified to just publish position vectors"""
        if not self.service_active or self.latest_color_image is None:
            return
        
        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            
            if not self.sam2_tracker.is_tracking_active():
                # Detection phase
                detection_result = self.detector.detect_food(frame)
                if detection_result is not None:
                    # Save debug image
                    self.image_viz.save_debug_image(frame, detection_result, self.detection_model)
    
                    # Update current item info from detector
                    self.current_item = self.detector.get_current_item()
                    self.single_bite = self.detector.is_single_bite()
    
                    # Initialize tracking
                    success = self.sam2_tracker.initialize_tracking(frame, detection_result, self.detection_model)
                    if success:
                        # Immediately start tracking on the same frame
                        self._update_tracking_and_publish(frame)
            else:
                # Continue tracking on subsequent frames
                self._update_tracking_and_publish(frame)
                    
        except Exception as e:
            self.get_logger().error(f'Error in process_frame: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FoodDetectionServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()