#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float64, String
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
        # self.fx = 615.0
        # self.fy = 615.0
        # self.cx = 320.0
        # self.cy = 240.0

        self.tracking = False # state
        self.tracking_ready_sent = False # flag to ensure tracking_ready is only sent once
        # how far the robot should be above the food item before pickup sequence starts
        self.distance_from_target = self.config['feeding']['dist_from_food']
        self.timer = None # timer to process frames

        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 1)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 1)
        self.start_detection_sub = self.create_subscription(
            Bool, '/start_food_detection', self.start_food_detection_callback, 10)
        self.stop_detection_sub = self.create_subscription(
            Bool, '/stop_food_detection', self.stop_detection_callback, 10)
        
        # Publishers
        self.position_vector_pub = self.create_publisher(Vector3, '/position_vector', 1)
        self.food_angle_pub = self.create_publisher(Float64, '/food_angle', 1)
        self.grip_value_pub = self.create_publisher(Float64, '/grip_value', 1)
        self.food_height_pub = self.create_publisher(Float64, '/food_height', 1)
        self.single_bite_pub = self.create_publisher(Bool, '/single_bite', 1)

        self.segmented_image_pub = self.create_publisher(CompressedImage, '/segmented_image', 10)
        self.tracking_ready_pub = self.create_publisher(Bool, '/food_tracking_ready', 1)
        self.currently_serving_pub = self.create_publisher(String, '/currently_serving', 10)
        self.command_queue_pub = self.create_publisher(String, '/command_queue', 10)
        
        self.tracking_lost_pub = self.create_publisher(Bool, '/tracking_lost', 1)

        self.food_depth_pub = self.create_publisher(Float64, '/food_depth', 1)

        self.estop_publisher = self.create_publisher(Bool, '/my_gen3/estop', 10)

        self.get_logger().info('Food Detection Node initialized')

    def _load_prompt(self):
        """Load detection prompt and inject available items"""
        if self.detection_model == 'dinox':
            prompt_file = os.path.expanduser('~/raf-live/src/detection/prompts/gpt_identification.txt')
        else:
            prompt_file = os.path.expanduser('~/raf-live/src/detection/prompts/gemini_identification.txt')
        try:
            with open(prompt_file, 'r') as f:
                prompt = f.read().strip()

            # Inject available items from config
            available_items = self.config['feeding']['available_items']
            items_str = ', '.join(available_items)
            prompt = prompt.replace('{AVAILABLE_ITEMS}', items_str)

            return prompt
        except:
            self.get_logger().error(f"Prompt file not found: {prompt_file}, are you in the right directory?")

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

    def stop_detection_callback(self, msg):
        if msg.data:
            self.get_logger().info("Received stop detection command - resetting to fresh state")
            self.tracking = False
            self.tracking_ready_sent = False  # Reset flag for next detection cycle

            # Complete SAM2 tracker reset (like old version)
            self.sam2_tracker.reset_tracking()

            # Complete timer cleanup (like old version)
            if self.timer:
                self.timer.cancel()
                self.timer = None

            self.tracking_ready_pub.publish(Bool(data=False))

    def start_food_detection_callback(self, msg):
        if msg.data:
            self.get_logger().info("Starting food detection...")
            self.tracking = False
            self.tracking_ready_sent = False

            # run ChatGPT and DINOX on latest color image for initial detection
            if self.latest_color_image is None:
                self.get_logger().error("No color image received yet")
                return
            frame = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")

            # get the bounding box around the desired food item
            detection_result = self.detector.detect_food(frame)
            if detection_result is None:
                self.get_logger().error("GPT + DINOX stack failed to generate bounding box")
                self.tracking_lost_pub.publish(Bool(data=True))
                return
            self.image_viz.save_debug_image(frame, detection_result, self.detection_model)

            # initialize tracking by giving bounding box to SAM2
            self.tracking = self.sam2_tracker.initialize_tracking(
                frame, detection_result, self.detection_model)
            if not self.tracking:
                self.get_logger().error("Failed to initialize tracking with SAM2")
                self.tracking_lost_pub.publish(Bool(data=True))
                return
            
            # get current item data
            current_item = self.detector.get_current_item()
            single_bite = self.detector.is_single_bite()
            self.single_bite_pub.publish(Bool(data=single_bite))
            self.get_logger().info(f"Tracking started for: {current_item} (single_bite: {single_bite})")
            # Don't publish tracking_ready yet - wait for first position vector
            
            # now, start timer and begin processing frames in real-time
            if self.timer:
                self.timer.cancel()
            self.timer = self.create_timer(0.1, lambda: self._update_tracking_and_publish(current_item, single_bite)) # 10 Hz

    def _update_tracking_and_publish(self, current_item, single_bite):
        """Publishes real-time position vectors and grasp data from SAM2 segment"""
        if not self.tracking or self.latest_color_image is None:
            return


        try:
            # Get fresh frame every time
            frame = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")

            # Update tracking
            mask_2d, _ = self.sam2_tracker.update_tracking(frame)

            if mask_2d is None:
                # Check if this is a permanent tracking loss
                if self.sam2_tracker.is_tracking_lost():
                    self.get_logger().error("SAM2 tracking permanently lost!")
                    self.tracking_lost_pub.publish(Bool(data=True))
                    # Do complete reset like stop_detection_callback
                    self.tracking = False
                    self.tracking_ready_sent = False
                    self.sam2_tracker.reset_tracking()
                    if self.timer:
                        self.timer.cancel()
                        self.timer = None
                    self.tracking_ready_pub.publish(Bool(data=False))
                    return


            if self.latest_depth_image is None:
                self.get_logger().error("No depth image received")
                return
               
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')

            # get all grasp data
            grasp_info = self.grasp_analyzer.analyze_grasp(mask_2d, depth_image, single_bite, current_item)
            if not grasp_info['success']:
                self.get_logger().warn("Grasp analysis failed")
                # Grasp analysis failed - no position vector published
                return
            
            centroid = grasp_info['centroid']
            
            # publish grip value and height
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

                # Publish tracking_ready only once after first successful position vector
                if not self.tracking_ready_sent:
                    self.tracking_ready_pub.publish(Bool(data=True))
                    self.tracking_ready_sent = True
                    self.get_logger().info("First position vector published - tracking ready!")
            else:
                self.get_logger().warn("Failed to publish position vector!")
                pass

            # create visualization
            vis_image = self.image_viz.create_tracking_visualization(
                frame, mask_2d, centroid, grasp_info['width_p1'], grasp_info['width_p2'],
                grasp_info['food_angle'], current_item, single_bite)
            self.image_viz.publish_segmented_image(vis_image)
        
        except Exception as e:
            self.get_logger().error(f"Error in tracking update: {e}")
            self.tracking = False
            return
        
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
            
    
