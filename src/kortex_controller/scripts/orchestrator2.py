#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Point, Vector3, Twist
from std_msgs.msg import Float64, String, Bool
from raf_interfaces.srv import ProcessImage, SetTwist
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import cv2
import copy
import os
import asyncio
import sys
import yaml
import time
import pygame
import numpy as np
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs

# MediaPipe imports for face detection
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# import controller/autonomous checks
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from robot_controller_ros2 import KinovaRobotControllerROS2
from autonomous_checker import AutonomousChecker

class RAFOrchestrator(Node):
    def __init__(self):
        super().__init__('raf_orchestrator')
        
        config_path = os.path.expanduser('~/raf-deploy/config.yaml')
        
        # load config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.robot_controller = KinovaRobotControllerROS2(config_path)
        
        # Face detection configuration
        self.face_detection_enabled = self.config['feeding']['face_detection']['enabled']
        self.last_mouth_position = None
        if self.face_detection_enabled:
            self.setup_face_detection()
            self.gain_planar = self.config['feeding']['face_detection']['gain_planar']
            self.gain_depth = self.config['feeding']['face_detection']['gain_depth']
            self.target_distance = self.config['feeding']['face_detection'].get('target_distance', 0.05)
            self.visual_servo_timeout = self.config['feeding']['face_detection'].get('timeout', 30.0)
        
        # checks (whether food has been grasped/removed) can either be manual (through terminal) or autonomous (through depth camera)
        # this initializes autonomous checks if you set "autonomous" in the config
        self.mode = self.config['feeding']['mode']
        if self.mode == "autonomous":
            self.autonomous_checker = AutonomousChecker(self.config)
            self.get_logger().info("Autonomous mode enabled")
        else:
            self.autonomous_checker = None
            self.get_logger().info("Manual mode enabled")

        # handle audio
        pygame.mixer.init()
        self.power_on = self.config['feeding']['power_on']
        self.power_off = self.config['feeding']['power_off']
        self.snap = self.config['feeding']['snap']
        self.play_sound(self.power_on)
        
        # creates service client for perception node
        self.perception_client = self.create_client(ProcessImage, 'process_image_pipeline')
        
        # these come from the perception node
        self.food_pose = None
        self.grip_value = None
        self.drink_requested = False
        self.drinking_complete = False
        
        # subscribe to perception topics
        self.food_pose_sub = self.create_subscription(
            PoseStamped, '/food_pose_base_link', self.food_pose_callback, 10)
        self.grip_value_sub = self.create_subscription(
            Float64, '/grip_value', self.grip_value_callback, 10)
        self.food_height_sub = self.create_subscription(
            Float64, '/food_height', self.food_height_callback, 10
        )
        self.single_bite_sub = self.create_subscription(
            Bool, '/single_bite', self.single_bite_callback, 10
        )
        self.drink_request_sub = self.create_subscription(
            Bool, '/drink_request', self.drink_request_callback, 10
        )
        self.drink_complete_sub = self.create_subscription(
            Bool, '/drinking_complete', self.drinking_complete_callback, 10
        )

        # create retry topic
        self.retry_pub = self.create_publisher(
            Bool, '/retry', 10
        )

        self.processing_locked_pub = self.create_publisher(
            Bool, '/processing_locked', 10
        )

        # create state publisher
        # valid states: Scanning, Food Acquisition, Bite Transfer, Drink Acquisition, Sipping
        self.robot_state_pub = self.create_publisher(
            String, '/robot_state', 10
        )
        self.segmented_image_pub = self.create_publisher(
            CompressedImage, '/segmented_image', 10
        )

        self.current_item_pub = self.create_publisher(
            String, '/currently_serving', 10
        )

        # wait for perception to start up
        while not self.perception_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for perception service...')
        
        # config parameters
        self.z_down_offset = self.config['feeding']['z_down_offset'] # food height
        self.z_up_offset = self.config['feeding']['z_up_offset'] # how high to bring up the food
        self.grip_close = self.config['feeding']['grip_close'] # how tight to grip the food
        
        self.get_logger().info(f'RAF Orchestrator ready in {self.mode} mode!')
        if self.face_detection_enabled:
            self.get_logger().info('Face detection enabled for visual servoing')
    
    def setup_face_detection(self):
        """Initialize face detection components"""
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        self.camera_info = None
        
        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Camera parameters
        self.fx = 615.0  # Default values, will be updated
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_frame_id = 'realsense_link'
        
        # Initialize MediaPipe FaceLandmarker
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(options)
        
        # Subscribe to camera topics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.color_sub = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.color_callback, 
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        # Create service client for SetTwist
        self.set_twist_client = self.create_client(SetTwist, '/my_gen3/set_twist')
        while not self.set_twist_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /my_gen3/set_twist service...')
    
    def camera_info_callback(self, msg):
        """Update camera intrinsic parameters"""
        if self.camera_info is None:  # Only update once
            self.camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(f"Updated camera parameters: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
    
    def color_callback(self, msg):
        if self.face_detection_enabled:
            self.latest_color_image = msg
    
    def depth_callback(self, msg):
        if self.face_detection_enabled:
            self.latest_depth_image = msg
    
    def food_pose_callback(self, msg):
        """Callback for food pose"""
        self.food_pose = msg
        self.get_logger().info("Received food pose")
    
    def grip_value_callback(self, msg):
        """Callback for grip value"""
        self.grip_value = msg.data
        self.get_logger().info(f"Received grip value: {self.grip_value}")

    def food_height_callback(self, msg):
        """Callback for food height"""
        self.food_height = msg.data
        self.get_logger().info(f"Received food height: {self.food_height} m")

    def single_bite_callback(self, msg):
        """Callback for single/multibite foods"""
        self.single_bite = msg.data
        self.get_logger().info(f"Received single bite: {self.single_bite}")

    def drink_request_callback(self, msg):
        """Callback for drink request"""
        self.drink_requested = msg.data
        self.get_logger().info(f"Drink request: {self.drink_requested}")

    def drinking_complete_callback(self, msg):
        self.drinking_complete = msg.data
        self.get_logger().info(f"Drinking complete: {self.drinking_complete}")
    
    def get_depth_at_pixel(self, x, y):
        """Get depth value at specific pixel coordinates"""
        if self.latest_depth_image is None:
            return 0.0
        
        try:
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
            height, width = depth_image.shape
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            depth_mm = depth_image[y, x]
            depth_m = depth_mm / 1000.0
            return depth_m
        except:
            return 0.0
    
    def get_finger_midpoint_in_end_effector_frame(self):
        """Get the finger midpoint position in the end effector frame"""
        try:
            # Get finger pad positions in the end-effector frame
            right_finger_transform = self.tf_buffer.lookup_transform(
                'end_effector_link', 'right_inner_finger_pad', rclpy.time.Time())
            left_finger_transform = self.tf_buffer.lookup_transform(
                'end_effector_link', 'left_inner_finger_pad', rclpy.time.Time())
                
            # Calculate midpoint of finger pads in the end-effector frame
            right_pos = right_finger_transform.transform.translation
            left_pos = left_finger_transform.transform.translation
            finger_midpoint = Point()
            finger_midpoint.x = (right_pos.x + left_pos.x) / 2.0
            finger_midpoint.y = (right_pos.y + left_pos.y) / 2.0
            finger_midpoint.z = (right_pos.z + left_pos.z) / 2.0
            # Add half the 2f 140 finger pad length to the z coordinate
            finger_midpoint.z += 0.019  # Adjust based on your gripper's finger length
            
            return finger_midpoint
            
        except Exception as e:
            self.get_logger().warn(f"Could not get finger midpoint: {e}", throttle_duration_sec=2.0)
            return None
    
    def detect_mouth_position(self):
        """Detect mouth position and return vector from finger to mouth in end effector frame"""
        if self.latest_color_image is None or self.latest_depth_image is None:
            return None
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect face landmarks
            detection_result = self.face_detector.detect(mp_image)

            #annotated_image = self.draw_mouth_landmarks_on_image(rgb_frame, detection_result)

            #cv2.imshow('Mouth Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            if not detection_result.face_landmarks:
                return None
            
            # Get mouth center from landmarks
            face_landmarks = detection_result.face_landmarks[0]
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            mouth_center_x = (upper_lip.x + lower_lip.x) / 2
            mouth_center_y = (upper_lip.y + lower_lip.y) / 2
            
            # Convert to pixel coordinates
            image_height, image_width = cv_image.shape[:2]
            center_x = int(mouth_center_x * image_width)
            center_y = int(mouth_center_y * image_height)
            
            # Get depth
            depth_value = self.get_depth_at_pixel(center_x, center_y)
            if depth_value <= 0:
                return None
            
            # Transform mouth position to end effector frame
            mouth_in_effector = self.transform_mouth_to_end_effector_frame(center_x, center_y, depth_value)
            if mouth_in_effector is None:
                return None
            
            # Get finger midpoint
            finger_midpoint = self.get_finger_midpoint_in_end_effector_frame()
            if finger_midpoint is None:
                return None
            
            # Calculate vector from finger to mouth
            vector = Vector3(
                x = mouth_in_effector.x - finger_midpoint.x,
                y = mouth_in_effector.y - finger_midpoint.y,
                z = mouth_in_effector.z - finger_midpoint.z
            )

            if self.last_mouth_position and abs(np.linalg.norm([vector.x, vector.y, vector.z]) - np.linalg.norm([self.last_mouth_position.x, self.last_mouth_position.y, self.last_mouth_position.z])) > 0.1:
                self.get_logger().warn("Detected large change in mouth position, ignoring")
                return None


            self.last_mouth_position = copy.deepcopy(vector)
            
            return vector
            
        except Exception as e:
            self.get_logger().error(f"Error in mouth detection: {str(e)}")
            return None
    
    def transform_mouth_to_end_effector_frame(self, mouth_pixel_x, mouth_pixel_y, depth):
        """Transform mouth position from camera frame to end effector frame"""
        # Project pixel to 3D point in camera frame
        mouth_z = depth
        mouth_x = (mouth_pixel_x - self.cx) * depth / self.fx
        mouth_y = (mouth_pixel_y - self.cy) * depth / self.fy

        # Create a stamped point for transformation
        mouth_stamped = tf2_geometry_msgs.PointStamped()
        mouth_stamped.header.stamp = self.get_clock().now().to_msg()
        mouth_stamped.header.frame_id = self.camera_frame_id
        mouth_stamped.point.x = mouth_x
        mouth_stamped.point.y = mouth_y
        mouth_stamped.point.z = mouth_z

        try:
            # Transform mouth point to the end-effector frame
            mouth_in_effector_frame = self.tf_buffer.transform(
                mouth_stamped, 'end_effector_link', timeout=rclpy.duration.Duration(seconds=0.5))
            return mouth_in_effector_frame.point
        except Exception as e:
            self.get_logger().warn(f"Could not transform mouth to end effector frame: {e}", throttle_duration_sec=2.0)
            return None
    
    async def spin_for_updates(self, stop_event):
        """Continuously spin to receive camera updates"""
        while not stop_event.is_set():
            rclpy.spin_once(self, timeout_sec=0.001)
            await asyncio.sleep(0.001)

    async def visual_servo_to_mouth(self):
        """Perform visual servoing to bring food to user's mouth"""
        self.get_logger().info("Starting visual servoing to mouth...")
        self.last_mouth_position = None
        
        # Create stop event for the spinning task
        stop_spinning = asyncio.Event()
        
        # Start background task to continuously spin for camera updates
        spin_task = asyncio.create_task(self.spin_for_updates(stop_spinning))
        
        # Give a moment for fresh images to arrive
        await asyncio.sleep(0.5)
        
        start_time = time.time()
        
        try:
            while rclpy.ok() and (time.time() - start_time) < self.visual_servo_timeout:
                # Detect mouth position and get vector
                position_vector = self.detect_mouth_position()
                
                if position_vector is None:
                    self.get_logger().warn("Could not detect mouth position, stopping")
                    await self.stop_robot()
                    await asyncio.sleep(0.1)
                    continue
                
                # Calculate distance to mouth
                distance = np.linalg.norm([position_vector.x, position_vector.y, position_vector.z])
                self.get_logger().info(f"Distance to mouth: {distance:.3f} m")
                
                # Check if we're close enough
                if distance < self.target_distance:
                    self.get_logger().info(f"Reached target distance ({self.target_distance} m), stopping")
                    await self.stop_robot()
                    return True
                
                # Create twist command
                twist = Twist()
                twist.linear.x = self.gain_planar * position_vector.x
                twist.linear.y = self.gain_planar * position_vector.y
                twist.linear.z = self.gain_depth * position_vector.z
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = 0.0
                
                # Send twist command
                request = SetTwist.Request()
                request.twist = twist
                request.timeout = 0.0  # Continuous mode
                
                try:
                    await self.set_twist_client.call_async(request)
                except Exception as e:
                    self.get_logger().error(f"Failed to send twist command: {e}")
                    await self.stop_robot()
                    return False
                
                # Small delay for control loop
                await asyncio.sleep(0.05)
            
            # Timeout reached
            self.get_logger().warn("Visual servoing timeout reached")
            await self.stop_robot()
            return False
            
        finally:
            # Stop the spinning task
            stop_spinning.set()
            await spin_task
    
    async def stop_robot(self):
        """Send zero twist to stop the robot"""
        request = SetTwist.Request()
        request.twist = Twist()  # Zero twist
        request.timeout = 0.1
        
        try:
            await self.set_twist_client.call_async(request)
        except Exception as e:
            self.get_logger().error(f"Failed to stop robot: {e}")
    
    def wait_for_keypress(self, message="Press any key to continue..."):
        """Wait for user keypress (manual mode only)"""
        if self.mode == "manual":
            self.get_logger().info(message)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def play_sound(self, file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for the sound to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    def clear_segmented_image(self):
        """Clear the segmented image display by publishing empty image"""
        try:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = b''  # Empty bytes - this will clear the image
            self.segmented_image_pub.publish(msg)
            self.get_logger().info("Published empty image to clear display")
        except Exception as e:
            self.get_logger().error(f"Failed to clear segmented image: {str(e)}")

    def validate_with_user(self, question):
        """Get user input for manual confirmations"""
        if self.mode == "manual":
            user_input = input(question + "(y/n): ")
            while user_input != "y" and user_input != "n":
                user_input = input(question + "(y/n): ")
            return user_input == "y"
        else:
            # in autonomous mode its assumed true except for grasp/removal
            return True

    def wait_for_grasp_confirmation(self):
        """Confirm if the food has been picked up (can be manual or autonomous confirmation)"""
        if self.mode == "autonomous":
            if self.autonomous_checker is None:
                self.get_logger().error("Autonomous checker not initialized!")
                return False
            
            self.get_logger().info("Checking if food was picked up autonomously...")
            # Spin the autonomous checker a few times to get latest data
            for _ in range(10):
                rclpy.spin_once(self.autonomous_checker, timeout_sec=0.1)
                time.sleep(0.1)
            
            return self.autonomous_checker.check_object_grasped()
        else:
            return self.validate_with_user('Did the robot pick up the food?')

    def wait_for_removal_confirmation(self):
        """Confirm when food has been removed from gripper in bite acquisition pose"""
        if self.mode == "autonomous":
            if self.autonomous_checker is None:
                self.get_logger().error("Autonomous checker not initialized!")
                return False
            
            return self.autonomous_checker.check_object_removed()
        else:
            self.validate_with_user('Has the food been removed?')
            return True

    def take_picture_and_get_pose(self):
        """Step 2: Take picture and get food pose"""
        self.get_logger().info("Step 2: Taking picture and processing...")

        # Reset perception data
        self.food_pose = None
        self.grip_value = None

        # Call perception service for food detection
        request = ProcessImage.Request()
        request.detection_type = "food"  # Specify food detection
        
        try:
            future = self.perception_client.call_async(request)
            
            # Wait for perception to complete
            while not future.done():
                time.sleep(0.1)
                rclpy.spin_once(self, timeout_sec=0)

            response = future.result()
            if not response or not response.success:
                self.get_logger().error(f"Perception failed: {response.message if response else 'No response'}")
                return False

            self.get_logger().info("Perception service completed successfully")

            # Wait for topics to update with timeout
            timeout_count = 0
            max_timeout = 50  # 5 seconds at 0.1s intervals

            while (self.food_pose is None or self.grip_value is None) and timeout_count < max_timeout:
                time.sleep(0.1)
                rclpy.spin_once(self, timeout_sec=0)
                timeout_count += 1

            # Check if we got the data
            if self.food_pose is None or self.grip_value is None:
                self.get_logger().error("Did not receive food pose or grip value within timeout!")
                self.get_logger().error(f"Food pose: {self.food_pose is not None}, Grip value: {self.grip_value is not None}")
                return False

            self.get_logger().info(f"Successfully got food pose and grip value: {self.grip_value}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Perception service call failed: {str(e)}")
            return False
    
    def take_picture_and_get_cup_pose(self):
        """Take picture and get cup pose"""
        self.get_logger().info("Taking picture and processing cup...")

        # Reset perception data
        self.food_pose = None
        self.grip_value = None

        # Call perception service for cup detection
        request = ProcessImage.Request()
        request.detection_type = "cup"
        
        try:
            future = self.perception_client.call_async(request)

            # Wait for perception to complete
            while not future.done():
                time.sleep(0.1)
                rclpy.spin_once(self, timeout_sec=0)

            response = future.result()
            if not response or not response.success:
                self.get_logger().error(f"Cup perception failed: {response.message if response else 'No response'}")
                return False

            self.get_logger().info("Cup perception service completed successfully")

            # Wait for topics to update with timeout
            timeout_count = 0
            max_timeout = 50  # 5 seconds at 0.1s intervals

            while (self.food_pose is None or self.grip_value is None) and timeout_count < max_timeout:
                time.sleep(0.1)
                rclpy.spin_once(self, timeout_sec=0)
                timeout_count += 1

            # Check if we got the data
            if self.food_pose is None or self.grip_value is None:
                self.get_logger().error("Did not receive cup pose or grip value within timeout!")
                return False

            self.get_logger().info(f"Successfully got cup pose and grip value: {self.grip_value}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Cup perception service call failed: {str(e)}")
            return False
    
    async def run_feeding_cycle(self):
        """Main feeding cycle with autonomous/manual mode support"""
        cycle_count = 1

        while rclpy.ok():
            self.get_logger().info(f"\n=== FEEDING CYCLE {cycle_count} ===")

            try:
                # Step 1: Move to overlook position
                self.get_logger().info("Step 1: Moving to overlook position...")
                if not await self.robot_controller.reset():
                    self.get_logger().error("Failed to reset robot!")
                    break
                if not await self.robot_controller.set_gripper(0.25):
                    self.get_logger().error("Failed to close gripper!")
                    break   
                
                # Step 2: Take picture and get pose
                time.sleep(1)

                self.robot_state_pub.publish(String(data='scanning'))
                if not self.take_picture_and_get_pose():
                    self.get_logger().error("Failed to get food pose!")
                    break
                
                # Step 3: Show picture and wait for keypress (manual mode only)
                if self.mode == "manual":
                    self.get_logger().info("Step 3: Image shown, waiting for keypress...")
                    self.wait_for_keypress("Image displayed. Press any key to continue to grasping...")

                # Step 4: Set gripper
                self.play_sound(self.snap)
                self.get_logger().info("Step 4: Setting gripper...")
                gripper_success = await self.robot_controller.set_gripper(self.grip_value)
                if not gripper_success:
                    self.get_logger().error("Failed to set gripper!")
                    break

                # Wait for gripper to open
                time.sleep(0.5)
                
                # Step 5: Move to food position
                
                self.robot_state_pub.publish(String(data='bite acquisition'))
                self.get_logger().info("Step 5: Moving to food position...")
                move_success = await self.robot_controller.move_to_pose(self.food_pose.pose)
                if not move_success:
                    self.get_logger().error("Failed to move to food position!")
                    break

                # Step 6: Move down to grasp
                self.get_logger().info("Step 6: Moving down to grasp...")
                grasp_pose = copy.deepcopy(self.food_pose.pose)

                # you want to go down the food's height to grab the food item
                grasp_pose.position.z -= (self.food_height + 0.003)

                if not await self.robot_controller.move_to_pose(grasp_pose):
                    self.get_logger().error("Failed to move down!")
                    break
                
                # Step 7: Close gripper
                self.get_logger().info("Step 7: Closing gripper...")
                close_value = min(1.0, self.grip_value + self.grip_close)
                if not await self.robot_controller.set_gripper(close_value):
                    self.get_logger().error("Failed to close gripper!")
                    break
                
                # Step 8: Move up with food
                self.get_logger().info("Step 8: Moving up with food...")
                self.robot_state_pub.publish(String(data='bite transfer'))

                bring_up_pose = copy.deepcopy(self.food_pose.pose)
                bring_up_pose.position.z += self.z_up_offset
                if not await self.robot_controller.move_to_pose(bring_up_pose):
                    self.get_logger().error("Failed to move up!")
                    break
                
                # Step 8.5: Check if food was picked up (autonomous or manual)
                self.get_logger().info("Step 8.5: Checking if food was picked up...")
                if not self.wait_for_grasp_confirmation():
                    self.retry_pub.publish(Bool(data=True))
                    self.get_logger().warn("Food pickup not confirmed. Returning to overlook...")
                    continue

                # Step 9: Move to bite transfer position (either preset or visual servoing)
                if self.face_detection_enabled:
                    # Use visual servoing
                    self.get_logger().info("Step 9: Using visual servoing to move to mouth...")
                    self.robot_state_pub.publish(String(data='servoing'))
                    
                    # Move to intermediate position first
                    self.get_logger().info("Moving to intermediate position...")
                    if not await self.robot_controller.move_to_intermediate():
                        self.get_logger().error("Failed to move to intermediate!")
                        break
                    
                    # Perform visual servoing
                    success = await self.visual_servo_to_mouth()
                    if not success:
                        self.get_logger().error("Visual servoing failed!")
                        # Fall back to preset position
                        self.get_logger().info("Falling back to preset bite transfer position...")
                        if self.single_bite:
                            await self.robot_controller.move_to_bite_transfer()
                        else:
                            await self.robot_controller.move_to_multi_bite_transfer()
                else:
                    # Use preset positions
                    self.get_logger().info("Moving to intermediate position...")
                    if not await self.robot_controller.move_to_intermediate():
                        self.get_logger().error("Failed to move to intermediate!")
                        break
                    
                    if self.single_bite: 
                        self.get_logger().info("Step 9: Moving to single bite transfer position...")
                        if not await self.robot_controller.move_to_bite_transfer():
                            self.get_logger().error("Failed to move to bite transfer!")
                            break
                    else:
                        self.get_logger().info("Step 9: Moving to multi-bite transfer position...")
                        if not await self.robot_controller.move_to_multi_bite_transfer():
                            self.get_logger().error("Failed to move to multi-bite transfer!")
                            break
                
                self.play_sound(self.snap)
                # Step 10: Check if food was removed (autonomous or manual)
                self.robot_state_pub.publish(String(data='bite transfer'))
                self.get_logger().info("Step 10: Checking if food was removed...")
                self.wait_for_removal_confirmation()
                self.clear_segmented_image()
                self.current_item_pub.publish(String(data=' '))

                self.processing_locked_pub.publish(Bool(data=False))

                # ask if they want a drink (manual mode)
                if self.mode == "manual" and self.validate_with_user('Would you like a drink?'):
                    self.get_logger().info("User requested drink. Starting drinking cycle...")
                    await self.run_drinking_cycle()
                    # After drinking, return to feeding choice
                    continue
                
                # Step 11: Return to overlook for next cycle
                self.get_logger().info("Returning to overlook for next cycle...")
                self.retry_pub.publish(Bool(data=False))
                cycle_count += 1

            except KeyboardInterrupt:
                self.get_logger().info("Feeding cycle interrupted by user")
                break
            except Exception as e:
                self.get_logger().error(f"Error in feeding cycle: {str(e)}")
                break
            
        # Final return to overlook
        self.get_logger().info("Feeding complete. Returning to overlook position...")
        await self.robot_controller.reset()

    async def run_drinking_cycle(self):
        """Drinking cycle"""
        self.get_logger().info("=== STARTING DRINKING CYCLE ===")

        try:
            # Step 1: Move to cup scan position and open gripper
            self.get_logger().info("Step 1: Moving to cup scan position and opening gripper...")

            # Do both operations concurrently
            move_task = asyncio.create_task(self.robot_controller.move_to_cup_scan())
            gripper_task = asyncio.create_task(self.robot_controller.set_gripper(0.0))  # Fully open

            move_success, gripper_success = await asyncio.gather(move_task, gripper_task)

            if not move_success or not gripper_success:
                self.get_logger().error("Failed to move to cup scan position or open gripper!")
                return False

            # Step 2: Take picture and get cup pose
            time.sleep(1)
            if not self.take_picture_and_get_cup_pose():
                self.get_logger().error("Failed to get cup pose!")
                return False

            # Step 3: Show picture and wait for keypress (manual mode only)
            if self.mode == "manual":
                self.get_logger().info("Step 3: Cup detected, waiting for keypress...")
                self.wait_for_keypress("Cup displayed. Press any key to continue to grasping...")

            # Step 4: Move to cup position
            self.get_logger().info("Step 4: Moving to cup position...")
            if not await self.robot_controller.move_to_pose(self.food_pose.pose):  # Reuse food_pose topic
                self.get_logger().error("Failed to move to cup position!")
                return False

            # Step 5: Close gripper (use the grip value from perception)
            self.get_logger().info("Step 5: Closing gripper on cup...")
            if not await self.robot_controller.set_gripper(self.grip_value):
                self.get_logger().error("Failed to close gripper on cup!")
                return False

            time.sleep(1)
            # Step 6: Move upwards
            self.get_logger().info("Step 6: Moving up with cup...")
            bring_up_pose = copy.deepcopy(self.food_pose.pose)
            bring_up_pose.position.z += self.z_up_offset
            if not await self.robot_controller.move_to_pose(bring_up_pose):
                self.get_logger().error("Failed to move up with cup!")
                return False

            # Step 7: Move to sip position
            self.get_logger().info("Step 7: Moving to sip position...")
            self.robot_state_pub.publish(String(data='sipping'))

            if not await self.robot_controller.move_to_sip():
                self.get_logger().error("Failed to move to sip position!")
                return False

            # Step 8: Wait for user to finish drinking
            if self.mode == "manual":
                self.validate_with_user('Have you finished drinking? Press y when done.')
            else:
                # In autonomous mode, wait for drinking_complete signal
                self.get_logger().info("Autonomous mode: Waiting for drinking complete signal...")
                self.drinking_complete = False  # Reset the flag
                
                # Wait for the drinking_complete signal with timeout
                timeout_counter = 0
                max_timeout = 600  # 60 seconds timeout (at 0.1s intervals)
                
                while not self.drinking_complete and timeout_counter < max_timeout:
                    await asyncio.sleep(0.1)
                    rclpy.spin_once(self, timeout_sec=0)
                    timeout_counter += 1
                
                if timeout_counter >= max_timeout:
                    self.get_logger().warn("Timeout waiting for drinking complete signal")
                    # Continue anyway - don't fail the cycle
                else:
                    self.get_logger().info("Received drinking complete signal!")
    
            # Step 9: Return cup to original position
            self.get_logger().info("Step 9: Returning cup...")
            if not await self.robot_controller.move_to_pose(bring_up_pose):
                self.get_logger().error("Failed to move cup back!")
                return False

            if not await self.robot_controller.move_to_pose(self.food_pose.pose):
                self.get_logger().error("Failed to lower cup!")
                return False

            # Step 10: Open gripper to release cup
            if not await self.robot_controller.set_gripper(0.0):
                self.get_logger().error("Failed to open gripper!")
                return False

            # Step 11: Move up and away
            if not await self.robot_controller.move_to_pose(bring_up_pose):
                self.get_logger().error("Failed to move away from cup!")
                return False
            self.retry_pub.publish(Bool(data=False))
            
            self.get_logger().info("Drinking cycle completed successfully!")
            return True

        except Exception as e:
            self.get_logger().error(f"Error in drinking cycle: {str(e)}")
            return False 
        
    async def run_system(self):
        """Main system coordinator - handles choice between food and drink"""
        while rclpy.ok():
            try:
                if self.mode == "autonomous":
                    # In autonomous mode, just run feeding cycles continuously
                    if self.drink_requested:
                        self.get_logger().info("Processing drink request in autonomous mode")
                        self.drink_requested = False  # Reset flag
                        # Move to overlook first for drink
                        await self.robot_controller.reset()
                        await self.run_drinking_cycle()
                    else:
                        await self.run_feeding_cycle()
                else:
                    # Manual mode - ask user what they want
                    if self.validate_with_user('Would you like food? (n for drink)'):
                        await self.run_feeding_cycle()
                    else:
                        # Move to overlook first for drink (user chose drink directly)
                        self.get_logger().info("Moving to overlook position before drink...")
                        await self.robot_controller.reset()
                        await self.run_drinking_cycle()

            except KeyboardInterrupt:
                self.get_logger().info("System interrupted by user")
                break
            except Exception as e:
                self.get_logger().error(f"Error in system: {str(e)}")
                break

def main(args=None):
    print("Starting orchestrator...")
    rclpy.init(args=args)
    print("ROS2 initialized")
    
    try:
        print("Creating orchestrator node...")
        orchestrator = RAFOrchestrator()
        print("Orchestrator created, starting system...")
        
        # Start the system coordinator
        async def run():
            print("Inside async run function")
            await orchestrator.run_system()
        
        print("About to run async system...")
        # Run the async system
        asyncio.run(run())
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
        pass
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        print("Shutting down...")
        rclpy.shutdown()

if __name__ == '__main__':
    print("Script started")
    main()