#!/usr/bin/env python3

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

        # Initialize models based on config
        if self.detection_model == 'dinox':
            self.setup_dinox()
            self.model = None
        else:
            self.model = genai.GenerativeModel('gemini-2.5-pro')

        # API headers for ChatGPT (only needed for DINOX stack)
        if self.detection_model == 'dinox':
            self.openai_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
        
        # Load prompt
        self.prompt = self.load_prompt()
        
        # Initialize SAM2
        self.setup_sam2()
        
        # ROS setup
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Camera data
        self.latest_color_image = None
        self.latest_depth_image = None
        self.camera_info = None
        
        # Service state
        self.service_active = False
        self.current_gains = Vector3()
        self.current_min_distance = 0.015  # Default 1.5cm
        
        # Tracking state
        self.tracking_active = False
        self.tracking_initialized = False
        
        # Detection and grasp state
        self.single_bite = True
        self.current_item = ""
        self.grip_value = None
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        # Subscribe to finished servoing signal
        self.finished_servoing_sub = self.create_subscription(
            Bool, '/finished_servoing', self.finished_servoing_callback, 10)
        
        # Publishers
        self.position_vector_pub = self.create_publisher(Vector3, '/position_vector', 10)
        self.food_angle_pub = self.create_publisher(Float64, '/food_angle', 10)
        self.grip_val_pub = self.create_publisher(Float64, '/grip_value', 10)
        self.segmented_image_pub = self.create_publisher(CompressedImage, '/segmented_image', 10)
        
        # Publishers for servoing node configuration
        self.twist_gains_pub = self.create_publisher(Vector3, '/twist_gains', 10)
        self.min_distance_pub = self.create_publisher(Float64, '/min_distance', 10)
        
        # RViz marker publishers
        self.target_point_pub = self.create_publisher(Marker, '/target_point', 10)
        self.gripper_point_pub = self.create_publisher(Marker, '/gripper_point', 10)
        self.vis_vector_pub = self.create_publisher(MarkerArray, '/vis_vector', 10)
        
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
    
    def setup_dinox(self):
        """Initialize DINOX client"""
        try:
            config = Config(self.dinox_api_key)
            self.dinox_client = Client(config)
            self.get_logger().info('DINOX client initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize DINOX: {e}')
    
    def setup_sam2(self):
        """Initialize SAM2 for real-time tracking"""
        try:
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            sam2_checkpoint = "/home/mcrr-lab/raf-live/SAM2_streaming/checkpoints/sam2.1/sam2.1_hiera_tiny.pt"
            model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
            self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
            self.tracking_initialized = False
            
            self.get_logger().info('SAM2 initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize SAM2: {e}')
    
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
    
    def finished_servoing_callback(self, msg):
        """Handle finished servoing signal to end service"""
        if msg.data and self.service_active:
            self.get_logger().info('Received finished servoing signal - ending food detection service')
            self.service_active = False
            self.tracking_active = False
            self.tracking_initialized = False
            if self.timer:
                self.timer.cancel()
                self.timer = None
            # Publish final zero vector to ensure servoing stops
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
        self.current_min_distance = request.target_distance
        
        # Publish gains and min distance to servoing node
        self.twist_gains_pub.publish(self.current_gains)
        self.min_distance_pub.publish(Float64(data=self.current_min_distance))
        
        # Reset tracking state
        self.tracking_active = False
        self.tracking_initialized = False
        
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
    
    def save_debug_image(self, frame, detection_input, detection_type="gemini"):
        """Save debug image with detected coordinates or bounding box"""
        try:
            height, width = frame.shape[:2]
            self.get_logger().info(f"Debug image dimensions: {width}x{height}")
            debug_frame = frame.copy()
            
            if detection_type == "dinox":
                # Draw bounding box for DINOX detection
                bbox = detection_input
                cv2.rectangle(debug_frame, (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
                
                # Save to dinox_detection directory
                import time
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
                import time
                timestamp = int(time.time())
                debug_path = os.path.join(self.save_dir, f"gemini_detection_{timestamp}.jpg")
            
            cv2.imwrite(debug_path, debug_frame)
            self.get_logger().info(f"Debug image saved to {debug_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save debug image: {e}")

    def identify_with_chatgpt(self, image):
        """Identify food items using ChatGPT"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            _, buffer = cv2.imencode('.jpg', cv_image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                "max_tokens": 300
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=self.openai_headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                if ',' in content:
                    return [item.strip() for item in content.split(',')]
                else:
                    return [content]
            return None
        except Exception as e:
            self.get_logger().error(f"ChatGPT identification failed: {e}")
            return None
        
    def detect_with_dinox(self, image_path, text_prompt):
        """Detect objects using DINOX and return bounding boxes"""
        try:
            image_url = self.dinox_client.upload_file(image_path)
            
            task = V2Task(
                api_path="/v2/task/dinox/detection",
                api_body={
                    "model": "DINO-X-1.0",
                    "image": image_url,
                    "prompt": {"type": "text", "text": text_prompt},
                    "targets": ["bbox"],
                    "bbox_threshold": 0.25,
                    "iou_threshold": 0.8,
                }
            )
            
            self.dinox_client.run_task(task)
            result = task.result
            
            if not result or "objects" not in result or len(result["objects"]) == 0:
                return None, None, None, None
            
            objects = result["objects"]
            input_boxes = []
            confidences = []
            class_names = []
            class_ids = []
            
            classes = [x.strip().lower() for x in text_prompt.split('.') if x.strip()]
            class_name_to_id = {name: id for id, name in enumerate(classes)}
            
            for obj in objects:
                input_boxes.append(obj["bbox"])
                confidences.append(obj["score"])
                category = obj["category"].lower().strip()
                class_names.append(category)
                class_ids.append(class_name_to_id.get(category, 0))

            return np.array(input_boxes), confidences, class_names, np.array(class_ids)
        except Exception as e:
            self.get_logger().error(f"DINOX detection failed: {e}")
            return None, None, None, None
    
    def detect_food_with_dinox_stack(self, frame):
        """Use ChatGPT + DINOX stack to detect food and return highest confidence bounding box"""
        try:
            # Save frame to temporary file for DINOX
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
                temp_path = tmpfile.name
            cv2.imwrite(temp_path, frame)
            
            # Step 1: Identify food items with ChatGPT
            self.get_logger().info("Identifying food items with ChatGPT...")
            identified_objects = self.identify_with_chatgpt(self.latest_color_image)
            
            if not identified_objects:
                self.get_logger().error("ChatGPT failed to identify food items")
                os.remove(temp_path)
                return None
            
            self.get_logger().info(f"ChatGPT identified: {identified_objects}")
            
            # Step 2: Randomly select one item
            selected_item = random.choice(identified_objects)
            self.get_logger().info(f"Randomly selected: {selected_item}")
            
            # Parse bite information like perception node
            parts = selected_item.rsplit(' ', 1)
            if len(parts) == 2 and parts[1].isdigit():
                item_name = parts[0]
                bite_number = int(parts[1])
            else:
                item_name = selected_item
                bite_number = 1
            
            # Set single/multi bite based on number
            self.single_bite = bite_number <= 1
            self.current_item = item_name
            
            # Step 3: Create DINOX prompt
            text_prompt = item_name + " ."
            
            # Step 4: Detect with DINOX
            self.get_logger().info(f"Detecting with DINOX using prompt: '{text_prompt}'")
            input_boxes, confidences, class_names, class_ids = self.detect_with_dinox(temp_path, text_prompt)
            
            # Cleanup temp file
            os.remove(temp_path)
            
            if input_boxes is None or len(input_boxes) == 0:
                self.get_logger().warn("DINOX could not detect the selected food item")
                return None
            
            # Step 5: Get highest confidence detection
            highest_idx = np.argmax(confidences)
            highest_bbox = input_boxes[highest_idx]
            highest_confidence = confidences[highest_idx]
            
            self.get_logger().info(f"Highest confidence detection: {class_names[highest_idx]} "
                                 f"with confidence {highest_confidence:.3f}")
            self.get_logger().info(f"Bounding box: {highest_bbox}")
            
            # Save debug image
            self.save_debug_image(frame, highest_bbox, "dinox")
            
            return highest_bbox
            
        except Exception as e:
            self.get_logger().error(f"Error with DINOX stack detection: {e}")
            return None
    
    def detect_food_with_gemini(self, frame):
        """Use Gemini to detect food and return center point coordinates"""
        try:
            # Get image dimensions
            height, width = frame.shape[:2]
            self.get_logger().info(f"Frame dimensions sent to Gemini: {width}x{height}")

            # Convert frame to RGB and encode as base64
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame_rgb)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            # Modify the loaded prompt with image dimensions and target food
            prompt_text = self.prompt

            # Add specific food target if we have one
            if self.current_food_target:
                prompt_text = prompt_text.replace(
                    "identify the food item you see", 
                    f"find the {self.current_food_target}"
                )
                prompt_text = prompt_text.replace(
                    "cannot find any food", 
                    f"cannot find a {self.current_food_target}"
                )

            # Generate response
            response = self.model.generate_content([
                prompt_text,
                {"mime_type": "image/jpeg", "data": base64_image}
            ])

            response_text = response.text.strip()
            self.get_logger().info(f"Gemini response: {response_text}")

            # Parse response
            if "FOUND:" in response_text:
                try:
                    coords_part = response_text.split("FOUND:")[1].strip()
                    coords_part = coords_part.split('\n')[0].split(',')

                    if len(coords_part) >= 2:
                        x = int(float(int(coords_part[0].strip())/1000)*width)
                        y = int(float(int(coords_part[1].strip())/1000)*height)

                        # Ensure coordinates are within image bounds
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))

                        self.get_logger().info(f"Parsed coordinates: ({x}, {y})")
                        
                        # Save debug image
                        self.save_debug_image(frame, (x, y), "gemini")
                        
                        # Parse bite information from Gemini response
                        if "FOUND:" in response_text:
                            # Extract item name and bite info if present
                            lines = response_text.split('\n')
                            for line in lines:
                                if 'item:' in line.lower() or 'food:' in line.lower():
                                    item_text = line.split(':')[-1].strip()
                                    parts = item_text.rsplit(' ', 1)
                                    if len(parts) == 2 and parts[1].isdigit():
                                        self.current_item = parts[0]
                                        bite_number = int(parts[1])
                                    else:
                                        self.current_item = item_text
                                        bite_number = 1
                                    self.single_bite = bite_number <= 1
                                    break
                        
                        return (x, y)

                except (ValueError, IndexError) as e:
                    self.get_logger().error(f"Failed to parse coordinates: {e}")

            elif "NOT_FOUND" in response_text:
                self.get_logger().warn("Gemini could not find food item")
            else:
                self.get_logger().warn(f"Unexpected response format: {response_text}")

            return None

        except Exception as e:
            self.get_logger().error(f"Error with Gemini detection: {e}")
            return None
        
    def initialize_sam2_tracking(self, frame, detection_input):
        """Initialize SAM2 tracking with either point or bounding box"""
        try:
            self.predictor.load_first_frame(frame)
            ann_frame_idx = 0
            ann_obj_id = (1,)
            
            if self.detection_model == 'dinox':
                # Use bounding box for DINOX stack
                bbox = np.array([[detection_input[0], detection_input[1]], 
                               [detection_input[2], detection_input[3]]], dtype=np.float32)
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
                )
                self.get_logger().info(f"SAM2 tracking initialized with bounding box: {detection_input}")
            else:
                # Use point for Gemini
                labels = np.array([1], dtype=np.int32)
                points = np.array([detection_input], dtype=np.float32)

                _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
                )
                self.get_logger().info(f"SAM2 tracking initialized with point: {detection_input}")

            self.tracking_initialized = True
            self.tracking_active = True

        except Exception as e:
            self.get_logger().error(f"Failed to initialize SAM2 tracking: {e}")

    def get_mask_centroid(self, mask):
        """Find the centroid of a binary mask"""
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None 
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)

    def get_food_width(self, mask, depth_image, single_bite):
        """Calculate food width and grip points exactly like perception node"""
        # Convert mask to proper format for findContours
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroid = self.get_mask_centroid(mask)

        if not contours:
            self.get_logger().error("No contours found in mask")
            return None, None, None, None, None

        if centroid is None:
            self.get_logger().error("Could not calculate centroid")
            return None, None, None, None, None

        largest_contour = max(contours, key=cv2.contourArea)

        # Check if contour has enough points for minAreaRect
        if len(largest_contour) < 5:
            self.get_logger().error("Contour has insufficient points for minAreaRect")
            return None, None, None, None, None

        # get a rotated rectangle around the segmentation
        rect = cv2.minAreaRect(largest_contour)
        if rect is None:
            self.get_logger().error("minAreaRect returned None")
            return None, None, None, None, None

        # get the box points of the rectangle and convert to integers
        box = cv2.boxPoints(rect)
        if box is None or len(box) != 4:
            self.get_logger().error("boxPoints returned invalid data")
            return None, None, None, None, None

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

                # really jank way of getting orientation
                p_orient = (box[0]+box[1])/2
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

                p_orient = (box[1]+box[2])/2

            # Convert midpoints to integers and find the nearest point on the mask
            width_p1 = self.proj_pix2mask(tuple(map(int, p1)), mask)
            width_p2 = self.proj_pix2mask(tuple(map(int, p2)), mask)

            if width_p1 is None or width_p2 is None:
                self.get_logger().error("Could not project points to mask")
                return None, None, None, None, None

            # get the coordinates relative to RealSense of width points
            rs_width_p1, success1 = self.pixel_to_rs_frame(width_p1[0], width_p1[1], depth_image)
            rs_width_p2, success2 = self.pixel_to_rs_frame(width_p2[0], width_p2[1], depth_image)

            if not success1 or not success2 or rs_width_p1 is None or rs_width_p2 is None:
                self.get_logger().warn("Could not convert width points to RealSense coordinates")
                grip_val = None
            else: 
                # get true distances of points from each other (ignore depth for accuracy)
                rs_width_p1_2d = rs_width_p1[:2]
                rs_width_p2_2d = rs_width_p2[:2]
    
                # Calculate the Euclidean distance between points
                width = np.linalg.norm(rs_width_p1_2d - rs_width_p2_2d)
                self.get_logger().info(f"Width of food item={width:.3f} m")
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
                self.get_logger().info(f"Grip value={grip_val}")

            food_angle = self.get_food_angle_pca(mask)

            self.food_angle_pub.publish(Float64(data=food_angle))

            return grip_val, width_p1, width_p2, food_angle, centroid

        except Exception as e:
            self.get_logger().error(f"Error in get_food_width: {str(e)}")
            return None, None, None, None, None

    def get_food_angle_pca(self, mask):
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

    def pixel_to_rs_frame(self, pixel_x, pixel_y, depth_image):
        """Convert pixel coordinates to 3D coordinates relative to RealSense camera"""
        if self.camera_info is None:
            return None, False
            
        # Camera intrinsics
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4] 
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        
        # Get depth value at pixel
        depth = depth_image[int(pixel_y), int(pixel_x)] / 1000.0  # Convert mm to m

        if depth <= 0:
            return None, False
            
        # Convert to 3D camera coordinates
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z]), True

    def proj_pix2mask(self, px, mask):
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

    def draw_grasp_visualization(self, image, centroid, width_p1=None, width_p2=None, food_angle=None, confidence=1.0):
        """Draw grasp visualization on the image exactly like perception node"""
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
        info_text = [
            f"Detected Item: {self.current_item}",
            f"Food Angle: {food_angle:.2f} deg",
            f"Single Bite: {self.single_bite}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(vis_image, text, (10, 40 + i * 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(vis_image, text, (10, 40 + i * 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            
        return vis_image

    def update_sam2_tracking(self, frame):
        """Update SAM2 tracking and publish position vector with grasp visualization"""
        try:
            # Track object
            out_obj_ids, out_mask_logits = self.predictor.track(frame)
            
            if len(out_obj_ids) == 0:
                self.get_logger().warn("Tracking lost")
                self.publish_zero_vector()
                return
            
            # Get mask and convert to proper format
            out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy()
            mask_2d = (out_mask.squeeze() * 255).astype(np.uint8)
            
            # Get depth image
            if self.latest_depth_image is None:
                self.get_logger().error("No depth image available")
                return
                
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
            
            # Calculate grasp points and width exactly like perception node
            grip_val, width_p1, width_p2, food_angle, centroid = self.get_food_width(mask_2d, depth_image, self.single_bite)
            
            if grip_val is not None:
                # Publish grip value
                grip_msg = Float64()
                grip_msg.data = grip_val
                self.grip_val_pub.publish(grip_msg)
                self.grip_value = grip_val
                
            if centroid is None:
                self.get_logger().warn("No centroid found")
                self.publish_zero_vector()
                return
            
            # Create visualization with grasp points
            vis_image = self.draw_grasp_visualization(frame, centroid, width_p1, width_p2, food_angle, 1.0)
            
            # Apply mask overlay like in perception node
            height, width = frame.shape[:2]
            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            all_mask = mask_2d.reshape((height, width, 1))
            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            vis_image = cv2.addWeighted(vis_image, 1, all_mask, 0.5, 0)
            
            # Publish segmented image
            try:
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', vis_image)[1]).tobytes()
                self.segmented_image_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish segmented image: {e}")

            cv2.imshow('Food Detection with Pose', vis_image)
            cv2.waitKey(1)  # Non-blocking
            
            # Convert to position vector (finger midpoint to food centroid)
            position_vector = self.calculate_position_vector(centroid[0], centroid[1], mask_2d)
            
            if position_vector is not None:
                self.position_vector_pub.publish(position_vector)
                self.get_logger().info(f"Published position vector: ({position_vector.x:.3f}, {position_vector.y:.3f}, {position_vector.z:.3f})")
                
                # Publish RViz markers
                self.publish_rviz_markers(centroid[0], centroid[1], position_vector)
            else:
                self.publish_zero_vector()
                
        except Exception as e:
            self.get_logger().error(f"Error in SAM2 tracking: {e}")
            self.publish_zero_vector()
    
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
            finger_midpoint.z += 0.03  # Adjust based on your gripper's finger length
            
            return finger_midpoint
            
        except Exception as e:
            self.get_logger().warn(f"Could not get finger midpoint: {e}", throttle_duration_sec=2.0)
            return None
    
    def calculate_position_vector(self, pixel_x, pixel_y, segmentation_mask):
        """Calculate position vector from finger midpoint to food centroid"""
        if self.camera_info is None or self.latest_depth_image is None:
            return None
        
        # get the pixel indeces of the segmentation
        object_pixels = segmentation_mask == 255
        
        try:
            # Get depth at pixel
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
            mask_depths = depth_image[object_pixels]
            mask_depths = mask_depths[mask_depths > 0]  # Filter out invalid depths
    
            # use the average mask depth instead of a single point
            avg_depth = np.mean(mask_depths) / 1000.0  # Convert mm to m
            
            # Convert pixel to 3D point in camera frame
            food_x = (pixel_x - self.cx) * avg_depth / self.fx
            food_y = (pixel_y - self.cy) * avg_depth / self.fy
            food_z = avg_depth
            
            # Transform food position to end effector frame
            food_stamped = PointStamped()
            food_stamped.header.frame_id = 'realsense_link'
            food_stamped.header.stamp = self.get_clock().now().to_msg()
            food_stamped.point.x = food_x
            food_stamped.point.y = food_y
            food_stamped.point.z = food_z
            
            # Transform to end effector frame
            food_in_effector = self.tf_buffer.transform(
                food_stamped, 'end_effector_link', timeout=rclpy.duration.Duration(seconds=0.5))
            
            # Get finger midpoint
            finger_midpoint = self.get_finger_midpoint_in_end_effector_frame()
            if finger_midpoint is None:
                return None
            
            # Calculate vector from finger to food
            vector = Vector3()
            vector.x = food_in_effector.point.x - finger_midpoint.x
            vector.y = food_in_effector.point.y - finger_midpoint.y
            vector.z = food_in_effector.point.z - finger_midpoint.z
            
            # Check if we're close enough (within current min distance)
            # distance = np.linalg.norm([vector.x, vector.y, vector.z])
            # if distance < self.current_min_distance:
            #     self.get_logger().info(f"Robot is within {self.current_min_distance}m of target, stopping")
            #     return Vector3(x=0.0, y=0.0, z=0.0)
            
            return vector
            
        except Exception as e:
            self.get_logger().error(f"Error calculating position vector: {e}")
            return None
    
    def publish_rviz_markers(self, cx, cy, position_vector):
        """Publish RViz visualization markers"""
        try:
            current_time = self.get_clock().now().to_msg()
            
            # Get finger midpoint
            finger_midpoint = self.get_finger_midpoint_in_end_effector_frame()
            if finger_midpoint is None:
                return
            
            # Get food position in end effector frame
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
            depth_mm = depth_image[cy, cx]
            depth_m = depth_mm / 1000.0
            
            if depth_m <= 0:
                return
            
            # Convert pixel to 3D point and transform to end effector frame
            food_x = (cx - self.cx) * depth_m / self.fx
            food_y = (cy - self.cy) * depth_m / self.fy
            food_z = depth_m
            
            food_stamped = PointStamped()
            food_stamped.header.frame_id = 'realsense_link'
            food_stamped.header.stamp = current_time
            food_stamped.point.x = food_x
            food_stamped.point.y = food_y
            food_stamped.point.z = food_z
            
            food_in_effector = self.tf_buffer.transform(
                food_stamped, 'end_effector_link', timeout=rclpy.duration.Duration(seconds=0.5))
            
            # 1. Target point marker (food centroid) - Green sphere
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
            
            # 2. Gripper point marker (finger midpoint) - Red sphere
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
            
            # 3. Vector arrow (from finger to food) - Blue arrow
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
            
        except Exception as e:
            self.get_logger().error(f"Error publishing RViz markers: {e}")
            
    def process_frame(self):
        """Main processing loop"""
        if not self.service_active or self.latest_color_image is None:
            return
        
        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            
            if not self.tracking_initialized:
                if self.detection_model == 'dinox':
                    detection_bbox = self.detect_food_with_dinox_stack(frame)
                    if detection_bbox is not None:
                        self.initialize_sam2_tracking(frame, detection_bbox)
                else: 
                    detection_point = self.detect_food_with_gemini(frame)
                    if detection_point is not None:
                        self.initialize_sam2_tracking(frame, detection_point)
            else:
                # Continue tracking with SAM2 and live visualization
                self.update_sam2_tracking(frame)
                
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