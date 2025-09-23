#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import yaml
import os

# MediaPipe imports
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        
        # Load config
        config_path = os.path.expanduser('~/raf-live/config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Face detection configuration
        self.target_distance = self.config['feeding']['face_detection'].get('target_distance', 0.04)
        # need to know single bite/multibite to know how the robot should approach the mouth
        self.single_bite = True
        
        # ROS setup
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        self.camera_info = None
        self.last_depth = 0.0
        
        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Camera parameters
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_frame_id = 'realsense_link'
        
        # State
        self.detection_active = False
        
        # Initialize MediaPipe FaceLandmarker
        try:
            base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.get_logger().info('MediaPipe FaceLandmarker initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize MediaPipe FaceLandmarker: {e}')
            self.detector = None
        
        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.single_bite_sub = self.create_subscription(
            Bool, '/single_bite', self.single_bite_callback, 10)
        
        # Control subscriber - simplified interface
        self.start_detection_sub = self.create_subscription(
            Bool, '/start_face_detection', self.start_detection_callback, 10)
        
        # Publishers
        self.position_vector_pub = self.create_publisher(Vector3, '/position_vector', 1)
        self.food_angle_pub = self.create_publisher(Float64, '/food_angle', 1)  # Always 0 for face detection

        # the orchestrator detects when the magnitude of the vector is really low, but doesnt differentiate between a zero vector because its close or a zero vector because the detection stopped
        # this will help the orchestrator so that it doesnt mistake the zero vector when mouth isnt detected as being close to the food
        self.vector_pause_pub = self.create_publisher(Bool, '/vector_pause', 1)
        
        # Add processed image publisher like in image_visualizer.py
        self.processed_image_pub = self.create_publisher(
            CompressedImage, '/processed_image', 10)
        
        # Processing timer (only active when detection is on)
        self.timer = None
        
        self.get_logger().info('Face Detection Node initialized and ready!')
    
    def camera_info_callback(self, msg):
        """Update camera intrinsic parameters"""
        if self.camera_info is None:
            self.camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(f"Camera parameters updated: fx={self.fx}, fy={self.fy}")
    
    def color_callback(self, msg):
        self.latest_color_image = msg

    def single_bite_callback(self, msg):
        """Handle single bite signal"""
        self.single_bite = msg.data
    
    def depth_callback(self, msg):
        self.latest_depth_image = msg
    
    def start_detection_callback(self, msg):
        """Simple on/off control"""
        if msg.data and not self.detection_active:
            self.get_logger().info("Starting face detection")
            self.detection_active = True
            self.last_depth = 0.0
            
            # Start processing timer
            if self.timer:
                self.timer.cancel()
            self.timer = self.create_timer(0.1, self.process_frame)
            
        elif not msg.data and self.detection_active:
            self.get_logger().info("Stopping face detection")
            self.detection_active = False
            
            # Stop processing timer
            if self.timer:
                self.timer.cancel()
                self.timer = None
            
            # Publish zero vector to stop servoing
            self._publish_zero_vector()
    
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
            finger_midpoint = Vector3()
            finger_midpoint.x = (right_pos.x + left_pos.x) / 2.0
            finger_midpoint.y = (right_pos.y + left_pos.y) / 2.0
            finger_midpoint.z = (right_pos.z + left_pos.z) / 2.0
            # Add finger pad length
            finger_midpoint.z += 0.03

            # robot goes lower for multi-bite foods so ser can take a bite
            if not self.single_bite:
                finger_midpoint.y += 0.025
            
            return finger_midpoint
            
        except Exception as e:
            self.get_logger().warn(f"Could not get finger midpoint: {e}", throttle_duration_sec=2.0)
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
    
    def calculate_mouth_position_vector(self, mouth_pixel_x, mouth_pixel_y, average_lip_depth):
        """Calculate position vector from finger to mouth"""
        # Transform mouth position to end effector frame
        mouth_in_effector = self.transform_mouth_to_end_effector_frame(mouth_pixel_x, mouth_pixel_y, average_lip_depth)
        if mouth_in_effector is None:
            return None
        
        # Get finger midpoint
        finger_midpoint = self.get_finger_midpoint_in_end_effector_frame()
        if finger_midpoint is None:
            return None
        
        # Calculate vector from finger to mouth
        vector = Vector3()
        vector.x = mouth_in_effector.x - finger_midpoint.x
        vector.y = mouth_in_effector.y - finger_midpoint.y
        vector.z = mouth_in_effector.z - finger_midpoint.z
        
        return vector
    
    def draw_mouth_landmarks_on_image(self, rgb_image, detection_result):
        """Draw mouth landmarks and determine mouth openness"""
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the mouth landmarks
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )

            # Calculate mouth center and openness
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            mouth_center_x = (upper_lip.x + lower_lip.x) / 2
            mouth_center_y = (upper_lip.y + lower_lip.y) / 2

            # Calculate distance between upper and lower lip
            mouth_open_distance = np.sqrt((upper_lip.x - lower_lip.x)**2 + (upper_lip.y - lower_lip.y)**2)

            # Get image dimensions
            image_height, image_width, _ = annotated_image.shape
            center_x = int(mouth_center_x * image_width)
            center_y = int(mouth_center_y * image_height)

            # Threshold to determine if the mouth is open
            if mouth_open_distance > 0.03:
                cv2.circle(annotated_image, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)
                mouth_status = "OPEN"
            else:
                cv2.circle(annotated_image, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
                mouth_status = "CLOSED"
            
            # Add status text
            cv2.putText(annotated_image, f"Mouth: {mouth_status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_image, f"Distance: {mouth_open_distance:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return annotated_image, center_x, center_y

        return annotated_image, None, None
    
    def publish_processed_image(self, vis_image):
        """
        Publish processed image as compressed image message - copied from image_visualizer.py
        
        Args:
            vis_image: Visualization image to publish
        """
        try:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', vis_image)[1]).tobytes()
            self.processed_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish processed image: {e}")
    
    def process_frame(self):
        """Process the latest frame for face detection"""
        if not self.detection_active or self.detector is None:
            return
        
        if self.latest_color_image is None or self.latest_depth_image is None:
            self._publish_zero_vector()
            return
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect face landmarks
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.face_landmarks:
                self._publish_zero_vector()
                self.get_logger().warn("No face detected, publishing zero vector")
                self.vector_pause_pub.publish(Bool(data=True))  # Indicate pause due to no detection
                # Still publish the image even if no face is detected
                bgr_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr_image, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.publish_processed_image(bgr_image)
                return
            else:
                self.vector_pause_pub.publish(Bool(data=False))
            
            # Draw mouth landmarks and get mouth center - using the same function as unlabeled script
            annotated_image, center_x, center_y = self.draw_mouth_landmarks_on_image(rgb_frame, detection_result)
            
            if center_x is None or center_y is None:
                self._publish_zero_vector()
                # Convert back to BGR for publishing
                bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                self.publish_processed_image(bgr_image)
                return
            
            # Get all lip landmarks for depth averaging
            face_landmarks = detection_result.face_landmarks[0]
            lip_indices = [13,312,311,310,318,402,317,14,87,178,88,80,81,82,40,39,37,0,267,269,270,321,405,314,17,84,181,91]
            lip_points = [face_landmarks[i] for i in lip_indices]
            
            # Get image dimensions for pixel conversion
            image_height, image_width, _ = cv_image.shape
            
            # Collect depth values at each lip landmark pixel
            lip_depths = []
            for landmark in lip_points:
                pixel_x = int(landmark.x * image_width)
                pixel_y = int(landmark.y * image_height)
                depth = self.get_depth_at_pixel(pixel_x, pixel_y)

                if depth > 0.22 and depth is not None:  # filter out readings of food on the gripper or invalid readings
                    lip_depths.append(depth)
            
            # Compute the average depth if any valid depths were found
            if not lip_depths:
                self.get_logger().warn("No valid mouth depth readings found.")
                self._publish_zero_vector()
                # Convert back to BGR for publishing
                bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr_image, "No valid depth", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.publish_processed_image(bgr_image)
                return
                
            # SAFETY FEATURES
            # make sure it doesn't go out of realistic bounds
            average_lip_depth = sum(lip_depths) / len(lip_depths)
            if average_lip_depth < 0.22:
                average_lip_depth = 0.22
                self.get_logger().warn("Average lip depth too close, setting to minimum 0.22m")
            elif average_lip_depth > 0.75:
                average_lip_depth = 0.75
                self.get_logger().warn("Average lip depth too far, setting to maximum 0.75m")

            if average_lip_depth > self.last_depth and self.last_depth != 0:
                average_lip_depth = self.last_depth
                self.get_logger().warn("Face detected further away, using last valid depth")

            self.last_depth = average_lip_depth
            
            # Add depth info to the annotated image
            cv2.putText(annotated_image, f"Depth: {average_lip_depth:.3f}m", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Calculate position vector using the helper method
            vector = self.calculate_mouth_position_vector(center_x, center_y, average_lip_depth)
            if vector is None:
                self._publish_zero_vector()
                # Convert back to BGR for publishing
                bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr_image, "Transform failed", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.publish_processed_image(bgr_image)
                return
            
            # Check if we're close enough (within target distance)
            distance = np.linalg.norm([vector.x, vector.y, vector.z])
            if distance < self.target_distance:
                self.get_logger().info(f"Robot is within {self.target_distance}m of mouth, publishing zero vector")
                self._publish_zero_vector()
                # Add "TARGET REACHED" text to image
                cv2.putText(annotated_image, "TARGET REACHED", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Publish the position vector and zero food angle (no orientation control)
                self.position_vector_pub.publish(vector)
                self.food_angle_pub.publish(Float64(data=0.0))  # Always 0 for face detection
                # Add vector info to image
                cv2.putText(annotated_image, f"Vector: [{vector.x:.3f}, {vector.y:.3f}, {vector.z:.3f}]", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_image, f"Distance: {distance:.3f}m", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert back to BGR for publishing (OpenCV uses BGR, but we've been working in RGB)
            bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            self.publish_processed_image(bgr_image)
                
        except Exception as e:
            self.get_logger().error(f"Error in mouth detection: {str(e)}")
            self._publish_zero_vector()
    
    def _publish_zero_vector(self):
        """Publish zero position vector"""
        zero_vector = Vector3(x=0.0, y=0.0, z=0.0)
        self.position_vector_pub.publish(zero_vector)
        self.food_angle_pub.publish(Float64(data=0.0))


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()