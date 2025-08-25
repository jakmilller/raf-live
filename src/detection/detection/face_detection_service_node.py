#!/home/mcrr-lab/deploy-env/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Bool, Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import rclpy.time
import rclpy.duration

# MediaPipe imports for face detection
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import the service interface
from raf_interfaces.srv import StartFaceServoing
from tracking import CoordinateTransforms

class FaceDetectionServiceNode(Node):
    def __init__(self):
        super().__init__('face_detection_service_node')
        
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        self.camera_info = None
        
        # Service state
        self.service_active = False
        self.current_gains = Vector3()
        self.current_target_distance = 0.03  # Default 3cm
        self.current_goal_handle = None  # Store the service goal handle
        
        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Initialize coordinate transforms
        self.coordinate_transforms = CoordinateTransforms(self)
        
        # Camera parameters
        self.fx = 615.0  # Default values, will be updated
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_frame_id = 'realsense_link'
        
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
        
        # Subscribe to camera info for intrinsic parameters
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Subscribe to RealSense color and depth image topics
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
        
        # Publishers for servoing control (matching food detection pattern)
        self.position_vector_pub = self.create_publisher(Vector3, '/position_vector', 1)
        self.food_angle_pub = self.create_publisher(Float64, '/food_angle', 1)  # Will always be 0
        self.twist_gains_pub = self.create_publisher(Vector3, '/twist_gains', 10)
        self.min_distance_pub = self.create_publisher(Float64, '/min_distance', 10)
        self.servoing_on_pub = self.create_publisher(Bool, '/servoing_on', 1)
        
        # Publisher to signal face servoing completion (like food_acquired)
        self.face_servoing_complete_pub = self.create_publisher(Bool, '/face_servoing_complete', 1)
        
        # Create the service
        self.face_servoing_service = self.create_service(
            StartFaceServoing, 
            'start_face_servoing', 
            self.start_face_servoing_callback
        )
        
        # Processing timer (only active during service)
        self.timer = None
        
        # State tracking
        self.consecutive_zero_count = 0
        self.max_zero_count = 2  # Stop after 2 consecutive zero vectors
        
        self.get_logger().info('Face Detection Service Node initialized and ready!')

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
        self.latest_color_image = msg

    def depth_callback(self, msg):
        self.latest_depth_image = msg

    def start_face_servoing_callback(self, request, response):
        """Service callback to start face servoing"""
        self.get_logger().info('Face servoing service called')
        
        if self.detector is None:
            response.success = False
            response.message = "MediaPipe face detector not initialized"
            return response
        
        if self.service_active:
            response.success = False
            response.message = "Face servoing already active"
            return response
        
        # Store service parameters
        self.current_gains.x = request.gain_planar
        self.current_gains.y = request.gain_planar  # Same for both x and y
        self.current_gains.z = request.gain_depth
        self.current_target_distance = request.target_distance
        
        # Publish gains and min distance to servoing node
        self.twist_gains_pub.publish(self.current_gains)
        self.min_distance_pub.publish(Float64(data=self.current_target_distance))
        
        # Reset state
        self.consecutive_zero_count = 0
        
        # Activate service
        self.service_active = True
        
        # Start processing timer
        if self.timer:
            self.timer.cancel()
        self.timer = self.create_timer(0.1, self.process_frame)
        
        # Enable servoing
        self.servoing_on_pub.publish(Bool(data=True))
        
        self.get_logger().info(f'Started face servoing with gains: planar={request.gain_planar}, depth={request.gain_depth}, target_distance={request.target_distance}')
        
        response.success = True
        response.message = "Face servoing started successfully"
        return response

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
        return self.coordinate_transforms.get_finger_midpoint_in_end_effector_frame()

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

    def detect_mouth_and_publish_vector(self):
        """Detect mouth position and publish position vector"""
        if self.latest_color_image is None or self.latest_depth_image is None:
            self.publish_zero_vector()
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
                self.publish_zero_vector()
                return
            
            # Get mouth center from landmarks
            face_landmarks = detection_result.face_landmarks[0]
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            mouth_center_x = (upper_lip.x + lower_lip.x) / 2
            mouth_center_y = (upper_lip.y + lower_lip.y) / 2
            
            # Get image dimensions
            image_height, image_width, _ = cv_image.shape
            center_x = int(mouth_center_x * image_width)
            center_y = int(mouth_center_y * image_height)
            
            # Get all lip landmarks for depth averaging
            lip_indices = [13,312,311,310,318,402,317,14,87,178,88,80,81,82,40,39,37,0,267,269,270,321,405,314,17,84,181,91]
            lip_points = [face_landmarks[i] for i in lip_indices]
            
            # Collect depth values at each lip landmark pixel
            lip_depths = []
            for landmark in lip_points:
                pixel_x = int(landmark.x * image_width)
                pixel_y = int(landmark.y * image_height)
                depth = self.get_depth_at_pixel(pixel_x, pixel_y)
                if depth > 0.22 and depth is not None:  # filter out readings of food on the gripper or invalid readings
                    lip_depths.append(depth)
            
            # Compute the average depth if any valid depths were found
            if lip_depths:
                average_lip_depth = sum(lip_depths) / len(lip_depths)
            else:
                self.get_logger().warn("No valid mouth depth readings found.")
                self.publish_zero_vector()
                return
            
            # Calculate position vector using the helper method
            vector = self.calculate_mouth_position_vector(center_x, center_y, average_lip_depth)
            if vector is None:
                self.publish_zero_vector()
                return
            
            # Check if we're close enough (within target distance)
            distance = np.linalg.norm([vector.x, vector.y, vector.z])
            if distance < self.current_target_distance:
                self.get_logger().info(f"Robot is within {self.current_target_distance}m of mouth, publishing zero vector")
                self.publish_zero_vector()
                return
            
            # Publish the position vector and zero food angle (no orientation control)
            self.position_vector_pub.publish(vector)
            self.food_angle_pub.publish(Float64(data=0.0))  # Always 0 for face detection
            
            # Reset consecutive zero count since we published a non-zero vector
            self.consecutive_zero_count = 0
            
            self.get_logger().info(f"Published position vector: ({vector.x:.3f}, {vector.y:.3f}, {vector.z:.3f})")
                
        except Exception as e:
            self.get_logger().error(f"Error in mouth detection: {str(e)}")
            self.publish_zero_vector()

    def publish_zero_vector(self):
        """Publish zero position vector and handle service completion"""
        zero_vector = Vector3()
        zero_vector.x = 0.0
        zero_vector.y = 0.0
        zero_vector.z = 0.0
        self.position_vector_pub.publish(zero_vector)
        self.food_angle_pub.publish(Float64(data=0.0))  # Always 0 for face detection
        
        # Track consecutive zero vectors
        self.consecutive_zero_count += 1
        
        # If we've published enough zero vectors, face servoing is complete
        if self.consecutive_zero_count >= self.max_zero_count and self.service_active:
            self.get_logger().info(f"Published {self.consecutive_zero_count} consecutive zero vectors - face servoing complete!")
            self._complete_face_servoing()

    def _complete_face_servoing(self):
        """Complete face servoing and signal orchestrator"""
        self.get_logger().info("Face servoing completed - disabling servoing and ending service")
        
        # Disable servoing
        self.servoing_on_pub.publish(Bool(data=False))
        
        # Stop the service
        self.service_active = False
        if self.timer:
            self.timer.cancel()
            self.timer = None
        
        # Signal completion to orchestrator (like food_acquired but for face)
        self.face_servoing_complete_pub.publish(Bool(data=True))
        
        # Reset state
        self.consecutive_zero_count = 0

    def process_frame(self):
        """Process the latest frame for face detection (only when service is active)"""
        if not self.service_active or self.detector is None:
            return
        
        self.detect_mouth_and_publish_vector()

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()