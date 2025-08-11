import rclpy
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, PointStamped
from std_msgs.msg import Float64MultiArray
import tf2_ros
import tf2_geometry_msgs

class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        
        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Camera parameters
        self.camera_info = None
        self.fx = 615.0  # Default values, will be updated
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_frame_id = 'realsense_link'
        
        # Initialize Mediapipe FaceLandmarker
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
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

        # RViz visualization publishers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/visual_servo_markers',
            10
        )
        
        self.vector_marker_pub = self.create_publisher(
            MarkerArray,
            '/position_vector_markers',
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        # Publisher for visual servo data (x, y, depth)
        self.visual_servo_pub = self.create_publisher(
            Float64MultiArray,
            '/visual_servo_data',
            10
        )

        self.visual_servo_vector_pub = self.create_publisher(
            Vector3,
            '/visual_servo_vector',
            10
        )
        
        # Publisher for finger midpoint marker
        self.midpoint_pub = self.create_publisher(Marker, 'finger_midpoint_marker', 10)
        
        # Timer for processing
        self.timer = self.create_timer(0.1, self.process_frame)
        
        self.get_logger().info('Face detection node initialized and ready!')

    def camera_info_callback(self, msg):
        """Update camera intrinsic parameters"""
        if self.camera_info is None:  # Only update once
            self.camera_info = msg
            self.fx = msg.k[0]  # focal length x
            self.fy = msg.k[4]  # focal length y
            self.cx = msg.k[2]  # principal point x
            self.cy = msg.k[5]  # principal point y
            
            self.get_logger().info(f"Updated camera parameters: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def color_callback(self, msg):
        self.latest_color_image = msg

    def depth_callback(self, msg):
        self.latest_depth_image = msg

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

    def pixel_to_3d_point(self, pixel_x, pixel_y, depth):
        """Convert pixel coordinates to 3D point in camera frame"""
        if self.camera_info is None:
            return None
            
        # Convert pixel to 3D point using camera intrinsics
        x = (pixel_x - self.cx) * depth / self.fx
        y = (pixel_y - self.cy) * depth / self.fy
        z = depth
        
        return Point(x=x, y=y, z=z)

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

    def transform_mouth_to_end_effector_frame(self, mouth_pixel_x, mouth_pixel_y, depth):
        """Transform mouth position from camera frame to end effector frame"""
        # Project pixel to 3D point in camera frame
        mouth_z = depth
        mouth_x = (mouth_pixel_x - self.cx) * depth / self.fx
        mouth_y = (mouth_pixel_y - self.cy) * depth / self.fy

        # Create a stamped point for transformation
        mouth_stamped = PointStamped()
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

    def publish_visualization_markers(self, mouth_center_x, mouth_center_y, depth_value):
        """Create and publish visualization markers for RViz"""
        marker_array = MarkerArray()
        current_time = rclpy.time.Time().to_msg()
        
        # Get finger midpoint in end effector frame
        finger_midpoint = self.get_finger_midpoint_in_end_effector_frame()
        if finger_midpoint is None:
            return None, None
            
        # Visualize the finger midpoint in RViz
        finger_marker = Marker()
        finger_marker.header.frame_id = "end_effector_link"
        finger_marker.header.stamp = current_time
        finger_marker.ns = "finger_midpoint"
        finger_marker.id = 0
        finger_marker.type = Marker.SPHERE
        finger_marker.action = Marker.ADD
        finger_marker.pose.position = finger_midpoint
        finger_marker.pose.orientation.w = 1.0
        finger_marker.scale.x = 0.01  # 1 cm sphere
        finger_marker.scale.y = 0.01
        finger_marker.scale.z = 0.01
        finger_marker.color.a = 1.0  # Opaque
        finger_marker.color.r = 1.0  # Red
        finger_marker.color.g = 0.0
        finger_marker.color.b = 0.0
        self.midpoint_pub.publish(finger_marker)
        
        # Transform mouth position to end effector frame
        mouth_in_effector = None
        if depth_value > 0:
            mouth_in_effector = self.transform_mouth_to_end_effector_frame(mouth_center_x, mouth_center_y, depth_value)
            if mouth_in_effector:
                # 1. Current mouth position (green sphere) in end effector frame
                mouth_marker = Marker()
                mouth_marker.header.frame_id = "end_effector_link"
                mouth_marker.header.stamp = current_time
                mouth_marker.ns = "face_detection"
                mouth_marker.id = 0
                mouth_marker.type = Marker.SPHERE
                mouth_marker.action = Marker.ADD
                mouth_marker.pose.position = mouth_in_effector
                mouth_marker.pose.orientation.w = 1.0
                mouth_marker.scale.x = 0.03
                mouth_marker.scale.y = 0.03
                mouth_marker.scale.z = 0.03
                mouth_marker.color.r = 0.0
                mouth_marker.color.g = 1.0
                mouth_marker.color.b = 0.0
                mouth_marker.color.a = 1.0
                marker_array.markers.append(mouth_marker)

        # Publish markers
        self.marker_pub.publish(marker_array)

        # Calculate the vector from finger midpoint to mouth (both in end effector frame)
        if mouth_in_effector and finger_midpoint:
            vector = Vector3(
                x = mouth_in_effector.x - finger_midpoint.x,
                y = mouth_in_effector.y - finger_midpoint.y,
                z = mouth_in_effector.z - finger_midpoint.z
            )

            if np.linalg.norm([vector.x, vector.y, vector.z]) < 0.03:
                self.get_logger().info("Robot is within 3cm, goal reached")

                vector = Vector3(
                    x = 0,
                    y = 0,
                    z = 0
                )

                return vector, mouth_in_effector
            return vector, mouth_in_effector
        return None, mouth_in_effector

    def publish_position_vector(self, position_vector):
        """Publish as vec3 and visualize in RViz"""
        if position_vector:
            vector_msg = Vector3()
            vector_msg.x = position_vector.x
            vector_msg.y = position_vector.y
            vector_msg.z = position_vector.z
            
            self.visual_servo_vector_pub.publish(vector_msg)
            self.get_logger().info(f"Published visual servo vector (finger->mouth): {vector_msg}")
        else:
            self.get_logger().warn("No valid position vector to publish.")

    def publish_vector_marker(self, vector, mouth_point):
        """Publish arrow marker from finger midpoint to mouth"""
        if not vector or not mouth_point:
            return
        
        finger_midpoint = self.get_finger_midpoint_in_end_effector_frame()
        if not finger_midpoint:
            return
        
        marker_array = MarkerArray()
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "end_effector_link"
        arrow_marker.header.stamp = rclpy.time.Time().to_msg()
        arrow_marker.ns = "position_vector"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # Arrow points FROM finger midpoint TO mouth
        arrow_marker.points = [finger_midpoint, mouth_point]
        
        arrow_marker.scale.x = 0.01  # shaft width
        arrow_marker.scale.y = 0.02  # head width
        arrow_marker.color.b = 1.0
        arrow_marker.color.a = 0.8
        
        marker_array.markers.append(arrow_marker)
        self.vector_marker_pub.publish(marker_array)

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

            # identify all of the lip landmarks: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
            lip_indices = [13,312,311,310,318,402,317,14,87,178,88,80,81,82,40,39,37,0,267,269,270,321,405,314,17,84,181,91]
            lip_points = [face_landmarks[i] for i in lip_indices]
            
            # Get image dimensions
            image_height, image_width, _ = annotated_image.shape
            
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

            # Calculate distance between upper and lower lip
            mouth_open_distance = np.sqrt((upper_lip.x - lower_lip.x)**2 + (upper_lip.y - lower_lip.y)**2)

            # Get image dimensions
            image_height, image_width, _ = annotated_image.shape
            center_x = int(mouth_center_x * image_width)
            center_y = int(mouth_center_y * image_height)

            # Get depth and publish visual servo data
            # depth_value = self.get_depth_at_pixel(center_x, center_y)
            
            # Publish markers and get vector from finger midpoint to mouth
            position_vector, mouth_point_in_effector = self.publish_visualization_markers(center_x, center_y, average_lip_depth)

            # Publish position vector and markers
            servo_data = Float64MultiArray()
            servo_data.data = [float(center_x), float(center_y), average_lip_depth]
            self.visual_servo_pub.publish(servo_data)

            self.publish_position_vector(position_vector)
            self.publish_vector_marker(position_vector, mouth_point_in_effector)

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
            cv2.putText(annotated_image, f"Depth: {average_lip_depth:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated_image

    def process_frame(self):
        """Process the latest frame for face detection"""
        if self.latest_color_image is None:
            return
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            
            # Convert BGR to RGB for Mediapipe processing
            rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Create Mediapipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect face landmarks
            detection_result = self.detector.detect(mp_image)

            # Draw mouth landmarks and check mouth openness
            annotated_image = self.draw_mouth_landmarks_on_image(rgb_frame, detection_result)

            # Display the annotated image
            cv2.imshow('Mouth Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('Shutting down face detection node...')
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    node = FaceDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()