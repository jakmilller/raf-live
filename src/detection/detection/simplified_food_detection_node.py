#!/usr/bin/env python3

"""
Simplified Food Detection Node with Clean State Machine

States:
- IDLE: Waiting for start command
- DETECTING: Running ChatGPT + DINOX to get bounding box
- TRACKING: Running SAM2 and publishing position vectors
- FAILED: Segmentation lost, notifying orchestrator

Flow:
1. Receive start_food_detection -> DETECTING
2. Run ChatGPT + DINOX -> TRACKING (publish ready signal)
3. Continuously track with SAM2 and publish vectors
4. If tracking lost -> FAILED (notify orchestrator)
5. Receive stop_food_detection -> IDLE
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float64, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import asyncio
import threading
from enum import Enum

# Import your existing detection modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detectors.dinox_detector import DinoxDetector
from tracking.sam2_tracker import SAM2Tracker


class DetectionState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    TRACKING = "tracking"
    FAILED = "failed"


class SimplifiedFoodDetectionNode(Node):
    def __init__(self):
        super().__init__('simplified_food_detection')

        # State machine
        self.state = DetectionState.IDLE
        self.state_lock = threading.Lock()

        # Detection components
        self.dinox_detector = None
        self.sam2_tracker = None
        self.bridge = CvBridge()

        # Current detection data
        self.current_image = None
        self.current_depth = None
        self.camera_info = None
        self.bounding_box = None
        self.food_item = None
        self.single_bite = False

        # Tracking state
        self.tracking_failures = 0
        self.max_tracking_failures = 3
        self.table_depth = 0.31  # Default table depth

        # Publishers
        self.position_vector_pub = self.create_publisher(Vector3, '/position_vector', 10)
        self.grip_value_pub = self.create_publisher(Float64, '/grip_value', 10)
        self.food_height_pub = self.create_publisher(Float64, '/food_height', 10)
        self.food_depth_pub = self.create_publisher(Float64, '/food_depth', 10)
        self.food_detection_ready_pub = self.create_publisher(Bool, '/food_detection_ready', 10)
        self.tracking_lost_pub = self.create_publisher(Bool, '/tracking_lost', 10)
        self.currently_serving_pub = self.create_publisher(String, '/currently_serving', 10)
        self.vector_pause_pub = self.create_publisher(Bool, '/vector_pause', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.start_detection_sub = self.create_subscription(Bool, '/start_food_detection', self.start_detection_callback, 10)
        self.voice_command_sub = self.create_subscription(String, '/voice_commands', self.voice_command_callback, 10)

        # Voice commands queue
        self.voice_commands = []
        self.voice_lock = threading.Lock()

        # Initialize detection components
        self.initialize_detectors()

        # Main processing timer
        self.timer = self.create_timer(0.033, self.process_loop)  # 30 FPS

        self.get_logger().info("Simplified Food Detection Node initialized")

    def initialize_detectors(self):
        """Initialize detection components"""
        try:
            self.dinox_detector = DinoxDetector()
            self.sam2_tracker = SAM2Tracker()
            self.get_logger().info("Detection components initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize detectors: {e}")

    def image_callback(self, msg):
        """Store latest color image"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def depth_callback(self, msg):
        """Store latest depth image"""
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def camera_info_callback(self, msg):
        """Store camera info"""
        self.camera_info = msg

    def voice_command_callback(self, msg):
        """Store voice commands"""
        with self.voice_lock:
            self.voice_commands.append(msg.data)
            self.get_logger().info(f"Voice command received: {msg.data}")

    def start_detection_callback(self, msg):
        """Handle start/stop detection commands"""
        with self.state_lock:
            if msg.data and self.state == DetectionState.IDLE:
                self.get_logger().info("🔍 Starting food detection")
                self.state = DetectionState.DETECTING
                self.reset_detection_state()
            elif not msg.data:
                self.get_logger().info("⏹️ Stopping food detection")
                self.state = DetectionState.IDLE
                self.reset_detection_state()

    def reset_detection_state(self):
        """Reset all detection state"""
        self.bounding_box = None
        self.food_item = None
        self.single_bite = False
        self.tracking_failures = 0
        if self.sam2_tracker:
            self.sam2_tracker.reset()

    def process_loop(self):
        """Main processing loop based on current state"""
        if not self.current_image is not None or self.current_depth is None:
            return

        with self.state_lock:
            current_state = self.state

        try:
            if current_state == DetectionState.DETECTING:
                self.process_detection()
            elif current_state == DetectionState.TRACKING:
                self.process_tracking()
            elif current_state == DetectionState.FAILED:
                self.process_failure()
            # IDLE state does nothing

        except Exception as e:
            self.get_logger().error(f"Error in process loop: {e}")
            self.transition_to_failed()

    def process_detection(self):
        """Run initial detection (ChatGPT + DINOX)"""
        self.get_logger().info("🔍 Running initial food detection...")

        # Step 1: Get food item from voice or ChatGPT
        food_item = self.get_food_item()
        if not food_item:
            self.get_logger().error("Failed to identify food item")
            self.transition_to_failed()
            return

        # Step 2: Run DINOX detection
        bounding_box = self.run_dinox_detection(food_item)
        if bounding_box is None:
            self.get_logger().error("DINOX detection failed")
            self.transition_to_failed()
            return

        # Step 3: Initialize SAM2 tracking
        if not self.initialize_sam2_tracking(bounding_box):
            self.get_logger().error("SAM2 initialization failed")
            self.transition_to_failed()
            return

        # Step 4: Transition to tracking
        self.bounding_box = bounding_box
        self.food_item = food_item
        self.get_logger().info(f"✅ Detection complete - tracking {food_item}")

        # Publish ready signal
        self.food_detection_ready_pub.publish(Bool(data=True))
        self.currently_serving_pub.publish(String(data=food_item))

        with self.state_lock:
            self.state = DetectionState.TRACKING

    def process_tracking(self):
        """Continuously track with SAM2 and publish position vectors"""
        try:
            # Run SAM2 tracking
            mask, centroid = self.sam2_tracker.track_frame(self.current_image)

            if mask is None or centroid is None:
                self.tracking_failures += 1
                self.get_logger().warn(f"Tracking failure {self.tracking_failures}/{self.max_tracking_failures}")

                if self.tracking_failures >= self.max_tracking_failures:
                    self.get_logger().error("❌ SAM2 tracking lost permanently!")
                    self.transition_to_failed()
                    return
            else:
                # Reset failure count on successful tracking
                self.tracking_failures = 0

                # Calculate and publish position vector
                position_vector = self.calculate_position_vector(centroid)
                if position_vector:
                    self.position_vector_pub.publish(position_vector)

                # Calculate and publish other metrics
                self.publish_food_metrics(mask, centroid)

        except Exception as e:
            self.get_logger().error(f"Tracking error: {e}")
            self.transition_to_failed()

    def process_failure(self):
        """Handle failure state - notify orchestrator and reset"""
        self.get_logger().error("🚨 In FAILED state - notifying orchestrator")

        # Notify orchestrator of tracking loss
        self.tracking_lost_pub.publish(Bool(data=True))

        # Transition back to IDLE and wait for new start command
        with self.state_lock:
            self.state = DetectionState.IDLE

        self.reset_detection_state()

    def transition_to_failed(self):
        """Transition to failed state"""
        with self.state_lock:
            if self.state != DetectionState.FAILED:
                self.state = DetectionState.FAILED

    def get_food_item(self):
        """Get food item from voice commands or ChatGPT"""
        # Check for voice commands first
        with self.voice_lock:
            if self.voice_commands:
                command = self.voice_commands.pop(0)
                self.get_logger().info(f"Using voice command: {command}")
                return command

        # Fall back to ChatGPT identification
        self.get_logger().info("No voice commands, using ChatGPT...")
        try:
            # Your existing ChatGPT logic here
            # For now, return a default item
            return "grape"
        except Exception as e:
            self.get_logger().error(f"ChatGPT identification failed: {e}")
            return None

    def run_dinox_detection(self, food_item):
        """Run DINOX detection for the specified food item"""
        try:
            prompt = f"{food_item} ."
            detections = self.dinox_detector.detect(self.current_image, prompt)

            if not detections or len(detections[0]) == 0:
                self.get_logger().error(f"No {food_item} detected")
                return None

            # Get highest confidence detection
            confidences = detections[1]
            best_idx = np.argmax(confidences)
            best_confidence = confidences[best_idx]

            if best_confidence < 0.3:  # Confidence threshold
                self.get_logger().error(f"Low confidence detection: {best_confidence}")
                return None

            bounding_box = detections[0][best_idx]
            self.get_logger().info(f"DINOX detected {food_item} with confidence {best_confidence:.3f}")

            return bounding_box

        except Exception as e:
            self.get_logger().error(f"DINOX detection error: {e}")
            return None

    def initialize_sam2_tracking(self, bounding_box):
        """Initialize SAM2 tracking with the bounding box"""
        try:
            success = self.sam2_tracker.initialize_tracking(self.current_image, bounding_box)
            if success:
                self.get_logger().info("SAM2 tracking initialized successfully")
                return True
            else:
                self.get_logger().error("SAM2 tracking initialization failed")
                return False
        except Exception as e:
            self.get_logger().error(f"SAM2 initialization error: {e}")
            return False

    def calculate_position_vector(self, centroid):
        """Calculate position vector from centroid"""
        if not self.camera_info or not centroid:
            return None

        try:
            # Convert pixel coordinates to camera coordinates
            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]

            # Get depth at centroid
            u, v = int(centroid[0]), int(centroid[1])
            if (0 <= u < self.current_depth.shape[1] and 0 <= v < self.current_depth.shape[0]):
                depth = self.current_depth[v, u] / 1000.0  # Convert to meters

                if depth > 0:
                    # Convert to 3D coordinates
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth

                    # Create position vector (relative to desired position)
                    # This should be your existing position vector calculation logic
                    position_vector = Vector3()
                    position_vector.x = x  # Adjust based on your coordinate system
                    position_vector.y = y
                    position_vector.z = z

                    return position_vector

        except Exception as e:
            self.get_logger().error(f"Position vector calculation error: {e}")

        return None

    def publish_food_metrics(self, mask, centroid):
        """Publish additional food metrics"""
        try:
            # Calculate grip value (food width)
            grip_value = self.calculate_grip_value(mask)
            if grip_value:
                self.grip_value_pub.publish(Float64(data=grip_value))

            # Calculate food height
            food_height = self.calculate_food_height(centroid)
            if food_height is not None:
                self.food_height_pub.publish(Float64(data=food_height))

            # Calculate food depth
            food_depth = self.calculate_food_depth(centroid)
            if food_depth is not None:
                self.food_depth_pub.publish(Float64(data=food_depth))

        except Exception as e:
            self.get_logger().error(f"Metrics calculation error: {e}")

    def calculate_grip_value(self, mask):
        """Calculate gripper width needed for food"""
        # Your existing grip calculation logic
        try:
            # Find contours and calculate width
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                width = min(rect[1])  # Smaller dimension
                # Convert to gripper units (this needs your calibration)
                return width * 0.001  # Example conversion
        except:
            pass
        return None

    def calculate_food_height(self, centroid):
        """Calculate food height above table"""
        try:
            u, v = int(centroid[0]), int(centroid[1])
            if (0 <= u < self.current_depth.shape[1] and 0 <= v < self.current_depth.shape[0]):
                food_depth = self.current_depth[v, u] / 1000.0
                if food_depth > 0:
                    height = max(0, self.table_depth - food_depth)
                    return height
        except:
            pass
        return None

    def calculate_food_depth(self, centroid):
        """Calculate food depth for pickup verification"""
        try:
            u, v = int(centroid[0]), int(centroid[1])
            if (0 <= u < self.current_depth.shape[1] and 0 <= v < self.current_depth.shape[0]):
                depth = self.current_depth[v, u] / 1000.0
                return depth
        except:
            pass
        return None


def main(args=None):
    rclpy.init(args=args)

    try:
        node = SimplifiedFoodDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()