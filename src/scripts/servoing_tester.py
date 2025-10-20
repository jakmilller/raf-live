#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import os

# Add detection package to path
sys.path.append(os.path.expanduser('~/raf-live/src/detection/detection'))
from tracking import SAM2Tracker


class ServoingTester(Node):
    def __init__(self):
        super().__init__('servoing_tester')

        # ROS setup
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.camera_info = None

        # State variables
        self.sam2_tracker = SAM2Tracker(self)
        self.tracking_active = False
        self.servoing_enabled = False
        self.click_point = None
        self.angle_history = []  # Initialize angle history for moving average

        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 1)

        # Publishers
        self.food_angle_pub = self.create_publisher(Float64, '/food_angle', 1)
        self.servoing_on_pub = self.create_publisher(Bool, '/servoing_on', 1)
        self.position_vector_pub = self.create_publisher(Vector3, '/position_vector', 1)

        # Timer for tracking updates (10 Hz)
        self.timer = None

        self.get_logger().info('Servoing Tester initialized')
        self.get_logger().info('Waiting for camera image...')

    def color_callback(self, msg):
        """Store latest color image"""
        self.latest_color_image = msg

    def camera_info_callback(self, msg):
        """Store camera info"""
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info('Camera info received')

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to initialize tracking"""
        if event == cv2.EVENT_LBUTTONDOWN and not self.tracking_active:
            self.click_point = (x, y)
            self.get_logger().info(f'Clicked at point: ({x}, {y})')

            # Initialize SAM2 tracking with the clicked point
            frame = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            success = self.sam2_tracker.initialize_tracking(frame, self.click_point, 'gemini')

            if success:
                self.tracking_active = True
                self.get_logger().info('SAM2 tracking initialized successfully!')

                # Ask user if they want to enable servoing
                self.prompt_servoing()

                # Start tracking timer
                if self.timer:
                    self.timer.cancel()
                self.timer = self.create_timer(0.1, self.update_tracking)  # 10 Hz
            else:
                self.get_logger().error('Failed to initialize SAM2 tracking')

    def prompt_servoing(self):
        """Prompt user to enable servoing in a separate thread"""
        import threading

        def ask_servoing():
            response = input("\nDo you want to perform orientation servoing? (y/n): ").strip().lower()
            if response == 'y' or response == 'yes':
                self.servoing_enabled = True
                self.servoing_on_pub.publish(Bool(data=True))
                self.get_logger().info('Servoing ENABLED - publishing food angles')
            else:
                self.servoing_enabled = False
                self.servoing_on_pub.publish(Bool(data=False))
                self.get_logger().info('Servoing DISABLED - only showing visualization')

        thread = threading.Thread(target=ask_servoing, daemon=True)
        thread.start()

    def update_tracking(self):
        """Update SAM2 tracking and publish food angle"""
        if not self.tracking_active or self.latest_color_image is None:
            return

        try:
            # Get current frame
            frame = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")

            # Update tracking
            mask_2d, centroid = self.sam2_tracker.update_tracking(frame)

            if mask_2d is None or centroid is None:
                self.get_logger().warn('Tracking lost!')
                self.tracking_active = False
                if self.timer:
                    self.timer.cancel()
                    self.timer = None
                cv2.destroyAllWindows()
                return

            # Calculate food angle using PCA (same logic as GraspAnalyzer)
            food_angle = self._get_food_angle_pca(mask_2d)

            # Publish food angle
            self.food_angle_pub.publish(Float64(data=food_angle))

            # Publish dummy position vector (all zeros) so servoing node doesn't complain
            dummy_vector = Vector3(x=0.0, y=0.0, z=0.0)
            self.position_vector_pub.publish(dummy_vector)

            # Create visualization
            vis_image = self._create_visualization(frame, mask_2d, centroid, food_angle)

            # Display image
            cv2.imshow('Servoing Tester', vis_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in tracking update: {e}')

    def _get_food_angle_pca(self, mask):
        """Calculate food angle using PCA (same as GraspAnalyzer)"""
        mask = self.smooth_mask(mask, kernel_size=9, sigma=2.0)
        ys, xs = np.where(mask > 0)
        points = np.column_stack((xs, ys))

        if points.shape[0] < 2:
            return 0.0

        # Mean center the data
        mean = np.mean(points, axis=0)
        centered = points - mean

        # Compute covariance and eigenvectors
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Check if object is elongated enough
        # eigenvalue_ratio = eigvals[0] / eigvals[1] if eigvals[1] > 0 else float('inf')
        # elongation_threshold = 1.3

        # if eigenvalue_ratio < elongation_threshold:
        #     return 0.0

        # Calculate angle from major axis
        major_axis = eigvecs[:, 0]
        angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

        # Wrap angle to correct range
        if -180 <= angle <= -45:
            angle += 180

        # Make vertical 0 degrees
        angle -= 90

        # map angle to nearest whole number, but keep it the same data type
        # angle = float(round(angle))

        # apply moving average of last 5 angles to smooth out noise
        self.angle_history.append(angle)
        if len(self.angle_history) > 10:
            self.angle_history.pop(0)
        angle = np.mean(self.angle_history)

        return angle

    def _create_visualization(self, frame, mask, centroid, food_angle):
        """Create visualization with mask overlay, centroid, angle, and major axis"""
        vis_image = frame.copy()

        # Apply mask overlay
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        vis_image = cv2.addWeighted(vis_image, 1, mask_3d, 0.5, 0)

        # Draw centroid
        cv2.circle(vis_image, centroid, 5, (255, 0, 0), -1)

        # Draw major axis line (PCA direction)
        # Calculate the major axis direction for visualization
        ys, xs = np.where(mask > 0)
        points = np.column_stack((xs, ys))

        if points.shape[0] >= 2:
            mean = np.mean(points, axis=0)
            centered = points - mean
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]

            # Draw major axis line
            major_axis = eigvecs[:, 0]
            line_length = 100
            pt1 = (int(centroid[0] - major_axis[0] * line_length),
                   int(centroid[1] - major_axis[1] * line_length))
            pt2 = (int(centroid[0] + major_axis[0] * line_length),
                   int(centroid[1] + major_axis[1] * line_length))
            cv2.line(vis_image, pt1, pt2, (0, 255, 255), 2)

        # Display angle text in top-left corner
        angle_text = f"Food Angle: {food_angle:.1f} deg"
        servoing_text = f"Servoing: {'ON' if self.servoing_enabled else 'OFF'}"

        # White background for text
        cv2.putText(vis_image, angle_text, (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(vis_image, servoing_text, (10, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)

        # Black text on top
        cv2.putText(vis_image, angle_text, (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis_image, servoing_text, (10, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0) if self.servoing_enabled else (0, 0, 255), 2, cv2.LINE_AA)

        return vis_image
    
    def smooth_mask(self, mask, kernel_size=5, sigma=1.0):
        return cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)

    def run(self):
        """Main run loop - show initial image and wait for click"""
        self.get_logger().info('Waiting for camera image...')

        # Wait for first image
        while self.latest_color_image is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

        if not rclpy.ok():
            return

        self.get_logger().info('Camera image received! Displaying popup...')
        self.get_logger().info('Click on the food item to start tracking')

        # Show initial image
        frame = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
        cv2.imshow('Servoing Tester', frame)
        cv2.setMouseCallback('Servoing Tester', self.mouse_callback)

        # Main loop
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)

                # If not tracking, update the display with latest image
                if not self.tracking_active and self.latest_color_image is not None:
                    frame = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
                    cv2.imshow('Servoing Tester', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC to quit
                    self.get_logger().info('Quitting...')
                    break
                elif key == ord('r'):  # r to reset
                    self.get_logger().info('Resetting tracking...')
                    self.tracking_active = False
                    self.servoing_enabled = False
                    self.servoing_on_pub.publish(Bool(data=False))
                    self.sam2_tracker.reset_tracking()
                    self.angle_history = []  # Clear angle history on reset
                    if self.timer:
                        self.timer.cancel()
                        self.timer = None

        except KeyboardInterrupt:
            self.get_logger().info('Interrupted by user')
        finally:
            # Cleanup
            self.servoing_on_pub.publish(Bool(data=False))
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    tester = ServoingTester()

    try:
        tester.run()
    except Exception as e:
        tester.get_logger().error(f'Error: {e}')
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
