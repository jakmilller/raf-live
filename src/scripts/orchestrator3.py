#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float64
from geometry_msgs.msg import Vector3
import time
import yaml
import os
import sys
import asyncio
import copy
import numpy as np
from enum import Enum

# Import controllers/checkers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_controller_ros2 import KinovaRobotControllerROS2
from autonomous_checker import AutonomousChecker
from raf_interfaces.srv import GetPose


class FeedingState(Enum):
    RESET_POSITION = "reset_position"
    FOOD_DETECTION = "food_detection"
    SERVOING_TO_FOOD = "servoing_to_food"
    PICKUP_SEQUENCE = "pickup_sequence"
    OVERLOOK_VERIFICATION = "overlook_verification"
    INTERMEDIATE_POSITION = "intermediate_position"
    MOUTH_SERVOING = "mouth_servoing"
    FOOD_REMOVAL_DETECTION = "food_removal_detection"
    CYCLE_COMPLETE = "cycle_complete"
    SCRIPT_END = "script_end"


class SimplifiedOrchestrator(Node):
    def __init__(self):
        super().__init__('simplified_orchestrator')

        # Load configuration
        config_path = os.path.expanduser('~/raf-live/config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize robot controller and autonomous checker
        self.robot_controller = KinovaRobotControllerROS2(config_path)
        self.autonomous_checker = AutonomousChecker(self.config)

        # State management
        self.current_state = FeedingState.RESET_POSITION
        self.current_food_item = None
        self.retry_count = 0
        self.max_retries = 3

        # Thresholds from config
        self.position_vector_threshold = self.config['feeding']['position_vector_threshold']
        self.servoing_timeout = self.config['feeding']['servoing_timeout']
        self.distance_from_target = self.config['feeding']['dist_from_food']
        self.face_detection_enabled = self.config['feeding'].get('face_detection', {}).get('enabled', False)

        # ROS subscribers for monitoring
        self.position_vector_sub = self.create_subscription(
            Vector3, '/position_vector', self.position_vector_callback, 1)
        self.food_detection_ready_sub = self.create_subscription(
            Bool, '/food_detection_ready', self.food_detection_ready_callback, 1)
        self.tracking_lost_sub = self.create_subscription(
            Bool, '/tracking_lost', self.tracking_lost_callback, 1)
        self.currently_serving_sub = self.create_subscription(
            String, '/currently_serving', self.currently_serving_callback, 1)
        self.food_height_sub = self.create_subscription(
            Float64, '/food_height', self.food_height_callback, 1)
        self.grip_value_sub = self.create_subscription(
            Float64, '/grip_value', self.grip_value_callback, 1)
        self.food_angle_sub = self.create_subscription(
            Float64, '/food_angle', self.food_angle_callback, 1)

        # ROS publishers for control
        self.start_food_detection_pub = self.create_publisher(Bool, '/start_food_detection', 1)
        self.servoing_on_pub = self.create_publisher(Bool, '/servoing_on', 1)
        self.start_face_detection_pub = self.create_publisher(Bool, '/start_face_detection', 1)
        self.twist_gains_pub = self.create_publisher(Vector3, '/twist_gains', 1)

        # State variables
        self.latest_position_vector = None
        self.food_detection_ready = False
        self.tracking_lost = False
        self.food_height = None
        self.grip_value = None
        self.food_angle = None

        # Timing variables
        self.state_start_time = None

        # Service client for getting current pose
        self.get_pose_client = self.create_client(GetPose, '/my_gen3/get_pose')

        self.get_logger().info('Simplified Orchestrator initialized - ready to start feeding cycle')

    async def run_feeding_cycle(self):
        """Main feeding cycle - runs continuously"""
        self._enter_state(FeedingState.RESET_POSITION)

        while rclpy.ok():
            await self.state_machine()
            await asyncio.sleep(0.1)  # 10Hz

    def _enter_state(self, new_state):
        """Enter a new state with proper logging and timing"""
        self.get_logger().info(f"Entering state: {new_state.value}")
        self.current_state = new_state
        self.state_start_time = time.time()

        # Reset state-specific flags when entering new states
        if new_state == FeedingState.FOOD_DETECTION:
            self.food_detection_ready = False
            self.tracking_lost = False
            # Reset detection started flag so it starts detection again
            if hasattr(self, '_detection_started_this_cycle'):
                delattr(self, '_detection_started_this_cycle')
            self.get_logger().info("Reset food_detection_ready flag for new detection cycle")
        elif new_state == FeedingState.SERVOING_TO_FOOD:
            self.tracking_lost = False

    async def state_machine(self):
        """Main state machine - linear feeding cycle"""
        current_time = time.time()

        if self.current_state == FeedingState.RESET_POSITION:
            await self._handle_reset_position()

        elif self.current_state == FeedingState.FOOD_DETECTION:
            await self._handle_food_detection()

        elif self.current_state == FeedingState.SERVOING_TO_FOOD:
            await self._handle_servoing_to_food(current_time)

        elif self.current_state == FeedingState.PICKUP_SEQUENCE:
            await self._handle_pickup_sequence()

        elif self.current_state == FeedingState.OVERLOOK_VERIFICATION:
            await self._handle_overlook_verification()

        elif self.current_state == FeedingState.INTERMEDIATE_POSITION:
            await self._handle_intermediate_position()

        elif self.current_state == FeedingState.MOUTH_SERVOING:
            await self._handle_mouth_servoing(current_time)

        elif self.current_state == FeedingState.FOOD_REMOVAL_DETECTION:
            self._handle_food_removal_detection()

        elif self.current_state == FeedingState.CYCLE_COMPLETE:
            self._handle_cycle_complete()

        elif self.current_state == FeedingState.SCRIPT_END:
            await self._handle_script_end()

    async def _handle_reset_position(self):
        """Move robot to reset position and proceed to food detection"""
        self.get_logger().info("Moving to reset position...")

        # Move to reset position (overlook)
        success = await self.robot_controller.reset()

        if success:
            # Open gripper to default position (0.5)
            self.get_logger().info("Opening gripper to default position (0.5)")
            await self.robot_controller.set_gripper(0.5)

            # Ensure servoing is explicitly disabled during reset
            self.servoing_on_pub.publish(Bool(data=False))
            await asyncio.sleep(0.5)  # Allow time for servoing to properly disable

            self.get_logger().info("Reset position reached - starting food detection")
            self._enter_state(FeedingState.FOOD_DETECTION)
        else:
            self.get_logger().error("Failed to reach reset position - retrying in 1 second")
            await asyncio.sleep(1.0)

    async def _handle_food_detection(self):
        """Start food detection and wait for ready signal"""
        # Start food detection every time we're in this state (only runs from reset position)
        if not hasattr(self, '_detection_started_this_cycle'):
            self.get_logger().info("Starting food detection for new cycle")
            self.start_food_detection_pub.publish(Bool(data=True))

            # Set gains for food detection/servoing (reset from face detection gains)
            food_gains = Vector3(x=0.5, y=0.5, z=0.5)
            self.twist_gains_pub.publish(food_gains)

            self._detection_started_this_cycle = True

        # Wait for food detection to be ready
        if self.food_detection_ready:
            if self.current_food_item == "none":
                self.get_logger().info("No more food items - ending script")
                self._enter_state(FeedingState.SCRIPT_END)
            else:
                self.get_logger().info(f"Food detection ready for: {self.current_food_item}")
                # Wait before enabling servoing to ensure proper sequencing
                await asyncio.sleep(0.1)
                self.get_logger().info("Enabling servoing for food detection")
                self.servoing_on_pub.publish(Bool(data=True))
                self._enter_state(FeedingState.SERVOING_TO_FOOD)

        # Check for tracking lost during detection
        if self.tracking_lost:
            self.get_logger().warn("Tracking lost during food detection - restarting cycle")
            self.servoing_on_pub.publish(Bool(data=False))
            await asyncio.sleep(0.1)  # Ensure servoing stops before restart
            await self._restart_complete_cycle()

    async def _handle_servoing_to_food(self, current_time):
        """Servo to food with timeout and small position vector detection"""
        # Ensure servoing is enabled (important for retries)
        self.get_logger().info("Publishing servoing_on=True for food servoing")
        self.servoing_on_pub.publish(Bool(data=True))

        # Check for tracking lost
        if self.tracking_lost:
            self.get_logger().warn("Tracking lost during servoing - restarting cycle")
            self.servoing_on_pub.publish(Bool(data=False))
            await asyncio.sleep(0.1)  # Ensure servoing stops before restart
            await self._restart_complete_cycle()
            return

        # Check for timeout
        if current_time - self.state_start_time > self.servoing_timeout:
            self.get_logger().warn("Servoing timeout - attempting retry")
            self.servoing_on_pub.publish(Bool(data=False))
            await asyncio.sleep(0.1)  # Ensure servoing stops before retry
            await self._handle_retry("servoing timeout")
            return

        # Check if robot reached food (small position vector)
        if self._is_small_position_vector():
            self.get_logger().info("Robot reached food - stopping servoing and starting pickup")
            self.servoing_on_pub.publish(Bool(data=False))
            # Add small delay to ensure servoing stops before pickup
            await asyncio.sleep(0.1)
            self._enter_state(FeedingState.PICKUP_SEQUENCE)

    async def _handle_pickup_sequence(self):
        """Execute pickup sequence and verify success"""
        self.get_logger().info("Executing pickup sequence...")

        # Stop food detection during pickup
        self.start_food_detection_pub.publish(Bool(data=False))
        await asyncio.sleep(0.1)  # Allow detection to stop properly

        # Execute pickup using stored values
        if self.food_height is not None and self.grip_value is not None:
            success = await self._execute_pickup_sequence()

            if success:
                self.get_logger().info("Pickup sequence completed successfully")
                self._enter_state(FeedingState.OVERLOOK_VERIFICATION)
            else:
                self.get_logger().warn("Pickup sequence failed - attempting retry")
                await self._handle_retry("pickup sequence failed")
        else:
            self.get_logger().error("Missing pickup parameters - attempting retry")
            await self._handle_retry("missing pickup parameters")

    async def _execute_pickup_sequence(self):
        """Execute the physical pickup sequence"""
        try:
            if self.tracking_lost:
                self.get_logger().error("Cannot execute pickup - tracking lost!")
                return False

            # Set gripper to food width
            self.get_logger().info(f"Setting gripper to food width: {self.grip_value:.3f}")
            if not await self.robot_controller.set_gripper(self.grip_value):
                self.get_logger().error("Failed to set gripper for pickup!")
                return False

            await asyncio.sleep(0.3)  # Allow gripper to settle

            # Get current pose and move down
            current_pose = await self.get_current_pose()
            if not current_pose:
                return False

            pickup_pose = copy.deepcopy(current_pose)
            move_down_distance = self.distance_from_target + self.food_height
            pickup_pose.position.z -= move_down_distance

            self.get_logger().info(f"Moving down {move_down_distance:.4f}m for pickup")
            if not await self.robot_controller.move_to_pose(pickup_pose):
                self.get_logger().error("Failed to move down for pickup!")
                return False

            # Close gripper more
            grip_close_amount = self.config['feeding']['grip_close']
            close_value = min(1.0, self.grip_value + grip_close_amount)
            self.get_logger().info(f"Closing gripper from {self.grip_value:.3f} to {close_value:.3f}")
            if not await self.robot_controller.set_gripper(close_value):
                self.get_logger().error("Failed to close gripper!")
                return False

            await asyncio.sleep(0.3)  # Allow gripper to close properly

            # Move back to overlook
            self.get_logger().info("Moving back to overlook after pickup...")
            if not await self.robot_controller.reset():
                self.get_logger().error("Failed to move back to overlook!")
                return False

            return True

        except Exception as e:
            self.get_logger().error(f"Error in pickup sequence: {e}")
            return False

    async def _handle_overlook_verification(self):
        """Move to overlook and verify pickup success"""
        self.get_logger().info("Verifying pickup success...")

        # Robot is already at overlook from pickup sequence
        # Wait for robot to settle
        await asyncio.sleep(0.5)

        # Restart food detection temporarily to get current depth images
        self.start_food_detection_pub.publish(Bool(data=True))
        await asyncio.sleep(0.7)  # Let detection initialize properly (match original timing)

        # Spin both nodes to get latest depth data
        self.get_logger().info("Checking if food was picked up using depth images...")
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)
            rclpy.spin_once(self.autonomous_checker, timeout_sec=0.1)
            await asyncio.sleep(0.1)

        # Use autonomous checker to verify pickup
        pickup_successful = self.autonomous_checker.check_object_grasped()

        # Stop food detection again after verification
        self.start_food_detection_pub.publish(Bool(data=False))
        await asyncio.sleep(0.1)  # Ensure detection stops properly

        if pickup_successful:
            self.get_logger().info("Pickup verified successful - proceeding to intermediate position")
            self._enter_state(FeedingState.INTERMEDIATE_POSITION)
        else:
            self.get_logger().warn("Pickup verification failed - attempting retry")
            await self._handle_retry("pickup verification failed")

    async def _handle_intermediate_position(self):
        """Move to intermediate position"""
        self.get_logger().info("Moving to intermediate position...")

        success = await self.robot_controller.move_to_intermediate()

        if success:
            self.get_logger().info("Intermediate position reached - starting mouth servoing")
            self._enter_state(FeedingState.MOUTH_SERVOING)
        else:
            self.get_logger().error("Failed to reach intermediate position - attempting retry")
            await self._handle_retry("failed to reach intermediate position")

    async def _handle_mouth_servoing(self, current_time):
        """Start face detection and servo to mouth with timeout"""
        if self.face_detection_enabled:
            self.get_logger().info("=== Moving food to mouth using face detection ===")

            # Start face detection
            self.get_logger().info("1. Starting face detection...")
            self.start_face_detection_pub.publish(Bool(data=True))

            # Set gains for face servoing
            self.get_logger().info("2. Setting face servoing gains...")
            face_gains = Vector3(x=0.35, y=0.35, z=0.35)
            self.twist_gains_pub.publish(face_gains)
            await asyncio.sleep(0.5)

            # Start servoing
            self.get_logger().info("3. Starting face servoing...")
            self.servoing_on_pub.publish(Bool(data=True))

            # Wait for robot to reach mouth
            self.get_logger().info("4. Waiting for robot to reach mouth...")
            success = await self.wait_for_small_position_vector(threshold=0.04, timeout=60.0)

            # Stop face detection and servoing
            self.get_logger().info("5. Stopping face detection and servoing...")
            self.servoing_on_pub.publish(Bool(data=False))
            await asyncio.sleep(0.5)  # Critical delay to ensure servoing node processes disable command
            self.start_face_detection_pub.publish(Bool(data=False))

            if success:
                self.get_logger().info("Face detection servoing completed successfully!")
                self._enter_state(FeedingState.FOOD_REMOVAL_DETECTION)
            else:
                self.get_logger().warn("Could not move to mouth! Proceeding to food removal anyway...")
                self._enter_state(FeedingState.FOOD_REMOVAL_DETECTION)
        else:
            # Face detection disabled - skip directly to food removal
            self.get_logger().info("Face detection disabled - proceeding directly to food removal")
            self._enter_state(FeedingState.FOOD_REMOVAL_DETECTION)

    def _handle_food_removal_detection(self):
        """Wait for food removal and complete cycle"""
        self.get_logger().info("Waiting for food removal...")

        # Use autonomous checker to detect food removal
        food_removed = self.autonomous_checker.check_object_removed()

        if food_removed:
            self.get_logger().info("Food removal detected - cycle complete")
            self._enter_state(FeedingState.CYCLE_COMPLETE)
        else:
            # Continue waiting (autonomous checker handles timing internally)
            pass

    def _handle_cycle_complete(self):
        """Complete current cycle and restart"""
        self.get_logger().info("Feeding cycle completed successfully - restarting")
        self.retry_count = 0  # Reset retry count for new cycle
        self._enter_state(FeedingState.RESET_POSITION)

    async def _handle_script_end(self):
        """End script gracefully"""
        self.get_logger().info("No more food items - script ending")

        # Stop all detection
        self.start_food_detection_pub.publish(Bool(data=False))
        self.start_face_detection_pub.publish(Bool(data=False))
        self.servoing_on_pub.publish(Bool(data=False))

        # Move to final position
        await self.robot_controller.reset()

        # Shutdown
        rclpy.shutdown()

    async def _handle_retry(self, reason):
        """Handle retry logic with max attempts"""
        self.retry_count += 1
        self.get_logger().info(f"Retry {self.retry_count}/{self.max_retries} due to: {reason}")

        if self.retry_count >= self.max_retries:
            self.get_logger().error(f"Max retries ({self.max_retries}) reached - restarting complete cycle")
            await self._restart_complete_cycle()
        else:
            # Retry from reset position (same food segment)
            self._enter_state(FeedingState.RESET_POSITION)

    async def _restart_complete_cycle(self):
        """Restart complete cycle (new food segment)"""
        self.get_logger().info("Restarting complete feeding cycle")
        self.retry_count = 0
        self.current_food_item = None

        # Stop all detection and servoing with proper timing
        self.start_food_detection_pub.publish(Bool(data=False))
        self.start_face_detection_pub.publish(Bool(data=False))
        self.servoing_on_pub.publish(Bool(data=False))
        await asyncio.sleep(0.5)  # Ensure all systems stop properly before restart

        self._enter_state(FeedingState.RESET_POSITION)

    def _is_small_position_vector(self):
        """Check if position vector magnitude is below threshold"""
        if self.latest_position_vector is None:
            return False

        magnitude = (self.latest_position_vector.x**2 +
                    self.latest_position_vector.y**2 +
                    self.latest_position_vector.z**2)**0.5

        return magnitude < self.position_vector_threshold

    async def wait_for_small_position_vector(self, threshold=0.004, timeout=30.0):
        """Wait until position vector magnitude is small (robot reached target)"""
        self.get_logger().info(f"Waiting for position vector < {threshold}m...")

        start_time = time.time()
        stable_count = 0
        required_stable_count = 8

        while rclpy.ok():
            # Check if tracking was lost during servoing
            if self.tracking_lost:
                self.get_logger().error("Tracking lost during servoing!")
                return False

            if time.time() - start_time > timeout:
                self.get_logger().warn(f"Timeout waiting for small position vector after {timeout}s")
                return False

            # Check position vector magnitude using the actual position vector data
            if self.latest_position_vector is not None:
                magnitude = np.linalg.norm([
                    self.latest_position_vector.x,
                    self.latest_position_vector.y,
                    self.latest_position_vector.z
                ])

                if magnitude < threshold:
                    stable_count += 1
                    if stable_count >= required_stable_count:
                        self.get_logger().info(f"Position vector stable at {magnitude:.4f}m")
                        return True
                else:
                    stable_count = 0

            await asyncio.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0)

        return False

    async def get_current_pose(self):
        """Get current pose using the GetPose service"""
        try:
            self.get_logger().info("Calling GetPose service at /my_gen3/get_pose...")

            if not self.get_pose_client.service_is_ready():
                self.get_logger().error("GetPose service is not ready - controller may have disconnected")
                return None

            request = GetPose.Request()
            future = self.get_pose_client.call_async(request)

            # Wait for the service call to complete with timeout
            timeout = 10.0  # seconds
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout:
                    self.get_logger().error("GetPose service call timed out")
                    return None
                await asyncio.sleep(0.1)
                rclpy.spin_once(self, timeout_sec=0)

            response = future.result()
            if response and response.success:
                self.get_logger().info("Successfully retrieved current pose")
                return response.current_pose
            else:
                self.get_logger().error("GetPose service call failed")
                return None

        except Exception as e:
            self.get_logger().error(f"Exception in get_current_pose: {e}")
            return None

    # ROS callback functions
    def position_vector_callback(self, msg):
        self.latest_position_vector = msg

    def food_detection_ready_callback(self, msg):
        self.get_logger().info(f"Received food_detection_ready: {msg.data}")
        self.food_detection_ready = msg.data

    def tracking_lost_callback(self, msg):
        if msg.data:
            self.tracking_lost = True

    def currently_serving_callback(self, msg):
        self.get_logger().info(f"Received currently_serving: {msg.data}")
        self.current_food_item = msg.data

    def food_height_callback(self, msg):
        self.food_height = msg.data

    def grip_value_callback(self, msg):
        self.grip_value = msg.data

    def food_angle_callback(self, msg):
        self.food_angle = msg.data


async def main():
    rclpy.init()

    try:
        orchestrator = SimplifiedOrchestrator()

        # Create an executor to run in the background
        from rclpy.executors import MultiThreadedExecutor
        import threading

        executor = MultiThreadedExecutor()
        executor.add_node(orchestrator)

        # Start spinning in a separate thread
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        # Start the feeding cycle
        await orchestrator.run_feeding_cycle()

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    asyncio.run(main())