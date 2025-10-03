#!/usr/bin/env python3

"""
Clean Orchestrator for Simplified Food Detection

This orchestrator works with the simplified food detection node that has clean state management.
Key improvements:
- Proper detection start/stop synchronization
- Clean tracking loss handling
- No race conditions in detection restart
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Pose
from std_msgs.msg import Float64, Bool, String
import asyncio
import sys
import yaml
import time
import copy
import numpy as np
import os
import pygame

# Import controllers/checkers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_controller_ros2 import KinovaRobotControllerROS2
from autonomous_checker import AutonomousChecker
from raf_interfaces.srv import GetPose


class CleanOrchestrator(Node):
    def __init__(self):
        super().__init__('clean_orchestrator')

        config_path = os.path.expanduser('~/raf-live/config.yaml')

        # Load config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize robot controller and autonomous checker
        self.robot_controller = KinovaRobotControllerROS2(config_path)
        self.autonomous_checker = AutonomousChecker(self.config)

        # Config parameters
        self.distance_from_target = self.config['feeding']['dist_from_food']
        self.grip_close_amount = self.config['feeding']['grip_close']
        self.face_detection_enabled = self.config['feeding'].get('face_detection', {}).get('enabled', False)

        # audio setup
        pygame.mixer.init()
        self.power_on = self.config['feeding']['power_on']
        self.power_off = self.config['feeding']['power_off']
        self.snap = self.config['feeding']['snap']
        self.play_sound(self.power_on)

        # State variables
        self.latest_position_vector = Vector3()
        self.latest_grip_value = None
        self.latest_food_height = None
        self.latest_food_depth = None
        self.food_detection_ready = False
        self.vector_pause = False
        self.tracking_lost = False
        self.detection_active = False  # Track if detection is currently active

        # Subscribers
        self.position_vector_sub = self.create_subscription(
            Vector3, '/position_vector', self.position_vector_callback, 1)
        self.grip_value_sub = self.create_subscription(
            Float64, '/grip_value', self.grip_value_callback, 1)
        self.food_height_sub = self.create_subscription(
            Float64, '/food_height', self.food_height_callback, 1)
        self.food_detection_ready_sub = self.create_subscription(
            Bool, '/food_detection_ready', self.food_detection_ready_callback, 1)
        self.vector_pause_sub = self.create_subscription(
            Bool, '/vector_pause', self.vector_pause_callback, 1)
        self.food_item_sub = self.create_subscription(
            String, '/currently_serving', self.food_item_callback, 1)
        self.tracking_lost_sub = self.create_subscription(
            Bool, '/tracking_lost', self.tracking_lost_callback, 1)
        self.food_depth_sub = self.create_subscription(
            Float64, '/food_depth', self.food_depth_callback, 1)

        # Publishers - simple control interface
        self.start_food_detection_pub = self.create_publisher(Bool, '/start_food_detection', 1)
        self.start_face_detection_pub = self.create_publisher(Bool, '/start_face_detection', 1)
        self.robot_state_pub = self.create_publisher(String, '/robot_state', 1)
        self.servoing_on_pub = self.create_publisher(Bool, '/servoing_on', 1)
        self.twist_gains_pub = self.create_publisher(Vector3, '/twist_gains', 1)
        self.food_angle_pub = self.create_publisher(Float64, '/food_angle', 1)

        # Service client for getting current pose
        self.get_pose_client = self.create_client(GetPose, '/my_gen3/get_pose')
        while not self.get_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /my_gen3/get_pose service...')

        self.get_logger().info(f'Clean Orchestrator initialized!')
        self.get_logger().info(f'Face detection enabled: {self.face_detection_enabled}')

    def position_vector_callback(self, msg):
        """Store latest position vector"""
        self.latest_position_vector = msg

    def grip_value_callback(self, msg):
        """Store latest grip value"""
        self.latest_grip_value = msg.data

    def food_height_callback(self, msg):
        """Store latest food height"""
        self.latest_food_height = msg.data

    def food_item_callback(self, msg):
        """Adjust close values based on food item"""
        if msg.data == "carrot":
            self.grip_close_amount = 0.075
            self.get_logger().info("Adjusting close value for carrot!")
        elif msg.data == "chicken nugget" or msg.data == "cut grilled chicken":
            self.grip_close_amount = 0.065
            self.get_logger().info("Adjusting close value for chicken!")
        else:
            self.grip_close_amount = self.config['feeding']['grip_close']

    def vector_pause_callback(self, msg):
        """Handle vector pause signal"""
        self.vector_pause = msg.data

    def food_detection_ready_callback(self, msg):
        """Track when food detection is ready"""
        self.food_detection_ready = msg.data
        if msg.data:
            self.get_logger().info("✅ Food detection ready - tracking initialized!")

    def tracking_lost_callback(self, msg):
        """Handle tracking lost signal - IMMEDIATE RESPONSE"""
        if msg.data:
            self.tracking_lost = True
            self.get_logger().error("🚨 TRACKING LOST - Will restart feeding cycle!")
            # Immediately stop any ongoing servoing
            self.servoing_on_pub.publish(Bool(data=False))

    def food_depth_callback(self, msg):
        """Store latest food depth"""
        self.latest_food_depth = msg.data

    def play_sound(self, file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    async def start_detection_cleanly(self):
        """Start food detection with proper cleanup"""
        self.get_logger().info("🔍 Starting food detection cleanly...")

        # Reset all state
        self.food_detection_ready = False
        self.tracking_lost = False
        self.latest_grip_value = None
        self.latest_food_height = None

        # Ensure any previous detection is stopped
        if self.detection_active:
            self.get_logger().info("Stopping previous detection session...")
            self.start_food_detection_pub.publish(Bool(data=False))
            await asyncio.sleep(1.0)  # Give time for clean shutdown

        # Start new detection
        self.start_food_detection_pub.publish(Bool(data=True))
        self.detection_active = True

    async def stop_detection_cleanly(self):
        """Stop food detection with proper cleanup"""
        if self.detection_active:
            self.get_logger().info("⏹️ Stopping food detection cleanly...")
            self.start_food_detection_pub.publish(Bool(data=False))
            self.detection_active = False
            await asyncio.sleep(0.5)  # Give time for clean shutdown

    async def wait_for_food_detection_ready(self, timeout=30.0):
        """Wait for food detection to be ready"""
        self.get_logger().info("⏳ Waiting for food detection to complete...")

        start_time = time.time()

        while rclpy.ok():

            if time.time() - start_time > timeout:
                self.get_logger().error(f"⏰ Timeout waiting for detection after {timeout}s")
                return False

            if self.food_detection_ready:
                self.get_logger().info("✅ Food detection completed successfully!")
                return True

            await asyncio.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0)

        return False

    async def wait_for_small_position_vector(self, threshold=0.004, timeout=30.0):
        """Wait until position vector magnitude is small (robot reached target)"""
        self.get_logger().info(f"⏳ Waiting for position vector < {threshold}m...")

        start_time = time.time()
        stable_count = 0
        required_stable_count = 8

        while rclpy.ok():
            # Check for tracking loss during servoing
            if self.tracking_lost:
                self.get_logger().error("❌ Tracking lost during servoing!")
                return False

            if time.time() - start_time > timeout:
                self.get_logger().warn(f"⏰ Timeout waiting for small position vector after {timeout}s")
                return False

            # Check position vector magnitude
            magnitude = np.linalg.norm([
                self.latest_position_vector.x,
                self.latest_position_vector.y,
                self.latest_position_vector.z
            ])

            if magnitude < threshold and not self.vector_pause:
                stable_count += 1
                if stable_count >= required_stable_count:
                    self.get_logger().info(f"✅ Position vector consistently small: {magnitude:.3f}m")
                    return True
            else:
                stable_count = 0

            await asyncio.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0)

        return False

    async def get_current_pose(self):
        """Get current pose using the GetPose service"""
        try:
            if not self.get_pose_client.service_is_ready():
                self.get_logger().error("GetPose service is not ready")
                return None

            request = GetPose.Request()
            future = self.get_pose_client.call_async(request)

            start_time = time.time()
            timeout = 5.0

            while not future.done() and rclpy.ok():
                if time.time() - start_time > timeout:
                    self.get_logger().error(f'GetPose service call timed out')
                    return None

                rclpy.spin_once(self, timeout_sec=0.1)
                await asyncio.sleep(0.01)

            response = future.result()

            if response.success:
                return response.current_pose
            else:
                self.get_logger().error(f'GetPose service returned failure: {response.message}')
                return None

        except Exception as e:
            self.get_logger().error(f'Error calling get pose service: {e}')
            return None

    async def acquire_food_with_retries(self, max_retries=3):
        """Attempt to acquire food with clean retry logic"""

        for attempt in range(max_retries):
            self.robot_state_pub.publish(String(data='Detecting food item...'))
            self.get_logger().info(f"=== Food acquisition attempt {attempt + 1}/{max_retries} ===")

            # Step 1: Start detection cleanly
            await self.start_detection_cleanly()

            # Step 2: Set gains for servoing
            self.get_logger().info("Setting servoing gains...")
            gains = Vector3(x=0.5, y=0.5, z=0.5)
            self.twist_gains_pub.publish(gains)

            # Step 3: Wait for detection to complete
            if not await self.wait_for_food_detection_ready(timeout=40.0):
                self.get_logger().error("❌ Detection failed")
                await self.stop_detection_cleanly()

                if self.tracking_lost:
                    self.get_logger().error("Detection failed due to tracking loss - restarting cycle")
                    return False

                continue  # Try next attempt

            # Step 4: Start servoing
            self.get_logger().info("🎯 Starting servoing...")
            self.robot_state_pub.publish(String(data='Moving to food item...'))
            self.servoing_on_pub.publish(Bool(data=True))

            # Step 5: Wait for robot to reach food
            success = await self.wait_for_small_position_vector(threshold=0.005, timeout=30.0)

            if not success:
                self.servoing_on_pub.publish(Bool(data=False))

                if self.tracking_lost:
                    self.get_logger().error("❌ Servoing failed due to tracking loss - restarting cycle")
                    await self.stop_detection_cleanly()
                    return False

                self.get_logger().error("❌ Failed to reach food - timeout")
                continue

            # Step 6: Stop servoing for pickup
            self.get_logger().info("🎯 Reached food! Stopping servoing for pickup...")
            self.servoing_on_pub.publish(Bool(data=False))
            await asyncio.sleep(0.5)

            # Step 7: Execute pickup sequence
            self.robot_state_pub.publish(String(data='Executing pickup sequence'))
            if not await self.execute_pickup_sequence():
                self.get_logger().warn(f"❌ Pickup sequence failed on attempt {attempt + 1}")
                continue

            # Step 8: Check if food was picked up
            self.get_logger().info("🔍 Checking if food was picked up...")
            await asyncio.sleep(0.5)
            pickup_success = self.wait_for_food_pickup_from_segment_depth()

            if pickup_success:
                self.get_logger().info("✅ Food pickup successful!")
                return True
            else:
                self.get_logger().warn(f"❌ Food pickup failed on attempt {attempt + 1}")
                if self.tracking_lost:
                    await self.stop_detection_cleanly()
                    return False

                if attempt < max_retries - 1:
                    self.grip_close_amount += 0.005  # Tighter grip for next attempt

        # All attempts failed
        self.get_logger().error(f"❌ Failed to acquire food after {max_retries} attempts")
        await self.stop_detection_cleanly()
        return False

    async def execute_pickup_sequence(self):
        """Execute the physical pickup sequence"""
        try:
            # Check required data
            if self.latest_grip_value is None:
                self.get_logger().error("No grip value available for pickup!")
                return False

            if self.latest_food_height is None:
                self.get_logger().error("No food height available for pickup!")
                return False

            if self.tracking_lost:
                self.get_logger().error("Cannot execute pickup - tracking lost!")
                return False

            # Set gripper to food width
            self.get_logger().info(f"Setting gripper to food width: {self.latest_grip_value:.3f}")
            if not await self.robot_controller.set_gripper(self.latest_grip_value):
                return False

            # Get current pose and move down
            current_pose = await self.get_current_pose()
            if not current_pose:
                return False

            pickup_pose = copy.deepcopy(current_pose)
            move_down_distance = self.distance_from_target + self.latest_food_height
            pickup_pose.position.z -= move_down_distance

            self.get_logger().info(f"Moving down {move_down_distance:.4f}m for pickup")
            if not await self.robot_controller.move_to_pose(pickup_pose):
                return False

            # Close gripper
            close_value = min(1.0, self.latest_grip_value + self.grip_close_amount)
            self.get_logger().info(f"Closing gripper from {self.latest_grip_value:.3f} to {close_value:.3f}")
            if not await self.robot_controller.set_gripper(close_value):
                return False

            # Reset food angle and move back to overlook
            self.food_angle_pub.publish(Float64(data=0.0))
            self.get_logger().info("Moving back to overlook after pickup...")
            if not await self.robot_controller.reset():
                return False

            return True

        except Exception as e:
            self.get_logger().error(f"Error in pickup sequence: {e}")
            return False

    def wait_for_food_pickup_from_segment_depth(self):
        """Check if food was picked up using food depth"""
        self.get_logger().info("Checking if food was picked up using food depth...")

        # Spin nodes to get latest data
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)
            rclpy.spin_once(self.autonomous_checker, timeout_sec=0.1)
            time.sleep(0.1)

        return self.autonomous_checker.is_object_grasped_from_food_depth(
            self.autonomous_checker.latest_food_depth, depth_threshold=0.25)

    def wait_for_food_pickup(self):
        """Check if food was picked up using autonomous checker"""
        for _ in range(10):
            rclpy.spin_once(self.autonomous_checker, timeout_sec=0.1)
            time.sleep(0.1)

        return self.autonomous_checker.check_object_grasped()

    def wait_for_food_removal(self):
        """Wait for food to be removed from gripper"""
        self.get_logger().info("Waiting for food to be removed...")
        return self.autonomous_checker.check_object_removed()

    async def move_food_to_mouth(self):
        """Move food to mouth using face detection or preset positions"""
        if self.face_detection_enabled:
            self.get_logger().info("=== Moving food to mouth using face detection ===")

            # Start face detection
            self.start_face_detection_pub.publish(Bool(data=True))
            face_gains = Vector3(x=0.35, y=0.35, z=0.35)
            self.twist_gains_pub.publish(face_gains)
            await asyncio.sleep(0.5)

            # Start servoing
            self.servoing_on_pub.publish(Bool(data=True))

            # Wait for robot to reach mouth
            success = await self.wait_for_small_position_vector(threshold=0.04, timeout=60.0)

            # Stop face detection and servoing
            self.servoing_on_pub.publish(Bool(data=False))
            self.start_face_detection_pub.publish(Bool(data=False))

            if success:
                return True
            else:
                self.get_logger().warn("Face detection failed, using preset position")
                return await self.robot_controller.move_to_bite_transfer()
        else:
            # Use preset bite transfer position
            self.get_logger().info("=== Moving to preset bite transfer position ===")
            return await self.robot_controller.move_to_bite_transfer()

    async def run_feeding_cycle(self):
        """Main feeding cycle with clean state management"""
        cycle_count = 1

        while rclpy.ok():
            self.get_logger().info(f"\n{'='*50}")
            self.get_logger().info(f"FEEDING CYCLE {cycle_count}")
            self.get_logger().info(f"{'='*50}")

            # Reset state for new cycle
            self.tracking_lost = False

            try:
                # Step 1: Reset robot
                self.robot_state_pub.publish(String(data='Setting up robot...'))
                if not await self.robot_controller.reset():
                    self.get_logger().error("Failed to reset robot!")
                    break
                if not await self.robot_controller.set_gripper(0.5):
                    self.get_logger().error("Failed to set gripper!")
                    break

                # Step 2: Acquire food
                if not await self.acquire_food_with_retries():
                    if self.tracking_lost:
                        self.get_logger().error("❌ Food acquisition failed due to tracking loss - RESTARTING")
                        continue
                    else:
                        self.get_logger().error("❌ Food acquisition failed completely - RESTARTING")
                        continue

                # Step 3: Move to intermediate
                self.robot_state_pub.publish(String(data='Moving to face scan position'))
                if not await self.robot_controller.move_to_intermediate():
                    self.get_logger().error("Failed to move to intermediate!")
                    continue

                # Step 4: Check if food is still grasped
                transfer_success = self.wait_for_food_pickup()
                if not transfer_success:
                    if self.tracking_lost:
                        self.get_logger().error("❌ Tracking lost during transfer - restarting")
                        await self.stop_detection_cleanly()
                        continue
                    else:
                        self.get_logger().error("❌ Food lost during transfer - restarting")
                        await self.stop_detection_cleanly()
                        continue

                # Step 5: Stop detection (food confirmed in gripper)
                await self.stop_detection_cleanly()

                # Step 6: Move food to mouth
                self.robot_state_pub.publish(String(data='Moving towards mouth'))
                if not await self.move_food_to_mouth():
                    self.get_logger().error("Failed to move food to mouth!")
                    continue
                self.play_sound(self.snap)

                # Step 7: Wait for food removal
                self.robot_state_pub.publish(String(data='Waiting for food removal...'))
                self.wait_for_food_removal()

                # Step 8: Reset to overlook
                if not await self.robot_controller.reset():
                    self.get_logger().error("Failed to reset after food removal!")
                    break

                self.get_logger().info(f"🎉 CYCLE {cycle_count} COMPLETED SUCCESSFULLY! 🎉")
                cycle_count += 1

            except KeyboardInterrupt:
                self.get_logger().info("Feeding cycle interrupted by user")
                break
            except Exception as e:
                self.get_logger().error(f"Error in feeding cycle: {str(e)}")
                if self.tracking_lost:
                    continue
                else:
                    break

        # Final cleanup
        self.get_logger().info("Feeding complete. Cleaning up...")
        await self.stop_detection_cleanly()
        self.servoing_on_pub.publish(Bool(data=False))
        self.start_face_detection_pub.publish(Bool(data=False))
        await self.robot_controller.reset()


async def main():
    rclpy.init()

    try:
        orchestrator = CleanOrchestrator()

        # Create executor and start spinning
        from rclpy.executors import MultiThreadedExecutor
        import threading

        executor = MultiThreadedExecutor()
        executor.add_node(orchestrator)

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