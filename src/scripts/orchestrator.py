#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Pose
from std_msgs.msg import Float64, Bool
import asyncio
import sys
import yaml
import time
import copy
import numpy as np
import os

# Import controllers/checkers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_controller_ros2 import KinovaRobotControllerROS2
from autonomous_checker import AutonomousChecker
from raf_interfaces.srv import GetPose


class SimplifiedOrchestrator(Node):
    def __init__(self):
        super().__init__('simplified_orchestrator')
        
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
        
        # State variables - only what we actually need
        self.latest_position_vector = Vector3()
        self.latest_grip_value = None
        self.latest_food_height = None
        self.food_detection_ready = False
        
        # Subscribers - only the essential data
        self.position_vector_sub = self.create_subscription(
            Vector3, '/position_vector', self.position_vector_callback, 1)
        self.grip_value_sub = self.create_subscription(
            Float64, '/grip_value', self.grip_value_callback, 1)
        self.food_height_sub = self.create_subscription(
            Float64, '/food_height', self.food_height_callback, 1)
        self.food_detection_ready_sub = self.create_subscription(
            Bool, '/food_detection_ready', self.food_detection_ready_callback, 1)
        
        # Publishers - simple control interface
        self.start_food_detection_pub = self.create_publisher(Bool, '/start_food_detection', 1)
        self.start_face_detection_pub = self.create_publisher(Bool, '/start_face_detection', 1)
        self.servoing_on_pub = self.create_publisher(Bool, '/servoing_on', 1)
        self.twist_gains_pub = self.create_publisher(Vector3, '/twist_gains', 1)
        
        # Service client for getting current pose
        self.get_pose_client = self.create_client(GetPose, '/my_gen3/get_pose')
        while not self.get_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /my_gen3/get_pose service...')
        
        self.get_logger().info(f'Simplified Orchestrator initialized!')
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
    
    def food_detection_ready_callback(self, msg):
        """Track when food detection is ready"""
        self.food_detection_ready = msg.data
        if msg.data:
            self.get_logger().info("Food detection is ready - tracking initialized!")
    
    async def wait_for_food_detection_ready(self, timeout=30.0):
        """Wait for food detection to be ready (tracking initialized)"""
        self.get_logger().info("Waiting for food detection to be ready...")
        
        start_time = time.time()
        
        while rclpy.ok():
            if time.time() - start_time > timeout:
                self.get_logger().error(f"Timeout waiting for food detection ready after {timeout}s")
                return False
            
            if self.food_detection_ready:
                self.get_logger().info("Food detection ready!")
                return True
            
            await asyncio.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0)
        
        return False
    
    async def wait_for_small_position_vector(self, threshold=0.005, timeout=30.0):
        """Wait until position vector magnitude is small (robot reached target)"""
        self.get_logger().info(f"Waiting for position vector < {threshold}m...")
        
        start_time = time.time()
        stable_count = 0
        required_stable_count = 10  # Need 10 consecutive small distances (1 second at 10Hz)
        
        while rclpy.ok():
            if time.time() - start_time > timeout:
                self.get_logger().warn(f"Timeout waiting for small position vector after {timeout}s")
                return False
            
            # Check position vector magnitude using the actual position vector data
            magnitude = np.linalg.norm([
                self.latest_position_vector.x,
                self.latest_position_vector.y,
                self.latest_position_vector.z
            ])
            
            if magnitude < threshold:
                stable_count += 1
                if stable_count >= required_stable_count:
                    self.get_logger().info(f"Position vector consistently small: {magnitude:.3f}m")
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
            
            start_time = time.time()
            timeout = 5.0
            
            while not future.done() and rclpy.ok():
                if time.time() - start_time > timeout:
                    self.get_logger().error(f'GetPose service call timed out after {timeout} seconds')
                    return None
                
                rclpy.spin_once(self, timeout_sec=0.1)
                await asyncio.sleep(0.01)
            
            if not future.done():
                self.get_logger().error('GetPose service call incomplete')
                return None
                
            response = future.result()
            
            if response.success:
                self.get_logger().info(f'Current pose: position=({response.current_pose.position.x:.3f}, {response.current_pose.position.y:.3f}, {response.current_pose.position.z:.3f})')
                return response.current_pose
            else:
                self.get_logger().error(f'GetPose service returned failure: {response.message}')
                return None
                
        except Exception as e:
            self.get_logger().error(f'Error calling get pose service: {e}')
            return None
    
    async def acquire_food_with_retries(self, max_retries=3):
        """Attempt to acquire food with retry logic"""
        self.food_detection_ready = False
        for attempt in range(max_retries):
            self.get_logger().info(f"=== Food acquisition attempt {attempt + 1}/{max_retries} ===")
            
            # Reset state
            self.latest_grip_value = None
            self.latest_food_height = None

            if not await self.robot_controller.set_gripper(0.5):
                self.get_logger().error("Failed to set gripper!")
            
            # Step 1: Start food detection
            self.get_logger().info("1. Starting food detection...")
            self.start_food_detection_pub.publish(Bool(data=True))
            
            # Step 2: Set gains and start servoing
            self.get_logger().info("2. Setting gains and waiting for detection ready...")
            gains = Vector3(x=0.6, y=0.6, z=0.6)
            self.twist_gains_pub.publish(gains)
            
            # Wait for detection to be ready before starting servoing
            if not await self.wait_for_food_detection_ready(timeout=30.0):
                self.get_logger().error("Food detection failed to initialize")
                self.start_food_detection_pub.publish(Bool(data=False))
                continue
            
            # Now start servoing
            self.get_logger().info("3. Starting servoing...")
            self.servoing_on_pub.publish(Bool(data=True))
            
            # Step 3: Wait for robot to reach food
            self.get_logger().info("4. Waiting for robot to reach food...")
            success = await self.wait_for_small_position_vector(threshold=0.005, timeout=30.0)
            
            if not success:
                self.get_logger().error("Failed to reach food - timeout")
                self.servoing_on_pub.publish(Bool(data=False))
                self.start_food_detection_pub.publish(Bool(data=False))
                continue
            
            # Step 4: Stop servoing for pickup
            self.get_logger().info("5. Reached food! Stopping servoing for pickup...")
            self.servoing_on_pub.publish(Bool(data=False))
            await asyncio.sleep(0.5)
            
            # Step 5: Execute pickup sequence
            if not await self.execute_pickup_sequence():
                self.get_logger().warn(f"Pickup sequence failed on attempt {attempt + 1}")
                # Keep detection running for retry
                continue
            
            # Step 6: Check if food was picked up
            self.get_logger().info("6. Checking if food was picked up...")
            pickup_success = self.wait_for_food_pickup()
            
            if pickup_success:
                self.get_logger().info("Food pickup successful!")
                # Stop food detection now that we have the food
                self.start_food_detection_pub.publish(Bool(data=False))
                return True
            else:
                self.get_logger().warn(f"Food pickup failed on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    # Restart servoing for next attempt (keep detection running)
                    self.get_logger().info("Restarting servoing for retry...")
                    self.servoing_on_pub.publish(Bool(data=True))
        
        # All attempts failed
        self.get_logger().error(f"Failed to acquire food after {max_retries} attempts")
        self.start_food_detection_pub.publish(Bool(data=False))
        return False
    
    async def execute_pickup_sequence(self):
        """Execute the physical pickup sequence"""
        try:
            # Check that we have the necessary data
            if self.latest_grip_value is None:
                self.get_logger().error("No grip value available for pickup!")
                return False
            
            if self.latest_food_height is None:
                self.get_logger().error("No food height available for pickup!")
                return False
            
            # Set gripper to food width
            self.get_logger().info(f"Setting gripper to food width: {self.latest_grip_value:.3f}")
            if not await self.robot_controller.set_gripper(self.latest_grip_value):
                self.get_logger().error("Failed to set gripper for pickup!")
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
                self.get_logger().error("Failed to move down for pickup!")
                return False
            
            # Close gripper more
            close_value = min(1.0, self.latest_grip_value + self.grip_close_amount)
            self.get_logger().info(f"Closing gripper from {self.latest_grip_value:.3f} to {close_value:.3f}")
            if not await self.robot_controller.set_gripper(close_value):
                self.get_logger().error("Failed to close gripper!")
                return False
            
            # Move back to overlook
            self.get_logger().info("Moving back to overlook after pickup...")
            if not await self.robot_controller.reset():
                self.get_logger().error("Failed to move back to overlook!")
                return False
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error in pickup sequence: {e}")
            return False
    
    def wait_for_food_pickup(self):
        """Check if food was picked up using autonomous checker"""
        self.get_logger().info("Checking if food was picked up...")
        
        # Spin the autonomous checker to get latest data
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
            self.start_face_detection_pub.publish(Bool(data=False))
            
            if success:
                self.get_logger().info("Face detection servoing completed successfully!")
                return True
            else:
                self.get_logger().warn("Face detection failed, falling back to preset position")
                # Fall back to preset position
                return await self.robot_controller.move_to_bite_transfer()
        else:
            # Use preset bite transfer position
            self.get_logger().info("=== Moving to preset bite transfer position ===")
            return await self.robot_controller.move_to_bite_transfer()
    
    async def run_feeding_cycle(self):
        """Main feeding cycle - simplified linear flow"""
        cycle_count = 1
        
        while rclpy.ok():
            self.get_logger().info(f"\n{'='*50}")
            self.get_logger().info(f"FEEDING CYCLE {cycle_count}")
            self.get_logger().info(f"{'='*50}")
            
            try:
                # Step 1: Move to overlook and open gripper
                self.get_logger().info("STEP 1: Moving to overlook and setting gripper...")
                if not await self.robot_controller.reset():
                    self.get_logger().error("Failed to reset robot!")
                    break
                if not await self.robot_controller.set_gripper(0.5):
                    self.get_logger().error("Failed to set gripper!")
                    break
                
                # Step 2: Acquire food with retries
                self.get_logger().info("STEP 2: Acquiring food...")
                if not await self.acquire_food_with_retries():
                    self.get_logger().error("Food acquisition failed completely, restarting cycle")
                    continue
                
                # Step 3: Move to intermediate position
                self.get_logger().info("STEP 3: Moving to intermediate position...")
                if not await self.robot_controller.move_to_intermediate():
                    self.get_logger().error("Failed to move to intermediate!")
                    continue
                
                # Step 4: Move food to mouth
                self.get_logger().info("STEP 4: Moving food to mouth...")
                if not await self.move_food_to_mouth():
                    self.get_logger().error("Failed to move food to mouth!")
                    continue
                
                # Step 5: Wait for food removal
                self.get_logger().info("STEP 5: Waiting for food to be removed...")
                self.wait_for_food_removal()
                
                # Step 6: Reset to overlook position
                self.get_logger().info("STEP 6: Resetting to overlook position...")
                if not await self.robot_controller.reset():
                    self.get_logger().error("Failed to reset robot after food removal!")
                    break
                
                self.get_logger().info(f"🎉 CYCLE {cycle_count} COMPLETED SUCCESSFULLY! 🎉")
                cycle_count += 1
                
            except KeyboardInterrupt:
                self.get_logger().info("Feeding cycle interrupted by user")
                break
            except Exception as e:
                self.get_logger().error(f"Error in feeding cycle: {str(e)}")
                break
        
        # Final cleanup
        self.get_logger().info("Feeding complete. Cleaning up...")
        self.servoing_on_pub.publish(Bool(data=False))
        self.start_food_detection_pub.publish(Bool(data=False))
        self.start_face_detection_pub.publish(Bool(data=False))
        await self.robot_controller.reset()


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