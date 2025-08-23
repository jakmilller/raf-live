#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64, Bool
import asyncio
import sys
import yaml
import time
import copy
import numpy as np

# Import service interfaces
from raf_interfaces.srv import StartFaceServoing, GetPose

# Import controllers/checkers from existing scripts
sys.path.append('/home/mcrr-lab/raf-live/src/scripts')
from robot_controller_ros2 import KinovaRobotControllerROS2
from autonomous_checker import AutonomousChecker

class MinimalOrchestrator(Node):
    def __init__(self):
        super().__init__('minimal_orchestrator')
        
        config_path = '/home/mcrr-lab/raf-live/config.yaml'
        
        # Load config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.robot_controller = KinovaRobotControllerROS2(config_path)
        self.autonomous_checker = AutonomousChecker(self.config)
        
        # Food detection data
        self.latest_grip_value = None
        self.servoing_on = False
        self.distance_from_target = self.config['feeding']['dist_from_food']
        
        # Subscribers for food detection outputs
        self.grip_value_sub = self.create_subscription(
            Float64, '/grip_value', self.grip_value_callback, 1)
        self.servoing_on_sub = self.create_subscription(
            Bool, '/servoing_on', self.servoing_on_callback, 1)
        self.food_height_sub = self.create_subscription(
            Float64, '/food_height', self.food_height_callback, 1)
        self.position_vector_sub = self.create_subscription(
            Vector3, '/position_vector', self.position_vector_callback, 1)
        self.detection_ready_sub = self.create_subscription(
            Bool, '/detection_ready', self.detection_ready_callback, 1)
        self.detection_ready = False
        
        self.latest_food_height = None
        self.latest_position_magnitude = 0.0
        
        # Publishers for new topics
        self.servoing_on_pub = self.create_publisher(Bool, '/servoing_on', 1)
        self.food_acquired_pub = self.create_publisher(Bool, '/food_acquired', 1)
        
        # Service clients for detection services
        self.food_service_client = self.create_client(StartFaceServoing, 'start_food_servoing')
        self.face_service_client = self.create_client(StartFaceServoing, 'start_face_servoing')
        self.get_pose_client = self.create_client(GetPose, '/my_gen3/get_pose')
        
        # Wait for services
        while not self.food_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for food servoing service...')
        while not self.face_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for face servoing service...')
        while not self.get_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for get pose service at /my_gen3/get_pose...')
            self.get_logger().info('Make sure the kortex controller (controller.cpp) is running!')
        
        self.get_logger().info('All services connected successfully!')
        self.get_logger().info('Minimal Orchestrator ready!')
    
    def grip_value_callback(self, msg):
        """Store latest grip value from food detection"""
        self.latest_grip_value = msg.data

    def food_height_callback(self, msg):
        """Store latest food height"""
        self.latest_food_height = msg.data
    
    def position_vector_callback(self, msg):
        """Track position vector magnitude for distance monitoring"""
        # Assuming this is the magnitude of the position vector
        self.latest_position_magnitude = np.linalg.norm([msg.x, msg.y, msg.z])

    def detection_ready_callback(self, msg):
        """Detection is ready and tracking has started"""
        if msg.data:
            self.detection_ready = True
            self.get_logger().info("Food detection and tracking is ready!")
    
    def servoing_on_callback(self, msg):
        """Track servoing state (for monitoring only)"""
        self.servoing_on = msg.data
        if not self.servoing_on:
            self.get_logger().debug("Servoing disabled")
        else:
            self.get_logger().debug("Servoing enabled")
    
    async def wait_for_target_reached(self, timeout=30.0):
        """Wait for robot to reach close to the food (small position vector)"""
        self.get_logger().info("Waiting for robot to reach close to food...")
        
        start_time = time.time()
        stable_count = 0
        required_stable_count = 10  # Need 10 consecutive small distances
        
        while rclpy.ok():
            if time.time() - start_time > timeout:
                self.get_logger().warn(f"Timeout waiting for target after {timeout}s")
                return False
            
            # Check if we're close to the food (small position vector magnitude)
            if self.latest_position_magnitude < 0.005:  # 5mm threshold
                stable_count += 1
                if stable_count >= required_stable_count:
                    self.get_logger().info("Robot has reached close to food!")
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
    
    async def call_food_detection_service(self):
        """Call food detection service and return immediately"""
        self.get_logger().info("Starting food detection servoing...")
        
        request = StartFaceServoing.Request()
        request.gain_planar = 0.6
        request.gain_depth = 0.6
        request.target_distance = self.distance_from_target
        
        try:
            response = await self.food_service_client.call_async(request)
            if not response.success:
                self.get_logger().error(f"Food detection service failed: {response.message}")
                return False
            
            self.get_logger().info("Food detection service started successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Food detection service failed: {e}")
            return False
    
    async def call_face_detection_service(self):
        """Call face detection service and wait for completion"""
        self.get_logger().info("Starting face detection servoing...")
        
        request = StartFaceServoing.Request()
        request.gain_planar = 0.35
        request.gain_depth = 0.35
        request.target_distance = 0.04  # 4cm
        
        try:
            response = await self.face_service_client.call_async(request)
            if not response.success:
                self.get_logger().error(f"Face detection service failed: {response.message}")
                return False
            
            self.get_logger().info("Face detection service completed")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Face detection service failed: {e}")
            return False
    
    def wait_for_food_pickup(self):
        """Check if food was picked up using autonomous checker"""
        self.get_logger().info("Checking if food was picked up...")
        
        for _ in range(10):
            rclpy.spin_once(self.autonomous_checker, timeout_sec=0.1)
            time.sleep(0.1)
        
        return self.autonomous_checker.check_object_grasped()
    
    def wait_for_food_removal(self):
        """Wait for food to be removed from gripper"""
        self.get_logger().info("Waiting for food to be removed...")
        return self.autonomous_checker.check_object_removed()
    
    async def acquire_food_with_retries(self, max_retries=3):
        """Attempt to acquire food with retry logic"""
        for attempt in range(max_retries):
            self.get_logger().info(f"Food acquisition attempt {attempt + 1}/{max_retries}")

            if not await self.robot_controller.set_gripper(0.5):
                self.get_logger().error("Failed to set gripper for pickup!")
                continue

            await asyncio.sleep(0.5)

            # Enable servoing for this attempt
            self.get_logger().info("Starting servoing for food approach...")
            self.servoing_on_pub.publish(Bool(data=True))

            # Wait for detection to be ready
            self.detection_ready = False  # Reset flag
            self.get_logger().info("Waiting for food detection to initialize...")

            timeout_start = time.time()
            while not self.detection_ready and rclpy.ok():
                if time.time() - timeout_start > 30.0:
                    self.get_logger().error("Timeout waiting for detection ready signal!")
                    self.servoing_on_pub.publish(Bool(data=False))
                    await asyncio.sleep(0.5)
                    break
                
                await asyncio.sleep(0.1)
                rclpy.spin_once(self, timeout_sec=0)

            if not self.detection_ready:
                self.get_logger().error("Detection failed to initialize, skipping to next attempt")
                continue  # Skip to next attempt

            self.get_logger().info("Detection ready! Proceeding with servoing...")

            # Wait for robot to reach close to food
            if not await self.wait_for_target_reached():
                self.get_logger().error("Failed to reach food - timeout")
                # Stop servoing before continuing to next attempt
                self.servoing_on_pub.publish(Bool(data=False))
                # await asyncio.sleep(0.5)
                continue
            
            # Stop servoing once we're close
            self.get_logger().info("Reached target, stopping servoing for pickup...")
            self.servoing_on_pub.publish(Bool(data=False))
            # await asyncio.sleep(0.5)  # Let robot settle

            # Now do the pickup sequence
            self.get_logger().info("Attempting food pickup...")

            grip_value = self.latest_grip_value

            # Check if we have a valid grip value
            if grip_value is None:
                self.get_logger().error("No grip value available for pickup!")
                continue
            
            # Set gripper to food width
            if not await self.robot_controller.set_gripper(grip_value):
                self.get_logger().error("Failed to set gripper for pickup!")
                continue
            
            # await asyncio.sleep(0.5)

            # Get current pose and move down
            self.get_logger().info("Getting current pose for pickup movement...")
            current_pose = await self.get_current_pose()
            if current_pose:
                pickup_pose = copy.deepcopy(current_pose)
                if self.latest_food_height is not None:
                    self.get_logger().info(f"Using measured food height: {self.latest_food_height:.4f}m")

                pickup_pose.position.z -= (self.distance_from_target + self.latest_food_height)

                self.get_logger().info(f"Moving down {self.latest_food_height+self.distance_from_target}m")
                move_result = await self.robot_controller.move_to_pose(pickup_pose)
                self.get_logger().info(f"Move down result: {move_result}")
                if not move_result:
                    self.get_logger().error("Failed to move down for pickup!")
                    continue
                else:
                    self.get_logger().info("Successfully moved down for pickup")
            else:
                self.get_logger().error("Could not get current pose - this is blocking the pickup sequence!")
                continue
            
            # Close gripper slightly more
            close_value = min(1.0, grip_value + 0.056)
            self.get_logger().info(f"Closing gripper more from {grip_value:.3f} to {close_value:.3f}")
            if not await self.robot_controller.set_gripper(close_value):
                self.get_logger().error("Failed to close gripper!")
                continue
            else:
                self.get_logger().info("Successfully closed gripper more")

            # Move up
            # if pickup_pose:
            #     up_pose = copy.deepcopy(pickup_pose)
            #     up_pose.position.z += 0.1  # Move up 10cm
            #     self.get_logger().info(f"Moving up 10cm from original z")
            #     if not await self.robot_controller.move_to_pose(up_pose):
            #         self.get_logger().error("Failed to move up!")
            #         continue
            #     else:
            #         self.get_logger().info("Successfully moved up after pickup")
            # else:
            #     self.get_logger().error("Cannot move up - no reference pose available")
            #     continue

            # instead of just moving up, move back to overlook so we dont lose the segmentation
            self.get_logger().info("Moving back to overlook position after pickup...")
            if not await self.robot_controller.reset():
                self.get_logger().error("Failed to move back to overlook!")
                continue
            
            # Check if food was picked up
            pickup_success = self.wait_for_food_pickup()
            if pickup_success:
                self.get_logger().info("Food pickup successful!")
                # Signal that food was acquired
                self.food_acquired_pub.publish(Bool(data=True))
                return True
            else:
                self.get_logger().warn(f"Food pickup failed on attempt {attempt + 1}")
                # Continue to next retry if not max attempts
                if attempt < max_retries - 1:
                    self.get_logger().info("Retrying food acquisition...")

                    # Explicitly stop servoing and wait before retry
                    self.get_logger().info("Ensuring servoing is stopped before retry...")
                    self.servoing_on_pub.publish(Bool(data=False))
                    # await asyncio.sleep(1.0)  # Wait for system to settle

        # All attempts failed
        self.get_logger().error(f"Failed to acquire food after {max_retries} attempts")
        # Signal that we're done trying (this will end the service)
        self.food_acquired_pub.publish(Bool(data=True))
        return False
    
    async def run_feeding_cycle(self):
        """Main feeding cycle with retry logic"""
        cycle_count = 1
        
        while rclpy.ok():
            self.get_logger().info(f"\n=== FEEDING CYCLE {cycle_count} ===")
            
            try:
                # Reset state for new cycle
                self.servoing_on_pub.publish(Bool(data=False))
                self.food_acquired_pub.publish(Bool(data=False))
                await asyncio.sleep(0.5)
                
                # Step 1: Move to overlook position and set gripper
                self.get_logger().info("Step 1: Moving to overlook and setting gripper...")
                if not await self.robot_controller.reset():
                    self.get_logger().error("Failed to reset robot!")
                    break
                if not await self.robot_controller.set_gripper(0.5):
                    self.get_logger().error("Failed to set gripper!")
                    break
                
                # Step 2: Start food detection service
                self.get_logger().info("Step 2: Starting food detection service...")
                if not await self.call_food_detection_service():
                    self.get_logger().error("Food detection service failed to start!")
                    continue
                
                # Step 3: Attempt food acquisition with retries
                self.get_logger().info("Step 3: Attempting food acquisition...")
                acquisition_success = await self.acquire_food_with_retries()
                
                if not acquisition_success:
                    self.get_logger().error("Food acquisition failed completely, restarting cycle")
                    continue
                
                # Step 4: Move to intermediate position
                self.get_logger().info("Step 4: Moving to intermediate position...")
                if not await self.robot_controller.move_to_intermediate():
                    self.get_logger().error("Failed to move to intermediate!")
                    continue
                
                # Step 5: Start face detection servoing
                self.get_logger().info("Step 5: Starting face detection servoing...")
                self.servoing_on_pub.publish(Bool(data=True))

                if not await self.call_face_detection_service():
                    self.get_logger().error("Face detection failed!")
                    continue
                
                # Step 6: Wait for food removal
                self.get_logger().info("Step 6: Waiting for food removal...")
                self.wait_for_food_removal()
                
                self.get_logger().info("Cycle completed successfully!")
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
        self.food_acquired_pub.publish(Bool(data=True))
        await self.robot_controller.reset()

async def main():
    rclpy.init()
    
    try:
        orchestrator = MinimalOrchestrator()
        
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