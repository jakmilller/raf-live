#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
import asyncio
import sys
import yaml
import time
import copy

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
        
        # Subscribers for food detection outputs
        self.grip_value_sub = self.create_subscription(
            Float64, '/grip_value', self.grip_value_callback, 10)
        self.finished_servoing_sub = self.create_subscription(
            Bool, '/finished_servoing', self.finished_servoing_callback, 10)
        
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
            self.get_logger().info('Waiting for get pose service...')
        
        # State
        self.servoing_finished = False
        
        self.get_logger().info('Minimal Orchestrator ready!')
    
    def grip_value_callback(self, msg):
        """Store latest grip value from food detection"""
        self.latest_grip_value = msg.data
    
    def finished_servoing_callback(self, msg):
        """Handle finished servoing signal"""
        if msg.data:
            self.get_logger().info("RECEIVED FINISHED SERVOING SIGNAL!")
            self.servoing_finished = True
        else:
            self.get_logger().info("Received servoing signal with False value")
    
    async def get_current_pose(self):
        """Get current pose using the GetPose service"""
        try:
            request = GetPose.Request()
            response = await self.get_pose_client.call_async(request)
            
            if response.success:
                self.get_logger().info(f'Current pose: position=({response.current_pose.position.x:.3f}, {response.current_pose.position.y:.3f}, {response.current_pose.position.z:.3f})')
                return response.current_pose
            else:
                self.get_logger().error(f'Failed to get current pose: {response.message}')
                return None
                
        except Exception as e:
            self.get_logger().error(f'Error calling get pose service: {e}')
            return None
    
    async def call_food_detection_service(self):
        """Call food detection service and wait for completion"""
        self.get_logger().info("Starting food detection servoing...")
        self.servoing_finished = False
        
        request = StartFaceServoing.Request()
        request.gain_planar = 0.4
        request.gain_depth = 0.4
        request.target_distance = 0.015
        
        try:
            response = await self.food_service_client.call_async(request)
            if not response.success:
                self.get_logger().error(f"Food detection service failed: {response.message}")
                return False
            
            self.get_logger().info("Food detection service started, waiting for completion...")
            
            # Wait for servoing to complete with timeout and better logging
            timeout_counter = 0
            max_timeout = 300  # 30 seconds timeout
            
            while not self.servoing_finished and rclpy.ok() and timeout_counter < max_timeout:
                await asyncio.sleep(0.1)
                timeout_counter += 1
                
                # Log every 5 seconds
                if timeout_counter % 50 == 0:
                    self.get_logger().info(f"Still waiting for food servoing to complete... ({timeout_counter/10:.1f}s)")
            
            if timeout_counter >= max_timeout:
                self.get_logger().error("Timeout waiting for food servoing to complete!")
                return False
            
            self.get_logger().info("Food detection servoing completed")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Food detection service failed: {e}")
            return False
    
    async def call_face_detection_service(self):
        """Call face detection service and wait for completion"""
        self.get_logger().info("Starting face detection servoing...")
        self.servoing_finished = False
        
        request = StartFaceServoing.Request()
        request.gain_planar = 0.35
        request.gain_depth = 0.35
        request.target_distance = 0.04  # 4cm
        
        try:
            response = await self.face_service_client.call_async(request)
            if not response.success:
                self.get_logger().error(f"Face detection service failed: {response.message}")
                return False
            
            self.get_logger().info("Face detection service started, waiting for completion...")
            
            # Wait for servoing to complete with timeout and better logging
            timeout_counter = 0
            max_timeout = 300  # 30 seconds timeout
            
            while not self.servoing_finished and rclpy.ok() and timeout_counter < max_timeout:
                await asyncio.sleep(0.1)
                timeout_counter += 1
                
                # Log every 5 seconds
                if timeout_counter % 50 == 0:
                    self.get_logger().info(f"Still waiting for face servoing to complete... ({timeout_counter/10:.1f}s)")
            
            if timeout_counter >= max_timeout:
                self.get_logger().error("Timeout waiting for face servoing to complete!")
                return False
            
            self.get_logger().info("Face detection servoing completed")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Face detection service failed: {e}")
            return False
    
    def wait_for_food_pickup(self):
        """Check if food was picked up using autonomous checker"""
        self.get_logger().info("Checking if food was picked up...")
        
        # Spin the autonomous checker a few times to get latest data
        for _ in range(10):
            rclpy.spin_once(self.autonomous_checker, timeout_sec=0.1)
            time.sleep(0.1)
        
        return self.autonomous_checker.check_object_grasped()
    
    def wait_for_food_removal(self):
        """Wait for food to be removed from gripper"""
        self.get_logger().info("Waiting for food to be removed...")
        return self.autonomous_checker.check_object_removed()
    
    async def run_feeding_cycle(self):
        """Main feeding cycle"""
        cycle_count = 1
        
        while rclpy.ok():
            self.get_logger().info(f"\n=== FEEDING CYCLE {cycle_count} ===")
            
            try:
                # Step 1: Move to overlook position and set gripper
                self.get_logger().info("Step 1: Moving to overlook and setting gripper...")
                if not await self.robot_controller.reset():
                    self.get_logger().error("Failed to reset robot!")
                    break
                if not await self.robot_controller.set_gripper(0.5):
                    self.get_logger().error("Failed to set gripper!")
                    break
                
                # Step 2: Start food detection and wait for servoing to complete
                self.get_logger().info("Step 2: Starting food detection servoing...")
                if not await self.call_food_detection_service():
                    self.get_logger().error("Food detection failed!")
                    continue
                
                # Step 3: Pick up the food
                self.get_logger().info("Step 3: Picking up food...")
                
                # Use latest grip value (wait a moment for it to be published)
                await asyncio.sleep(0.5)
                if self.latest_grip_value is None:
                    self.get_logger().warn("No grip value received, using default")
                    grip_value = 0.4
                else:
                    grip_value = self.latest_grip_value
                
                # Set gripper to food width
                if not await self.robot_controller.set_gripper(grip_value):
                    self.get_logger().error("Failed to set gripper for pickup!")
                    continue
                
                await asyncio.sleep(0.5)
                
                # Get current pose and move down 1cm
                current_feedback = await self.get_current_pose()
                if current_feedback:
                    pickup_pose = copy.deepcopy(current_feedback)
                    pickup_pose.position.z -= 0.01  # Move down 1cm
                    
                    if not await self.robot_controller.move_to_pose(pickup_pose):
                        self.get_logger().error("Failed to move down for pickup!")
                        continue
                else:
                    self.get_logger().warn("Could not get current pose, skipping precise pickup movement")
                
                # Close gripper slightly more
                close_value = min(1.0, grip_value + 0.056)  # Same as original orchestrator
                if not await self.robot_controller.set_gripper(close_value):
                    self.get_logger().error("Failed to close gripper!")
                    continue
                
                # Move up
                if current_feedback:
                    up_pose = copy.deepcopy(current_feedback)
                    up_pose.position.z += 0.1  # Move up 10cm
                    if not await self.robot_controller.move_to_pose(up_pose):
                        self.get_logger().error("Failed to move up!")
                        continue
                else:
                    self.get_logger().warn("Could not get current pose, skipping up movement")
                
                # Step 4: Check if food was picked up
                self.get_logger().info("Step 4: Checking pickup...")
                pickup_success = self.wait_for_food_pickup()
                if not pickup_success:
                    self.get_logger().warn("Food pickup not confirmed, continuing anyway...")
                
                # Step 5: Move to intermediate position
                self.get_logger().info("Step 5: Moving to intermediate position...")
                if not await self.robot_controller.move_to_intermediate():
                    self.get_logger().error("Failed to move to intermediate!")
                    continue
                
                # Step 6: Start face detection servoing
                self.get_logger().info("Step 6: Starting face detection servoing...")
                if not await self.call_face_detection_service():
                    self.get_logger().error("Face detection failed!")
                    continue
                
                # Step 7: Wait for food removal
                self.get_logger().info("Step 7: Waiting for food removal...")
                self.wait_for_food_removal()
                
                self.get_logger().info("Cycle completed successfully!")
                cycle_count += 1
                
            except KeyboardInterrupt:
                self.get_logger().info("Feeding cycle interrupted by user")
                break
            except Exception as e:
                self.get_logger().error(f"Error in feeding cycle: {str(e)}")
                break
        
        # Final return to overlook
        self.get_logger().info("Feeding complete. Returning to overlook position...")
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