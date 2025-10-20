#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Pose
from std_msgs.msg import Float64, Bool, String
from sensor_msgs.msg import Image
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

class Orchestrator(Node):
    def __init__(self):
        super().__init__('orchestrator')

        # load config
        config_path = os.path.expanduser('~/raf-live/config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

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

        # variables
        self.tracking_lost = False # when the tracking of the segment is accidentally lost
        self.food_tracking = False # whether the food detection node is currently tracking the food item

        # Initialize message storage variables
        self.latest_position_vector = None
        self.latest_grip_value = None
        self.latest_food_height = None
        self.vector_paused = False  # Track if position vector is paused due to detection failure

        # subscriptions
        self.position_vector_sub = self.create_subscription(Vector3, '/position_vector', self.position_vector_callback, 1)
        self.grip_value_sub = self.create_subscription(Float64, '/grip_value', self.grip_value_callback, 1)
        self.food_height_sub = self.create_subscription(Float64, '/food_height', self.food_height_callback, 1)
        self.food_item_sub = self.create_subscription(String, '/currently_serving', self.food_item_callback,1)
        self.tracking_lost_sub = self.create_subscription(Bool, '/tracking_lost', self.tracking_lost_callback, 1)
        self.food_tracking_ready_sub = self.create_subscription(Bool, '/food_tracking_ready', self.food_tracking_ready_callback, 1)
        self.vector_pause_sub = self.create_subscription(Bool, '/vector_pause', self.vector_pause_callback, 1)

        # self.camera_monitoring_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.camera_monitoring_callback, 1)

        # publishers
        self.start_food_detection_pub = self.create_publisher(Bool, '/start_food_detection', 1)
        self.stop_food_detection_pub = self.create_publisher(Bool, '/stop_food_detection', 1)
        self.start_face_detection_pub = self.create_publisher(Bool, '/start_face_detection', 1)
        self.robot_state_pub = self.create_publisher(String, '/robot_state', 1)
        self.servoing_on_pub = self.create_publisher(Bool, '/servoing_on', 1)
        self.twist_gains_pub = self.create_publisher(Vector3, '/twist_gains', 1)
        self.food_angle_pub = self.create_publisher(Float64, '/food_angle', 1)
        # self.estop_publisher = self.create_publisher(Bool, '/my_gen3/estop', 10)

    

        # Service client for getting current robot pose
        self.get_pose_client = self.create_client(GetPose, '/my_gen3/get_pose')
        while not self.get_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /my_gen3/get_pose service...')
        
        self.get_logger().info(f'Orchestrator initialized!')

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
        """Adjust close values (how tightly to close gripper) based on food item"""
        if msg.data == "grape":
            self.grip_close_amount = 0.056
            self.get_logger().info("Adjusting close value!")
        elif msg.data == "broccoli":
            self.grip_close_amount = 0.09
            self.get_logger().info("Adjusting close value!")

        else:
            self.grip_close_amount = self.config['feeding']['grip_close']

    def tracking_lost_callback(self, msg):
        """Handle when food_detection_node loses real-time tracking of food item"""
        if msg.data:
            self.get_logger().warn("Food detection lost tracking!")
            # turn servoing off immediately
            self.servoing_on_pub.publish(Bool(data=False))
            # stop food detection to reset state
            self.stop_food_detection_pub.publish(Bool(data=True))
            # set flag for immediate restart
            self.tracking_lost = True

    def food_tracking_ready_callback(self, msg):
        """Handle when food_detection_node begins tracking the food item"""
        self.food_tracking = msg.data

    def vector_pause_callback(self, msg):
        """Handle when position vector is paused due to detection failure"""
        self.vector_paused = msg.data

    async def wait_for_target_reached(self, threshold=0.003, timeout=30):
        """Wait for the robot to servo to the food item"""
        self.get_logger().info(f"Waiting for position vector < {threshold}m...")
        start_time = time.time()
        stable_count = 0
        required_stable_count = 5  # consecutive vectors within acceptable range of target

        while rclpy.ok():
            # Check for tracking lost immediately
            if self.tracking_lost:
                self.get_logger().warn("Tracking lost during servoing - aborting wait")
                self.servoing_on_pub.publish(Bool(data=False))
                return False

            if time.time() - start_time > timeout:
                self.get_logger().warn(f"Timeout waiting for small position vector after {timeout}s")
                return False

            # Check position vector magnitude using the actual position vector data
            magnitude = np.linalg.norm([
                self.latest_position_vector.x,
                self.latest_position_vector.y,
                self.latest_position_vector.z
            ])

            # Only count small magnitude vectors if they're not paused due to detection failure
            if magnitude < threshold and not self.vector_paused:
                stable_count += 1
                if stable_count >= required_stable_count:
                    self.get_logger().info(f"Position vector consistently small: {magnitude:.3f}m")
                    return True
            elif self.vector_paused and magnitude < threshold:
                # Don't increment stable_count when paused, but don't reset it either
                # This allows the robot to stay stationary until detection resumes
                pass
            else:
                stable_count = 0

            await asyncio.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0)
        
        return False
    
    async def wait_for_food_tracking(self, timeout=40.0):
        """Wait for the food detection node to beign tracking the food item (it takes time for GT/DINO-X to return)"""
        self.get_logger().info("Waiting for food detection node to begin tracking...")
        start_time = time.time()

        while rclpy.ok():
            # Check for tracking lost immediately
            if self.tracking_lost:
                self.get_logger().warn("Detection failed - tracking lost before tracking began")
                return False

            if time.time() - start_time > timeout:
                self.get_logger().error(f"Timeout waiting for food detection node to begin tracking after {timeout}s")
                return False

            if self.food_tracking:
                self.get_logger().info("Food detection node has begun tracking!")
                return True

            await asyncio.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0)

        return False
    
    async def get_current_pose(self):
        """Get current robot pose using the GetPose service"""
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
        
    def play_sound(self, file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for the sound to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    def verify_food_acquisition(self):
        """Check if food was picked up using autonomous checker"""
        
        # Spin the autonomous checker to get latest data
        for _ in range(10):
            rclpy.spin_once(self.autonomous_checker, timeout_sec=0.1)
            time.sleep(0.1)
        
        return self.autonomous_checker.check_object_grasped()
    
    def wait_for_food_removal(self):
        """Wait for food to be removed from gripper"""
        self.get_logger().info("Waiting for food to be removed...")
        return self.autonomous_checker.check_object_removed()

    async def execute_pickup_sequence(self):
        """Execute pickup sequence"""
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
            close_value = min(0.8, self.latest_grip_value + self.grip_close_amount)
            
            self.get_logger().info(f"Closing gripper from {self.latest_grip_value:.3f} to {close_value:.3f}")
            if not await self.robot_controller.set_gripper(close_value):
                self.get_logger().error("Failed to close gripper!")
                return False
            
            self.food_angle_pub.publish(Float64(data=0.0))  # Reset food angle
            # Move back to overlook
            self.get_logger().info("Moving back to overlook after pickup...")
            if not await self.robot_controller.reset():
                self.get_logger().error("Failed to move back to overlook!")
                return False
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error in pickup sequence: {e}")
            return False
        
    async def attempt_pickup_with_retries(self, max_retries=2):
        """Try servoing and pickup up to max_retries+1 times. Return True if successful, False if all fail."""
        for attempt in range(max_retries + 1):
            # Check tracking before each attempt
            if self.tracking_lost:
                self.get_logger().info("Tracking lost - aborting pickup attempts")
                return False
            
            if not await self.robot_controller.set_gripper(0.5):
                    self.get_logger().error("Failed to set gripper!")
                    break

            self.get_logger().info(f"Pickup attempt {attempt + 1} of {max_retries + 1}...")

            # Set gains for food servoing
            self.get_logger().info("2. Setting food servoing gains...")
            food_gains = Vector3(x=0.65, y=0.65, z=0.65)
            self.twist_gains_pub.publish(food_gains)
            await asyncio.sleep(0.3)

            # Turn on visual servoing
            self.servoing_on_pub.publish(Bool(data=True))
            self.get_logger().info("Servoing to food item...")
            self.robot_state_pub.publish(String(data='Servoing to food item...'))
            success = await self.wait_for_target_reached()

            if not success or self.tracking_lost:
                self.get_logger().error("Servoing failed to reach food item or tracking lost!")
                self.servoing_on_pub.publish(Bool(data=False))
                self.food_tracking = False

                if self.tracking_lost:
                    return False
                continue

            self.get_logger().info("Reached food! Executing pickup sequence")
            self.servoing_on_pub.publish(Bool(data=False))  # turn off servoing
            self.robot_state_pub.publish(String(data='Picking up food item...'))
            success = await self.execute_pickup_sequence()

            if not success:
                self.get_logger().error("Executing pickup sequence failed!")
                self.food_tracking = False
                continue

            self.get_logger().info("Verifying food acquisition...")
            self.robot_state_pub.publish(String(data='Verifying food acquisition...'))
            await asyncio.sleep(0.5)  # Let robot settle after moving to overlook

            pickup_success = self.verify_food_acquisition()

            if pickup_success:
                self.get_logger().info("Pickup successful!")
                return True
            else:
                self.get_logger().info("Unsuccessful acquisition, retrying...")

        # If we reach here, all attempts failed
        self.food_tracking = False
        self.get_logger().error("All pickup attempts failed. Restarting feeding cycle.")
        return False
    
    async def move_food_to_mouth(self):
        """Move food to mouth using face detection or preset positions"""
        if self.face_detection_enabled:
            self.get_logger().info("=== Moving food to mouth with visual servoing ===")
            
            # Start face detection
            self.get_logger().info("1. Starting face detection...")
            self.start_face_detection_pub.publish(Bool(data=True))
            
            # Set gains for face servoing
            self.get_logger().info("2. Setting face servoing gains...")
            face_gains = Vector3(x=0.45, y=0.45, z=0.45)
            self.twist_gains_pub.publish(face_gains)
            await asyncio.sleep(0.5)
            
            # Start servoing
            self.get_logger().info("3. Starting face servoing...")
            self.servoing_on_pub.publish(Bool(data=True))
            
            # Wait for robot to reach mouth
            self.get_logger().info("4. Waiting for robot to reach mouth...")
            success = await self.wait_for_target_reached(threshold=0.04, timeout=60.0)
            
            # Stop face detection and servoing
            self.get_logger().info("5. Stopping face detection and servoing...")
            self.servoing_on_pub.publish(Bool(data=False))
            self.start_face_detection_pub.publish(Bool(data=False))
            
            if success:
                self.get_logger().info("Face detection servoing completed successfully!")
                return True
            else:
                self.get_logger().warn("Could not move to mouth!")
                return False
            
        else:
            # Use preset bite transfer position
            self.get_logger().info("=== Moving to preset bite transfer position ===")
            return await self.robot_controller.move_to_bite_transfer()
    

    async def run_feeding_cycle(self):
        """Main feeding cycle"""
        cycle_count = 1


        while rclpy.ok():
            self.get_logger().info(f"Starting feeding cycle {cycle_count}...")
            self.servoing_on_pub.publish(Bool(data=False))

            try:
                # reset tracking lost flag
                self.tracking_lost = False
                # ------------ STEP 1: Move to overlook and open gripper ------------
                self.robot_state_pub.publish(String(data='Setting up robot...'))
                self.get_logger().info("STEP 1: Moving to overlook and setting gripper...")
                if not await self.robot_controller.reset():
                    self.get_logger().error("Failed to reset robot!")
                    break
                if not await self.robot_controller.set_gripper(0.5):
                    self.get_logger().error("Failed to set gripper!")
                    break

                # ------------ STEP 2: Start food detection and pickup food item ------------
                self.get_logger().info("1. Starting food detection...")
                self.robot_state_pub.publish(String(data='Detecting food item...'))
                self.start_food_detection_pub.publish(Bool(data=True))
                
                # wait for tracking to start
                if not await self.wait_for_food_tracking(timeout=40.0):
                    self.get_logger().error("Food detection node failed to begin tracking!")
                    continue

                success = await self.attempt_pickup_with_retries(max_retries=2)
                if not success or self.tracking_lost:
                    if self.tracking_lost:
                        self.get_logger().info("Tracking lost during pickup - restarting cycle")
                        self.servoing_on_pub.publish(Bool(data=False))
                        self.stop_food_detection_pub.publish(Bool(data=True))
                        self.food_tracking = False
                        if not await self.robot_controller.reset():
                            self.get_logger().error("Failed to reset robot")
                            break
                        if not await self.robot_controller.set_gripper(0.5):
                            self.get_logger().error("Failed to reset gripper")
                            break
                        
                    continue  # restart feeding cycle

                # Tell food detection node to stop tracking
                self.stop_food_detection_pub.publish(Bool(data=True))

                #------------ STEP 3: Move to Intermediate (Face Scan Position) ------------\
                self.get_logger().info("STEP 3: Moving to intermediate position...")
                self.robot_state_pub.publish(String(data='Moving to face scan position'))

                if not await self.robot_controller.move_to_intermediate():
                    self.get_logger().error("Failed to move to intermediate!")
                    continue

                self.get_logger().info("Checking if food is still grasped after moving to intermediate")
                transfer_success = self.verify_food_acquisition()

                if not transfer_success:
                    self.get_logger().info("Food lost going to intermediate")
                    # Ensure servoing is off and food detection is stopped
                    self.servoing_on_pub.publish(Bool(data=False))
                    self.stop_food_detection_pub.publish(Bool(data=True))
                    # Reset food tracking flag so we wait for new detection
                    self.food_tracking = False
                    continue

                # -------------- STEP 4: Move food to mouth -----------------
                self.get_logger().info("STEP 4: Moving food to mouth...")
                self.robot_state_pub.publish(String(data='Moving towards mouth'))

                if not await self.move_food_to_mouth():
                    self.get_logger().error("Failed to move food to mouth!")
                    # Ensure servoing is off and food detection is stopped
                    self.servoing_on_pub.publish(Bool(data=False))
                    self.stop_food_detection_pub.publish(Bool(data=True))
                    # Reset food tracking flag so we wait for new detection
                    self.food_tracking = False
                    continue
                self.play_sound(self.snap)

                # --------------- STEP 5: Wait for food removal ------------
                self.get_logger().info("STEP 5: Waiting for food to be removed...")
                self.robot_state_pub.publish(String(data='Waiting for food removal...'))
                self.wait_for_food_removal()

                # ----------------STEP 6: Reset to overlook position
                self.get_logger().info("STEP 6: Resetting to overlook position...")
                if not await self.robot_controller.reset():
                    self.get_logger().error("Failed to reset robot after food removal!")
                    break

                self.get_logger().info(f"Feeding Cycle {cycle_count} completed")

            except KeyboardInterrupt:
                self.get_logger().info("Feeding cycle interrupted by user")
                break

            except Exception as e:
                self.get_logger().error(f"Error in feeding cycle: {str(e)}")
                # On any exception, check if it might be tracking-related and restart
                if self.tracking_lost:
                    self.get_logger().error("Exception may be related to tracking loss - restarting cycle")
                    continue
                else:
                    break

        # final cleanup
        self.get_logger().info("Feeding complete. Cleaning up...")
        self.servoing_on_pub.publish(Bool(data=False))
        await self.robot_controller.reset()


async def main():
    rclpy.init()
    
    try:
        orchestrator = Orchestrator()
        
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















