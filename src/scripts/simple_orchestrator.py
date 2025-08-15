#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
import asyncio
import sys
import os
import yaml

# Import robot controller
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from robot_controller_ros2 import KinovaRobotControllerROS2

class MinimalOrchestrator(Node):
    def __init__(self):
        super().__init__('minimal_orchestrator')
        
        config_path = os.path.expanduser('~/raf-live/config.yaml')
        
        # Load config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.robot_controller = KinovaRobotControllerROS2(config_path)
        
        # State variables
        self.grip_value = None
        self.servoing_complete = False
        
        # Publishers
        self.start_detection_pub = self.create_publisher(Bool, '/start_food_detection', 10)
        self.stop_detection_pub = self.create_publisher(Bool, '/stop_food_detection', 10)
        
        # Subscribers
        self.grip_value_sub = self.create_subscription(
            Float64, '/grip_value', self.grip_value_callback, 10)
        self.finished_servoing_sub = self.create_subscription(
            Bool, '/finished_servoing', self.finished_servoing_callback, 10)
        
        self.get_logger().info('Minimal Orchestrator ready!')
    
    def grip_value_callback(self, msg):
        """Callback for grip value"""
        self.grip_value = msg.data
        self.get_logger().info(f"Received grip value: {self.grip_value}")
    
    def finished_servoing_callback(self, msg):
        """Callback for finished servoing"""
        if msg.data:
            self.servoing_complete = True
            self.get_logger().info("Servoing complete!")
    
    async def run_cycle(self):
        """Main cycle: overlook -> start detection -> wait for servoing -> close gripper"""
        
        try:
            # Step 1: Move to overlook position
            self.get_logger().info("Step 1: Moving to overlook position...")
            if not await self.robot_controller.reset():
                self.get_logger().error("Failed to reset robot!")
                return False
            
            # Step 2: Start food detection
            self.get_logger().info("Step 2: Starting food detection...")
            start_msg = Bool()
            start_msg.data = True
            self.start_detection_pub.publish(start_msg)
            
            # Reset state
            self.servoing_complete = False
            self.grip_value = None
            
            # Step 3: Wait for servoing to complete
            self.get_logger().info("Step 3: Waiting for servoing to complete...")
            while not self.servoing_complete and rclpy.ok():
                await asyncio.sleep(0.1)
                rclpy.spin_once(self, timeout_sec=0)
            
            if not self.servoing_complete:
                self.get_logger().error("Servoing did not complete!")
                return False
            
            # Step 4: Stop food detection
            self.get_logger().info("Step 4: Stopping food detection...")
            stop_msg = Bool()
            stop_msg.data = True
            self.stop_detection_pub.publish(stop_msg)
            
            # Step 5: Close gripper to proper grip value
            if self.grip_value is not None:
                self.get_logger().info(f"Step 5: Closing gripper to {self.grip_value}...")
                if not await self.robot_controller.set_gripper(self.grip_value):
                    self.get_logger().error("Failed to set gripper!")
                    return False
            else:
                self.get_logger().warn("No grip value received, skipping gripper close")
            
            self.get_logger().info("Cycle completed successfully!")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error in cycle: {str(e)}")
            return False

def main(args=None):
    print("Starting minimal orchestrator...")
    rclpy.init(args=args)
    
    try:
        orchestrator = MinimalOrchestrator()
        
        async def run():
            await orchestrator.run_cycle()
        
        # Run the cycle
        asyncio.run(run())
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()