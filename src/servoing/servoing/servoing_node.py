#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist
from std_msgs.msg import Float64, Bool
from raf_interfaces.srv import SetTwist
import numpy as np

class SimplifiedServoingNode(Node):
    def __init__(self):
        super().__init__('simplified_servoing_node')
        
        # State variables
        self.position_vector = Vector3()
        self.food_angle = Float64()
        self.linear_twist_gains = Vector3(x=0.4, y=0.4, z=0.4)  # Default gains
        self.orientation_gain = 0.008  # Default orientation gain
        self.servoing_on = False
        
        # Subscribers
        self.position_vector_sub = self.create_subscription(
            Vector3, '/position_vector', self.position_vector_callback, 1)
        self.food_angle_sub = self.create_subscription(
            Float64, '/food_angle', self.food_angle_callback, 1)
        self.linear_twist_gains_sub = self.create_subscription(
            Vector3, '/twist_gains', self.linear_twist_gains_callback, 1)
        self.servoing_on_sub = self.create_subscription(
            Bool, '/servoing_on', self.servoing_on_callback, 1)
        
        # Service client for sending twist commands
        self.set_twist_client = self.create_client(SetTwist, '/my_gen3/set_twist')
        while not self.set_twist_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /my_gen3/set_twist service...')
        
        # Timer for control loop (50Hz)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info('Simplified Servoing Node initialized')
        self.get_logger().info(f'Default gains: planar={self.linear_twist_gains.x}, depth={self.linear_twist_gains.z}')
    
    def position_vector_callback(self, msg):
        """Handle incoming position vectors"""
        self.position_vector = msg
        
        # Optional: log magnitude periodically for debugging
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
        
        # Log every 50 messages (~1 second at 50Hz)
        if self._log_counter % 50 == 0:
            magnitude = np.linalg.norm([msg.x, msg.y, msg.z])
            self.get_logger().info(f"Position vector magnitude: {magnitude:.3f}")
    
    def food_angle_callback(self, msg):
        """Handle incoming food angle"""
        self.food_angle = msg
    
    def linear_twist_gains_callback(self, msg):
        """Update twist gains"""
        self.linear_twist_gains = msg
        self.get_logger().info(f"Updated gains: x={msg.x}, y={msg.y}, z={msg.z}")
    
    def servoing_on_callback(self, msg):
        """Handle servoing on/off signal"""
        self.servoing_on = msg.data
        if self.servoing_on:
            self.get_logger().info("Servoing enabled")
        else:
            self.get_logger().info("Servoing disabled")
            # Send zero twist immediately when disabled
            self.send_zero_twist()
    
    def control_loop(self):
        """Main control loop - runs at 50Hz, only sends commands when servoing is on"""
        if not self.servoing_on:
            return
        
        # Calculate twist command from position vector
        twist = Twist()
        twist.linear.x = self.linear_twist_gains.x * self.position_vector.x
        twist.linear.y = self.linear_twist_gains.y * self.position_vector.y
        twist.linear.z = self.linear_twist_gains.z * self.position_vector.z
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = self.orientation_gain * self.food_angle.data
        
        # Send twist command
        self.send_twist_command(twist)
    
    def send_twist_command(self, twist):
        """Send twist command to robot"""
        try:
            request = SetTwist.Request()
            request.twist = twist
            request.timeout = 0.0  # Continuous mode
            
            # Non-blocking service call
            future = self.set_twist_client.call_async(request)
            
            # Less verbose logging
            if hasattr(self, '_twist_log_counter'):
                self._twist_log_counter += 1
            else:
                self._twist_log_counter = 0
            
            # Log every 100 calls (~2 seconds at 50Hz)
            if self._twist_log_counter % 100 == 0:
                linear_mag = np.linalg.norm([twist.linear.x, twist.linear.y, twist.linear.z])
                self.get_logger().info(f"Servoing active - linear magnitude: {linear_mag:.3f}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to send twist command: {e}")
    
    def send_zero_twist(self):
        """Send zero twist to stop the robot"""
        try:
            request = SetTwist.Request()
            request.twist = Twist()  # Zero twist
            request.timeout = 0.1
            
            future = self.set_twist_client.call_async(request)
            self.get_logger().info("Sent stop command (zero twist)")
            
        except Exception as e:
            self.get_logger().error(f"Failed to send stop command: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SimplifiedServoingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Servoing node interrupted")
    finally:
        # Send final stop command
        node.send_zero_twist()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()