#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist
from std_msgs.msg import Float64, Bool, Float32
from raf_interfaces.srv import SetTwist
import numpy as np

class ServoingNode(Node):
    def __init__(self):
        super().__init__('servoing_node')
        
        # State variables
        self.position_vector = Vector3()
        self.food_angle = None
        self.kp_linear = Vector3(x=0.65, y=0.65, z=0.65)  # default proportional gains for linear (x,y,z) movement
        self.kd_linear = Vector3(x=0.1, y=0.1, z=0.1)  # default derivative gains for linear (x,y,z) movement
        self.kp_orientation = 0.01  # default proportional gain for orientation (yaw)
        self.kd_orientation = 0.01  # default derivative gain for orientation (yaw)
        self.servoing_on = False
        self.last_position_vector = None
        self.last_food_angle = None
        self.frequency = 10.0  # Hz
        self.dt = 1.0 / self.frequency
        self.max_linear_speed = 0.2  # m/s

        # Subscribers
        self.position_vector_sub = self.create_subscription(
            Vector3, '/position_vector', self.position_vector_callback, 1)
        self.food_angle_sub = self.create_subscription(
            Float64, '/food_angle', self.food_angle_callback, 1)
        
        # these are the p gains for the twist controller
        self.kp_linear_sub = self.create_subscription(
            Vector3, '/twist_gains', self.kp_linear_callback, 1)
        self.servoing_on_sub = self.create_subscription(
            Bool, '/servoing_on', self.servoing_on_callback, 1)
        
        # Service client for sending twist commands
        self.set_twist_client = self.create_client(SetTwist, '/my_gen3/set_twist')
        while not self.set_twist_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /my_gen3/set_twist service...')
        
        # Timer for control loop (50Hz)
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info('Servoing Node initialized')
        self.get_logger().info(f'Default linear p gains: planar={self.kp_linear.x}, depth={self.kp_linear.z}')
    
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
            # self.get_logger().info(f"Position vector magnitude: {magnitude:.3f}")
    
    def food_angle_callback(self, msg):
        """Handle incoming food angle"""
        self.food_angle = float(msg.data)
    
    def kp_linear_callback(self, msg):
        """Update proportional linear gains"""
        self.kp_linear_gains = msg
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
        """Main control loop - runs at 10 Hz, only sends commands when servoing is on"""
        if not self.servoing_on:
            return
        
        # Calculate twist command from position vector using PD control
        twist = Twist()

        # Proportional control
        twist.linear.x = (self.kp_linear.x * self.position_vector.x)
        twist.linear.y = (self.kp_linear.y * self.position_vector.y)
        twist.linear.z = (self.kp_linear.z * self.position_vector.z)
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = self.kp_orientation * (self.food_angle if self.food_angle is not None else 0.0)

        # if its not the first run, caluclate the derivative control
        if self.last_position_vector is not None:
            twist.linear.x += self.kd_linear.x * ((self.position_vector.x - self.last_position_vector.x) / self.dt)
            twist.linear.y += self.kd_linear.y * ((self.position_vector.y - self.last_position_vector.y) / self.dt)
            twist.linear.z += self.kd_linear.z * ((self.position_vector.z - self.last_position_vector.z) / self.dt)
        else:
            self.get_logger().info("Derivative term not added to linear twist signal")

        if self.last_food_angle is not None and self.food_angle is not None:
            twist.angular.z += float(self.kd_orientation * ((self.food_angle - self.last_food_angle) / self.dt))
        else:
            self.get_logger().info("Derivative term not added to angular twist signal")

        # check to make sure we dont exceed max speed
        linear_speed = np.linalg.norm([twist.linear.x, twist.linear.y, twist.linear.z])
        if linear_speed > self.max_linear_speed:
            scale = linear_speed / self.max_linear_speed
            twist.linear.x *= scale
            twist.linear.y *= scale
            twist.linear.z *= scale
            self.get_logger().warn(f"Linear speed capped to {self.max_linear_speed} m/s")        

        # Send twist command
        self.send_twist_command(twist)
        # self.get_logger().info(f"Sent twist command: linear=({twist.linear.x:.3f}, {twist.linear.y:.3f}, {twist.linear.z:.3f}), angular_z={twist.angular.z:.3f}")

        self.last_position_vector = self.position_vector
        self.last_food_angle = self.food_angle
    
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
    node = ServoingNode()
    
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