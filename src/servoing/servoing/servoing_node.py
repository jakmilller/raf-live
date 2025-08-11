#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist
from std_msgs.msg import Float64, Bool
from raf_interfaces.srv import SetTwist
import numpy as np

class ServoingNode(Node):
    def __init__(self):
        super().__init__('servoing_node')
        
        # State variables
        self.position_vector = Vector3()
        self.food_angle = Float64()
        self.linear_twist_gains = Vector3(x=0.4, y=0.4, z=0.4)  # Default gains
        self.orientation_gain = 0.008  # Default orientation gain
        self.min_distance = 0.01  # Default minimum distance (1.5cm)
        self.servoing_active = False
        
        # Subscribers
        self.position_vector_sub = self.create_subscription(
            Vector3, '/position_vector', self.position_vector_callback, 10)
        self.food_angle_sub = self.create_subscription(
            Float64, '/food_angle', self.food_angle_callback, 10)
        self.linear_twist_gains_sub = self.create_subscription(
            Vector3, '/twist_gains', self.linear_twist_gains_callback, 10)
        self.min_distance_sub = self.create_subscription(
            Float64, '/min_distance', self.min_distance_callback, 10)
        
        # Publishers
        self.finished_servoing_pub = self.create_publisher(Bool, '/finished_servoing', 10)
        
        # Service client for sending twist commands
        self.set_twist_client = self.create_client(SetTwist, '/my_gen3/set_twist')
        while not self.set_twist_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /my_gen3/set_twist service...')
        
        # Timer for control loop (50Hz)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        # State tracking
        self.last_vector_time = self.get_clock().now()
        self.zero_vector_count = 0
        self.max_zero_count = 10  # Stop after 10 consecutive zero vectors
        
        self.get_logger().info('Servoing Node initialized')
        self.get_logger().info(f'Default gains: planar={self.linear_twist_gains.x}, depth={self.linear_twist_gains.z}')
        self.get_logger().info(f'Default min distance: {self.min_distance}m')
    
    def position_vector_callback(self, msg):
        """Handle incoming position vectors"""
        self.position_vector = msg
        self.last_vector_time = self.get_clock().now()
        
        # Check if this is a zero vector
        magnitude = np.linalg.norm([msg.x, msg.y, msg.z])
        
        if magnitude < 0.001:  # Essentially zero
            self.zero_vector_count += 1
            if not self.servoing_active:
                return  # Don't log if we're already stopped
            
            if self.zero_vector_count >= self.max_zero_count:
                self.get_logger().info("Received consecutive zero vectors, stopping servoing")
                self.servoing_active = False
                self.send_zero_twist()
                self.finished_servoing_pub.publish(Bool(data=True))
        else:
            # Non-zero vector received, start/continue servoing
            if not self.servoing_active:
                self.get_logger().info("Starting servoing - received position vector")
                self.servoing_active = True
            
            self.zero_vector_count = 0
            self.get_logger().info(f"Position vector: ({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}), "
                                 f"magnitude: {magnitude:.3f}")
            
    def food_angle_callback(self, msg):
        """Handle incoming food angle"""
        self.food_angle = msg
        # self.get_logger().info(f"Received food angle: {msg.data:.2f} radians")

    
    def linear_twist_gains_callback(self, msg):
        """Update twist gains"""
        self.linear_twist_gains = msg
        self.get_logger().info(f"Updated gains: x={msg.x}, y={msg.y}, z={msg.z}")
    
    def min_distance_callback(self, msg):
        """Update minimum distance threshold"""
        self.min_distance = msg.data
        self.get_logger().info(f"Updated min distance: {self.min_distance}m")
    
    def control_loop(self):
        """Main control loop - runs at 50Hz"""
        if not self.servoing_active:
            return
        
        # Check for timeout (no position vector updates)
        time_since_update = (self.get_clock().now() - self.last_vector_time).nanoseconds / 1e9
        if time_since_update > 1.0:  # 1 second timeout
            self.get_logger().warn("Position vector timeout - stopping servoing")
            self.servoing_active = False
            self.send_zero_twist()
            self.finished_servoing_pub.publish(Bool(data=True))
            return
        
        # Calculate distance to target
        planar_distance = np.linalg.norm([
            self.position_vector.x, 
            self.position_vector.y
        ])
        
        # Check if we've reached the target
        if self.position_vector.z < self.min_distance and planar_distance < 0.0005:
            self.get_logger().info(f"Reached target (distance: {self.position_vector.z:.3f}m < {self.min_distance}m)")
            self.servoing_active = False
            self.send_zero_twist()
            self.finished_servoing_pub.publish(Bool(data=True))
            return
        
        # Calculate twist command
        twist = Twist()
        twist.linear.x = self.linear_twist_gains.x * self.position_vector.x
        twist.linear.y = self.linear_twist_gains.y * self.position_vector.y

        if self.position_vector.z < self.min_distance:
            twist.linear.z = 0.0
        else:
            twist.linear.z = self.linear_twist_gains.z * self.position_vector.z

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = self.orientation_gain * self.food_angle.data
        print(f"Twist command: linear=({twist.linear.x:.3f}, {twist.linear.y:.3f}, {twist.linear.z:.3f}),angular=({twist.angular.x:.3f}, {twist.angular.y:.3f}, {twist.angular.z:.3f})")
        
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
            
            # Log occasionally (every 25 calls = ~0.5 seconds at 50Hz)
            if hasattr(self, '_log_counter'):
                self._log_counter += 1
            else:
                self._log_counter = 0
            
            if self._log_counter % 25 == 0:
                linear_mag = np.linalg.norm([twist.linear.x, twist.linear.y, twist.linear.z])
                self.get_logger().info(f"Twist command: linear_mag={linear_mag:.3f}")
                
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