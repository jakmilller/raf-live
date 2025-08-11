import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, Vector3, Vector3Stamped, Point
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import rclpy.time
import asyncio
import os

# Import the service definition
from raf_interfaces.srv import SetTwist

"""This node performs visual servoing to bring the food to the mouth.
It calls the SetTwist service, assuming the robot controller node is running separately."""

class VisualServoingRobot(Node):
    def __init__(self):
        super().__init__('visual_servoing_robot')

        # Create a buffer to store transforms
        self.tf_buffer = tf2_ros.Buffer()
        # Create a listener to receive transforms
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Create a client for the SetTwist service
        self.set_twist_client = self.create_client(SetTwist, '/my_gen3/set_twist')
        while not self.set_twist_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /my_gen3/set_twist service to be available...')
        
        # RViz visualization publisher for transformed vector
        self.base_vector_marker_pub = self.create_publisher(
            MarkerArray,
            '/base_frame_vector_markers',
            10
        )
        
        # Camera parameters (will be updated from camera_info)
        self.camera_info = None
        self.camera_frame_id = "" # Will be populated by the camera_info_callback
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        
        # Desired position will be updated based on camera
        self.target_center = (424, 240)
        
        # Control gains
        self.gain_planar = 0.25  # gain for side to side movement
        self.gain_depth = 0.25   # gain for approaching the mouth
        # Subscribe to camera info for intrinsic parameters
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Subscribe to position vector from face detection
        self.vector_sub = self.create_subscription(
            Vector3,
            '/visual_servo_vector',
            self.vector_callback,
            10
        )

        self.get_logger().info("Visual Servoing Robot initialized, using SetTwist service.")

    def camera_info_callback(self, msg):
        """Update camera intrinsic parameters and frame_id"""
        if self.camera_info is None:  # Only update once
            self.camera_info = msg
            self.camera_frame_id = msg.header.frame_id
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.target_center = (534, 434)
            self.get_logger().info(f"Camera info received. Frame ID: '{self.camera_frame_id}'.")
            self.get_logger().info(f"Updated camera parameters: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def publish_base_vector_marker(self, transformed_vector):
        """Publish arrow marker showing transformed vector in base frame"""
        marker_array = MarkerArray()
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "base_link"
        arrow_marker.header.stamp = self.get_clock().now().to_msg()
        arrow_marker.ns = "base_frame_vector"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        scale_factor = 10.0
        start_point = Point(x=0.0, y=0.0, z=0.0)
        end_point = Point(
            x=transformed_vector.vector.x * scale_factor,
            y=transformed_vector.vector.y * scale_factor,
            z=transformed_vector.vector.z * scale_factor
        )
        arrow_marker.points = [start_point, end_point]
        
        arrow_marker.scale.x = 0.02
        arrow_marker.scale.y = 0.04
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 0.5
        arrow_marker.color.a = 0.8
        
        marker_array.markers.append(arrow_marker)
        self.base_vector_marker_pub.publish(marker_array)

    def vector_callback(self, vector_msg):
        """Callback for the vector. Creates an asyncio task to handle the async service call."""
        # This ensures that when the callback is fired by the ROS executor,
        # the async processing is properly scheduled on the running asyncio event loop.
        asyncio.create_task(self.process_vector_async(vector_msg))

    async def process_vector_async(self, vector_msg):
        """Transform vector to base frame and call the twist service."""
        # Prevent processing if we haven't received camera info yet
        if not self.camera_frame_id:
            self.get_logger().warn("Camera info not yet received. Skipping vector processing.")
            return

        try:
            vector = vector_msg
            
            twist = Twist()
            twist.linear.x = self.gain_planar * vector.x
            twist.linear.y = self.gain_planar * vector.y
            twist.linear.z = self.gain_depth * vector.z
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            
            if np.linalg.norm([twist.linear.x, twist.linear.y, twist.linear.z]) < 0.01:
                self.get_logger().info("Movement too small, stopping robot.")
                await self.stop_robot()
                return
            
            # Prepare and call the service
            request = SetTwist.Request()
            request.twist = twist
            request.timeout = 0.0  # A short timeout makes the robot move in small steps
            
            # Asynchronously call the service and wait for the result.
            await self.set_twist_client.call_async(request)
            
            #self.publish_base_vector_marker(transformed_vector)
            
            self.get_logger().info(f"Called SetTwist service: linear=({twist.linear.x:.3f}, {twist.linear.y:.3f}, {twist.linear.z:.3f})")
            
        except Exception as e:
            self.get_logger().error(f'Transform or service call failed: {e}')
            await self.stop_robot()

    async def stop_robot(self):
        """Send zero twist via service to stop the robot."""
        self.get_logger().info("Sending stop command to robot via service...")
        request = SetTwist.Request()
        request.twist = Twist() # Zero twist
        request.timeout = 0.1
        
        # Call the service asynchronously.
        if self.set_twist_client.service_is_ready():
            await self.set_twist_client.call_async(request)
        else:
            self.get_logger().warn("Stop command not sent, service not available.")

async def spin_node_forever(node):
    """Helper function to spin the node in a non-blocking way."""
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
        await asyncio.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    
    visual_servoing_node = None
    try:
        visual_servoing_node = VisualServoingRobot()
        # asyncio.run() creates a new event loop and runs the passed coroutine until it completes.
        # This integrates the ROS spinning with the asyncio event loop.
        asyncio.run(spin_node_forever(visual_servoing_node))

    except KeyboardInterrupt:
        pass
    finally:
        if visual_servoing_node:
            # Ensure the stop command is sent on shutdown
            visual_servoing_node.get_logger().info("Shutting down, sending final stop command.")
            try:
                # Use asyncio.run() again to run the final async stop command.
                asyncio.run(visual_servoing_node.stop_robot())
            except Exception as e:
                 visual_servoing_node.get_logger().error(f"Error while stopping robot on shutdown: {e}")
            
            visual_servoing_node.destroy_node()
            
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
