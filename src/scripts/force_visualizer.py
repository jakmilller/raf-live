#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import threading
from raf_interfaces.msg import ForceFeedback

class Force3DVisualizer(Node):
    def __init__(self):
        super().__init__('force_3d_visualizer')
        self.get_logger().info("Starting 3D Force Visualizer...")
        
        # Current force values
        self.force_x = 0.0
        self.force_y = 0.0
        self.force_z = 0.0
        
        # Thread lock for data access
        self.data_lock = threading.Lock()
        
        # Subscribe to force feedback topic
        self.subscription = self.create_subscription(
            ForceFeedback,
            '/my_gen3/force_feedback',
            self.force_feedback_callback,
            10)
        
        # Set up 3D plot
        self.setup_3d_plot()
        
        self.get_logger().info("3D Force Visualizer initialized. Waiting for data...")
    
    def force_feedback_callback(self, msg):
        with self.data_lock:
            # Note: These are base frame forces as confirmed by API analysis
            self.force_x = msg.force.x
            self.force_y = msg.force.y
            self.force_z = msg.force.z
    
    def setup_3d_plot(self):
        # Create 3D figure and axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up the plot
        self.ax.set_xlabel('X Force (N)', fontsize=12)
        self.ax.set_ylabel('Y Force (N)', fontsize=12)
        self.ax.set_zlabel('Z Force (N)', fontsize=12)
        self.ax.set_title('Real-Time 3D Force Vector (Base Frame)', fontsize=14)
        
        # Set initial axis limits (will be dynamically updated)
        max_force = 10  # Initial limit
        self.ax.set_xlim([-max_force, max_force])
        self.ax.set_ylim([-max_force, max_force])
        self.ax.set_zlim([-max_force, max_force])
        
        # Create origin point
        self.ax.scatter([0], [0], [0], color='black', s=100, label='Origin')
        
        # Initialize force vector (from origin to force point)
        self.force_line, = self.ax.plot([0, 0], [0, 0], [0, 0], 
                                       color='red', linewidth=3, label='Force Vector')
        
        # Initialize force point
        self.force_point = self.ax.scatter([0], [0], [0], 
                                          color='red', s=50, label='Force Point')
        
        # Add coordinate system axes for reference
        axis_length = 5
        self.ax.plot([0, axis_length], [0, 0], [0, 0], 'b-', alpha=0.3, linewidth=1)
        self.ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', alpha=0.3, linewidth=1)
        self.ax.plot([0, 0], [0, 0], [0, axis_length], 'r-', alpha=0.3, linewidth=1)
        
        # Add axis labels at the end of reference axes
        self.ax.text(axis_length, 0, 0, 'X', fontsize=10)
        self.ax.text(0, axis_length, 0, 'Y', fontsize=10)
        self.ax.text(0, 0, axis_length, 'Z', fontsize=10)
        
        # Add legend
        self.ax.legend()
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Force magnitude text
        self.magnitude_text = self.ax.text2D(0.02, 0.95, '', transform=self.ax.transAxes, 
                                           fontsize=12, verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update_plot(self, frame):
        with self.data_lock:
            fx, fy, fz = self.force_x, self.force_y, self.force_z
        
        # Update force vector line (from origin to force point)
        self.force_line.set_data_3d([0, fx], [0, fy], [0, fz])
        
        # Update force point position
        self.force_point._offsets3d = ([fx], [fy], [fz])
        
        # Calculate force magnitude
        magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
        
        # Update magnitude text
        self.magnitude_text.set_text(
            f'Force Vector:\n'
            f'X: {fx:.2f} N\n'
            f'Y: {fy:.2f} N\n'
            f'Z: {fz:.2f} N\n'
            f'Magnitude: {magnitude:.2f} N'
        )
        
        # Dynamically adjust axis limits based on current forces
        max_force = max(10, magnitude * 1.2)  # At least 10N, or 120% of current magnitude
        self.ax.set_xlim([-max_force, max_force])
        self.ax.set_ylim([-max_force, max_force])
        self.ax.set_zlim([-max_force, max_force])
        
        # Color the vector based on magnitude (green for low, red for high)
        if magnitude < 5:
            color = 'green'
        elif magnitude < 10:
            color = 'orange'
        else:
            color = 'red'
        
        self.force_line.set_color(color)
        self.force_point.set_color(color)
        
        return [self.force_line, self.force_point]
    
    def run(self):
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=100,  # Update every 100ms
            blit=False,
            cache_frame_data=False
        )
        
        # Show plot
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = Force3DVisualizer()
        
        # Run ROS2 node in a separate thread
        def spin_node():
            try:
                rclpy.spin(node)
            except Exception as e:
                node.get_logger().error(f"Error in ROS2 spin: {e}")
        
        ros_thread = threading.Thread(target=spin_node, daemon=True)
        ros_thread.start()
        
        # Run matplotlib in main thread
        node.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()