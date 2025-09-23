#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import numpy as np
from raf_interfaces.msg import ForceFeedback

class ForceFeedbackVisualizer(Node):
    def __init__(self):
        super().__init__('force_feedback_visualizer')
        self.get_logger().info("Starting Force Feedback Visualizer...")
        
        # Data storage - keep last 10 seconds of data (at ~10Hz = 100 samples)
        self.max_samples = 100
        self.time_data = deque(maxlen=self.max_samples)
        
        # Force data
        self.force_x = deque(maxlen=self.max_samples)
        self.force_y = deque(maxlen=self.max_samples)
        self.force_z = deque(maxlen=self.max_samples)
        
        # Torque data
        self.torque_x = deque(maxlen=self.max_samples)
        self.torque_y = deque(maxlen=self.max_samples)
        self.torque_z = deque(maxlen=self.max_samples)
        
        # Gripper data
        self.gripper_position = deque(maxlen=self.max_samples)
        self.gripper_velocity = deque(maxlen=self.max_samples)
        self.gripper_effort = deque(maxlen=self.max_samples)
        
        # Time tracking
        self.start_time = self.get_clock().now()
        
        # Thread lock for data access
        self.data_lock = threading.Lock()
        
        # Subscribe to force feedback topic
        self.subscription = self.create_subscription(
            ForceFeedback,
            '/my_gen3/force_feedback',
            self.force_feedback_callback,
            10)
        
        # Set up matplotlib
        self.setup_plots()
        
        self.get_logger().info("Force Feedback Visualizer initialized. Waiting for data...")
    
    def force_feedback_callback(self, msg):
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        with self.data_lock:
            self.time_data.append(current_time)
            
            # Force data
            self.force_x.append(msg.force.x)
            self.force_y.append(msg.force.y)
            self.force_z.append(msg.force.z)
            
            # Torque data  
            self.torque_x.append(msg.torque.x)
            self.torque_y.append(msg.torque.y)
            self.torque_z.append(msg.torque.z)
            
            # Gripper data
            self.gripper_position.append(msg.gripper_position)
            self.gripper_velocity.append(msg.gripper_velocity)
            self.gripper_effort.append(msg.gripper_effort)
    
    def setup_plots(self):
        # Create figure with 3 subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('Real-Time Force/Torque/Gripper Feedback', fontsize=16)
        
        # Force plot - All forces in tool frame
        self.force_lines = {
            'x': self.ax1.plot([], [], 'r-', label='Force X (Tool Frame)', linewidth=2)[0],
            'y': self.ax1.plot([], [], 'g-', label='Force Y (Tool Frame)', linewidth=2)[0],
            'z': self.ax1.plot([], [], 'b-', label='Force Z (Tool Frame)', linewidth=2)[0]
        }
        self.ax1.set_ylabel('Force (N)')
        self.ax1.set_title('Force Feedback (Tool Frame)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(-10, 10)
        
        # Torque plot
        self.torque_lines = {
            'x': self.ax2.plot([], [], 'r-', label='Torque X', linewidth=2)[0],
            'y': self.ax2.plot([], [], 'g-', label='Torque Y', linewidth=2)[0],
            'z': self.ax2.plot([], [], 'b-', label='Torque Z', linewidth=2)[0]
        }
        self.ax2.set_ylabel('Torque (N⋅m)')
        self.ax2.set_title('Torque Feedback')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(-5, 5)
        
        # Gripper plot
        self.gripper_lines = {
            'position': self.ax3.plot([], [], 'purple', label='Position (rad)', linewidth=2)[0],
            'velocity': self.ax3.plot([], [], 'orange', label='Velocity (rad/s)', linewidth=2)[0],
            'effort': self.ax3.plot([], [], 'brown', label='Effort (A)', linewidth=2)[0]
        }
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Gripper Data')
        self.ax3.set_title('Gripper Feedback')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylim(-2, 2)
        
        # Adjust layout
        plt.tight_layout()
    
    def update_plots(self, frame):
        if not self.time_data:
            return []
        
        with self.data_lock:
            times = list(self.time_data)
            
            # Force data
            fx_data = list(self.force_x)
            fy_data = list(self.force_y)
            fz_data = list(self.force_z)
            
            # Torque data
            tx_data = list(self.torque_x)
            ty_data = list(self.torque_y)
            tz_data = list(self.torque_z)
            
            # Gripper data
            gp_data = list(self.gripper_position)
            gv_data = list(self.gripper_velocity)
            ge_data = list(self.gripper_effort)
        
        if len(times) < 2:
            return []
        
        # Update force lines
        self.force_lines['x'].set_data(times, fx_data)
        self.force_lines['y'].set_data(times, fy_data)
        self.force_lines['z'].set_data(times, fz_data)
        
        # Update torque lines
        self.torque_lines['x'].set_data(times, tx_data)
        self.torque_lines['y'].set_data(times, fy_data)
        self.torque_lines['z'].set_data(times, tz_data)
        
        # Update gripper lines
        self.gripper_lines['position'].set_data(times, gp_data)
        self.gripper_lines['velocity'].set_data(times, gv_data)
        self.gripper_lines['effort'].set_data(times, ge_data)
        
        # Auto-scale x-axis to show last 10 seconds
        if times:
            x_max = times[-1]
            x_min = max(0, x_max - 10)
            
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_xlim(x_min, x_max)
            
            # Auto-scale y-axis based on data
            if fx_data and fy_data and fz_data:
                force_data = fx_data + fy_data + fz_data
                f_min, f_max = min(force_data), max(force_data)
                f_range = f_max - f_min
                if f_range > 0:
                    self.ax1.set_ylim(f_min - 0.1*f_range, f_max + 0.1*f_range)
            
            if tx_data and ty_data and tz_data:
                torque_data = tx_data + ty_data + tz_data
                t_min, t_max = min(torque_data), max(torque_data)
                t_range = t_max - t_min
                if t_range > 0:
                    self.ax2.set_ylim(t_min - 0.1*t_range, t_max + 0.1*t_range)
            
            if gp_data and gv_data and ge_data:
                gripper_data = gp_data + gv_data + ge_data
                g_min, g_max = min(gripper_data), max(gripper_data)
                g_range = g_max - g_min
                if g_range > 0:
                    self.ax3.set_ylim(g_min - 0.1*g_range, g_max + 0.1*g_range)
        
        return (list(self.force_lines.values()) + 
                list(self.torque_lines.values()) + 
                list(self.gripper_lines.values()))
    
    def run(self):
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plots, 
            interval=100,  # Update every 100ms
            blit=False,
            cache_frame_data=False
        )
        
        # Show plot
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ForceFeedbackVisualizer()
        
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