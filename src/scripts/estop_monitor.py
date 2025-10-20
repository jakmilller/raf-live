#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import serial
import serial.tools.list_ports
import time
import threading
import subprocess


class EstopMonitor(Node):
    def __init__(self):
        super().__init__('estop_monitor')
        self.estop_publisher = self.create_publisher(Bool, '/my_gen3/estop', 10)
        self.serial_connection = None
        self.running = True
        self.realsense_connected = True

        threading.Thread(target=self.monitor_arduino, daemon=True).start()
        threading.Thread(target=self.monitor_realsense, daemon=True).start()

    def find_arduino(self):
        """Find Arduino port"""
        ports = serial.tools.list_ports.comports()

        # Try Arduino-specific ports first
        for port in ports:
            if 'arduino' in port.description.lower() or 'micro' in port.description.lower():
                return port.device
            if port.vid == 0x2341:  # Arduino VID
                return port.device

        # Try common ports
        for port_name in ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0']:
            try:
                test = serial.Serial(port_name, 115200, timeout=1)
                test.close()
                return port_name
            except:
                pass
        return None

    def monitor_arduino(self):
        """Monitor Arduino for ESTOP signal"""
        while self.running:
            try:
                if not self.serial_connection:
                    port = self.find_arduino()
                    if port:
                        self.serial_connection = serial.Serial(port, 115200, timeout=0.1)
                        self.get_logger().info(f"Connected to Arduino at {port}")
                    else:
                        time.sleep(2)
                        continue

                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if line == "ESTOP":
                        self.get_logger().error("🚨 EMERGENCY STOP TRIGGERED! 🚨")
                        msg = Bool()
                        msg.data = True
                        self.estop_publisher.publish(msg)

            except Exception as e:
                self.get_logger().error(f"Arduino error: {e}")
                if self.serial_connection:
                    self.serial_connection.close()
                self.serial_connection = None
                time.sleep(1)

    def check_realsense_connected(self):
        """Check if Intel RealSense is connected"""
        try:
            result = subprocess.run(['rs-enumerate-devices', '-s'],
                                    capture_output=True, text=True, timeout=2)
            return 'Intel RealSense' in result.stdout
        except:
            return False

    def monitor_realsense(self):
        """Monitor Intel RealSense connection status"""
        while self.running:
            try:
                is_connected = self.check_realsense_connected()

                if not is_connected and self.realsense_connected:
                    self.get_logger().error("🚨 REALSENSE DISCONNECTED - EMERGENCY STOP! 🚨")
                    msg = Bool()
                    msg.data = True
                    self.estop_publisher.publish(msg)
                    self.realsense_connected = False
                elif is_connected and not self.realsense_connected:
                    self.get_logger().info("RealSense reconnected")
                    self.realsense_connected = True

                time.sleep(1)

            except Exception as e:
                self.get_logger().error(f"RealSense monitor error: {e}")
                time.sleep(1)


def main():
    rclpy.init()
    monitor = EstopMonitor()
    monitor.get_logger().info("Emergency Stop Monitor Active")

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.running = False
        rclpy.shutdown()


if __name__ == '__main__':
    main()