#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import serial
import serial.tools.list_ports
import time
import threading


class EstopMonitor(Node):
    def __init__(self):
        super().__init__('estop_monitor')
        self.estop_publisher = self.create_publisher(Bool, '/my_gen3/estop', 10)
        self.serial_connection = None
        self.running = True

        threading.Thread(target=self.monitor_arduino, daemon=True).start()

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