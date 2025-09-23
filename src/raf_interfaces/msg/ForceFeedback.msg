# Custom message for force feedback including gripper data
# Force and torque values from the robot's force/torque sensor (tare-compensated)
geometry_msgs/Vector3 force      # Force in Newtons (x, y, z)
geometry_msgs/Vector3 torque     # Torque in Newton-meters (x, y, z)

# Gripper feedback data
float64 gripper_position         # Gripper position (0.0 to 0.81 radians for robotiq_2f_85)
float64 gripper_velocity         # Gripper velocity (rad/s)
float64 gripper_effort           # Gripper motor current (effort)