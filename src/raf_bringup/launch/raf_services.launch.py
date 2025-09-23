from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # food detection service node - direct path to source
    food_detection_service_node = Node(
        executable='/home/mcrr-lab/raf-live/src/detection/detection/food_detection_node.py',
        name='food_detection',
        output='screen'
    )

    # face detection service node - direct path to source  
    face_detection_service_node = Node(
        executable='/home/mcrr-lab/raf-live/src/detection/detection/face_detection_node.py',
        name='face_detection',
        output='screen'
    )

    # voice node - direct path to source
    voice_node = Node(
        executable='/home/mcrr-lab/raf-live/src/detection/detection/voice_node.py',
        name='voice_node',
        output='screen'
    )

    # servoing node
    servoing_node = Node(
        package='servoing',
        executable='servoing_node',
        name='servoing_node',
        output='screen'
    )

    # emergency stop monitor
    estop_monitor_node = Node(
        executable='/home/mcrr-lab/raf-live/src/scripts/estop_monitor.py',
        name='estop_monitor',
        output='screen'
    )

    return LaunchDescription([
        food_detection_service_node,
        face_detection_service_node,
        voice_node,
        servoing_node,
        estop_monitor_node
    ])