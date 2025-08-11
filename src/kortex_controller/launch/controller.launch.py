from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    username_arg = DeclareLaunchArgument(
        'username',
        default_value='admin',
        description='Username for robot connection'
    )
    
    password_arg = DeclareLaunchArgument(
        'password',
        default_value='admin',
        description='Password for robot connection'
    )
    
    ip_address_arg = DeclareLaunchArgument(
        'ip_address',
        default_value='192.168.1.10',
        description='IP address of the robot'
    )
    
    session_timeout_arg = DeclareLaunchArgument(
        'api_session_inactivity_timeout_ms',
        default_value='35000',
        description='Session inactivity timeout in milliseconds'
    )
    
    connection_timeout_arg = DeclareLaunchArgument(
        'api_connection_inactivity_timeout_ms',
        default_value='20000',
        description='Connection inactivity timeout in milliseconds'
    )

    # Create the controller node
    controller_node = Node(
        package='kortex_controller',
        executable='controller',
        name='kortex_controller',
        output='screen',
        parameters=[{
            'username': LaunchConfiguration('username'),
            'password': LaunchConfiguration('password'),
            'ip_address': LaunchConfiguration('ip_address'),
            'api_session_inactivity_timeout_ms': LaunchConfiguration('api_session_inactivity_timeout_ms'),
            'api_connection_inactivity_timeout_ms': LaunchConfiguration('api_connection_inactivity_timeout_ms'),
        }]
    )

    return LaunchDescription([
        username_arg,
        password_arg,
        ip_address_arg,
        session_timeout_arg,
        connection_timeout_arg,
        controller_node
    ])