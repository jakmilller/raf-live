from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.10',
        description='IP address of the robot'
    )
    
    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='gen3',
        description='Type/series of robot'
    )
    
    dof_arg = DeclareLaunchArgument(
        'dof',
        default_value='7',
        description="Robot's degrees of freedom"
    )
    
    gripper_arg = DeclareLaunchArgument(
        'gripper',
        default_value='robotiq_2f_140',
        description='Gripper type'
    )

    # robot description direct from pdf
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("kortex_description"), "robots", "kinova.urdf.xacro"]
            ),
            " ",
            "robot_ip:=", LaunchConfiguration('robot_ip'),
            " ",
            "name:=kinova",
            " ",
            "arm:=", LaunchConfiguration('robot_type'),
            " ",
            "gripper:=", LaunchConfiguration('gripper'),
            " ",
            "dof:=", LaunchConfiguration('dof'),
            " ",
            "vision:=true",
            " ",
            "use_fake_hardware:=false",
        ]
    )
    
    robot_description = {"robot_description": robot_description_content}

    # publish transform and description
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )
    
    # we have mounted RealSense camera, create static transform
    # should technically adjust urdf
    realsense_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '-0.03',  # x offset (all offsets calculated from mounted RealSense)
            '-0.003',    # y offset
            '0.031',   # z offset
            '0.0',    # roll
            '0.0',    # pitch
            '0.0',    # yaw
            'camera_link',  # kinova camera frame
            'realsense_link'  # external RealSense frame
        ],
        name='realsense_transform'
    )
    
    # custom controller
    controller_node = Node(
        package='kortex_controller',
        executable='controller',
        name='kortex_controller',
        output='screen',
        parameters=[{
            'robot_ip': LaunchConfiguration('robot_ip'),
            'username': 'admin',
            'password': 'admin',
        }],
        remappings=[
            ('/my_gen3/robot_joint_states', '/joint_states'),  # remap to standard topic
        ]
    )
    
    # # perception node
    # perception_node = Node(
    #     package='perception',
    #     executable='perception_node',
    #     name='perception_node',
    #     output='screen'
    # )

    # RViz for visualization
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("kortex_description"), "rviz", "view_robot.rviz"]
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_file],
        output="screen",
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'rgb_camera.color_profile': '848x480x30',
            'depth_module.profile': '848x480x30',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_rgbd': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true'
        }.items()
    )

    return LaunchDescription([
        robot_ip_arg,
        robot_type_arg,
        dof_arg,
        gripper_arg,
        robot_state_publisher,
        realsense_transform,
        controller_node,
        # perception_node,
        rviz_node,  # remove for no rviz
        realsense_launch
    ])