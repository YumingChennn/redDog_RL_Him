from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    xsens_launch_file = os.path.join(get_package_share_directory('xsens_mti_ros2_driver'), 'launch', 'xsens_mti_node.launch.py')

    return LaunchDescription([
        # 啟動 motor_manager
        Node(
            package='reddog_hardware',
            executable='motor_manager',
            name='motor_manager',
            output='screen',
        ),

        # 啟動 xsens_mti_ros2_driver 的 launch 文件
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(xsens_launch_file)
        ),

        # 啟動 joy_controller
        Node(
            package='rl_quadruped_controller',
            executable='joy_controller',
            name='joy_controller',
            output='screen'
        ),

        # 啟動 joy_node
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        ),
    ])
