from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='reddog_hardware',
            executable='motor_manager',
            name='motor_manager',
            output='screen',
        ),        
        Node(
            package='reddog_imu',
            executable='imu_estimate',
            name='imu_estimate',
            output='screen'
        ),
    ])