import os
import yaml
import serial
import time
import math
import random
import traceback
import threading
from DM_CAN import *

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('motor_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        self.get_logger().info('Subscribed to joint state topic.')
        self.joint_data = {}

    def joint_state_callback(self, msg):
        self.joint_data = dict(zip(msg.name, msg.position))
    
    def get_jointAngle_data(self):
        return self.joint_data
    
class PidGainSubscriber(Node):
    def __init__(self):
        super().__init__('pid_gain_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'pid_gain',
            self.pid_gain_callback,
            10
        )
        self.get_logger().info('Subscribed to pid_gain topic.')
        self.kp_kd_values = []  # 預設值

    def pid_gain_callback(self, msg):
        if len(msg.data) >= 2:
            self.kp_kd_values = [msg.data[0], msg.data[1]]  # Kp, Kd
            # self.get_logger().info(f'Updated PID gains: Kp={self.kp_kd_values[0]}, Kd={self.kp_kd_values[1]}')

    def get_pid_gains(self):
        return self.kp_kd_values

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.joint_state_publisher = self.create_publisher(
            JointState, 
            'low_level_info/joint_states',  # 移除空格，改為小寫
            10
        )
        
class MotorManager:
    def __init__(self, config_path=None):
        self.jointAngleSub = JointStateSubscriber()
        self.pidGainSub = PidGainSubscriber()
        self.jointStatePub = JointStatePublisher() 

        self.control_cmd = DualControlCmd()
        self.lock = threading.Lock()
        self.condition = threading.Condition()
        self.Is_Run = False
        self.run_thread = None
        self.read_thread = None  # 初始化

        self.jointAngle_data = {}
        self.joint_angles = None
        self.kp_kd_list = None

        self.motor_turn = True
        self.command_arrive = True
        self.run_interval = 1 / 50  # _run_motor() 限制到 50Hz

        with open(config_path, "r") as file:
            self.motor_limits = yaml.safe_load(file)["motor_limits"]
    
    def stop_executor(self):
        self.executor.shutdown()
        self.spin_thread.join()  

    def is_within_limits(self, joint_name, angle):
        for motor_group in self.motor_limits.values():
            if joint_name in motor_group:
                lower, upper = motor_group[joint_name]
                return lower <= angle <= upper
        return False  
    
    def get_jointAngle_data(self):
        self.jointAngle_data = self.jointAngleSub.get_jointAngle_data()
        self.kp_kd_list = self.pidGainSub.get_pid_gains()

        required_keys = ['frh', 'fru', 'frd', 'flh', 'flu', 'fld', 'rrh', 'rru', 'rrd', 'rlh', 'rlu', 'rld']
        for key in required_keys:
            if key not in self.jointAngle_data:
                print(f"Joint key '{key}' not found in joint states! Setting default value.")
                self.jointAngle_data[key] = 0.0  # Default safe value
            elif not self.is_within_limits(key, self.jointAngle_data[key]):
                print(f"Warning: {key} angle {self.jointAngle_data[key]} is out of bounds!")
                self.jointAngle_data[key] = max(min(self.jointAngle_data[key], self.motor_limits[key][1]), self.motor_limits[key][0])

        self.joint_angles = np.array([[self.jointAngle_data['frd'],    self.jointAngle_data['fld'], self.jointAngle_data['rrd'],  self.jointAngle_data['rld']],
                                      [self.jointAngle_data['fru'],    self.jointAngle_data['flu'], self.jointAngle_data['rru'],  self.jointAngle_data['rlu']],
                                      [self.jointAngle_data['frh'],    self.jointAngle_data['flh'], self.jointAngle_data['rrh'],  self.jointAngle_data['rlh']]]) 
        
    def _run_motor(self):
        interval = 1.0 / 100  # 150 Hz -> 每次執行間隔 6.67 ms
        prev_time = time.time()
        count = 0

        while self.Is_Run:
            start_time = time.time()
            
            self.get_jointAngle_data()
            self.control_cmd.motor_position_control(self.joint_angles, self.kp_kd_list)

            count += 1
            elapsed_time = time.time() - prev_time
            if elapsed_time >= 1.0:
                print(f"Motor control frequency: {count} Hz")
                count = 0
                prev_time = time.time()

            sleep_time = interval - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _read_motor(self):
        interval = 1.0 / 100  # 150 Hz -> 每次執行間隔 6.67 ms
        prev_time_read = time.time()
        count_read = 0

        msg = JointState()
        msg.name = [
            'frd', 'fld', 'rrd', 'rld',  # Lower legs
            'fru', 'flu', 'rru', 'rlu',  # Upper legs
            'frh', 'flh', 'rrh', 'rlh'   # Hips
        ]

        while self.Is_Run:
            start_time = time.time()
            
            self.control_cmd.update_joint_state()  # 放在 notify() 之後執行

            joint_velocity, joint_position = self.control_cmd.getAllJointState()

            msg.position = joint_position.flatten().tolist()
            msg.velocity = joint_velocity.flatten().tolist()
            msg.effort = []
            self.jointStatePub.joint_state_publisher.publish(msg)

            count_read += 1
            elapsed_time_read = time.time() - prev_time_read
            if elapsed_time_read >= 1.0:
                print(f"Motor reading frequency: {count_read} Hz")
                count_read = 0
                prev_time_read = time.time()

            sleep_time = interval - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run(self):
        if not self.Is_Run:
            self.Is_Run = True
            self.control_cmd.enable_motor()

            self.executor = rclpy.executors.SingleThreadedExecutor()
            self.executor.add_node(self.jointAngleSub)
            self.executor.add_node(self.pidGainSub)
            # self.executor.add_node(self.jointStatePub)
            self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.spin_thread.start()

            self.run_thread = threading.Thread(target=self._run_motor)
            self.run_thread.start()     
            self.read_thread = threading.Thread(target=self._read_motor) 
            self.read_thread.start()   

    def stop(self):
        if self.Is_Run:
            self.Is_Run = False
            self.executor.shutdown()
            if self.spin_thread and self.spin_thread.is_alive():
                self.spin_thread.join()
            if self.run_thread and self.run_thread.is_alive():
                self.run_thread.join()
            if self.read_thread and self.read_thread.is_alive():
                self.read_thread.join()
        print("Motors Set stopped running")

class DualControlCmd:
    """Control two sets of DM_CAN motors using different serial ports."""
    def __init__(self):
        self.setup_serials()
        self.setup_motors()
        self.lock = threading.Lock()
        self.joint_positions = np.zeros((3, 4))
        self.joint_velocity = np.zeros((3, 4))
        
    def getAllJointState(self):

        return self.joint_velocity, self.joint_positions

    # [Previous setup_serials, setup_motors, load_config methods remain the same]
    def setup_serials(self):
        self.serial_device_1 = serial.Serial('/dev/ttyACM1', 921600, timeout=0.5)
        self.motor_control_1 = MotorControl(self.serial_device_1)
        #/dev/ttyRedDogRight
        
        self.serial_device_2 = serial.Serial('/dev/ttyACM0', 921600, timeout=0.5)
        self.motor_control_2 = MotorControl(self.serial_device_2)
        #/dev/ttyRedDogLeft

    def setup_motors(self):
        motor_names_1 = [ 'FR_hip', 'FR_higher', 'FR_lower', 'RR_hip', 'RR_higher', 'RR_lower']
        motor_params_1 = [(0x01, 0x11), (0x02, 0x12), (0x03, 0x13),
                         (0x05, 0x15), (0x06, 0x16), (0x07, 0x17)]

        motor_names_2 = [ 'FL_hip', 'FL_higher', 'FL_lower', 'RL_hip', 'RL_higher', 'RL_lower']
        motor_params_2 = [(0x01, 0x11), (0x02, 0x12), (0x03, 0x13),
                         (0x05, 0x15), (0x06, 0x16), (0x07, 0x17)]

        self.motors_1 = {
            name: Motor(DM_Motor_Type.DM4310, id_1, id_2)
            for name, (id_1, id_2) in zip(motor_names_1, motor_params_1)
        }
        
        self.motors_2 = {
            name: Motor(DM_Motor_Type.DM4310, id_1, id_2)
            for name, (id_1, id_2) in zip(motor_names_2, motor_params_2)
        }

        for motor in self.motors_1.values():
            self.motor_control_1.addMotor(motor)
            if self.motor_control_1.switchControlMode(motor, Control_Type.MIT):
                print(f"DM_CAN Motor {motor.SlaveID} (Set 1): switched to MIT control mode")
            self.motor_control_1.save_motor_param(motor)
            self.motor_control_1.enable(motor)

        for motor in self.motors_2.values():
            self.motor_control_2.addMotor(motor)
            if self.motor_control_2.switchControlMode(motor, Control_Type.MIT):
                print(f"DM_CAN Motor {motor.SlaveID} (Set 2): switched to MIT control mode")
            self.motor_control_2.save_motor_param(motor)
            self.motor_control_2.enable(motor)

        self.leg_motor_list = [
                        [self.motors_1['FR_lower'],   self.motors_2['FL_lower'],   self.motors_1['RR_lower'],    self.motors_2['RL_lower']  ],
                        [self.motors_1['FR_higher'],  self.motors_2['FL_higher'],  self.motors_1['RR_higher'],   self.motors_2['RL_higher']],
                        [self.motors_1['FR_hip'],     self.motors_2['FL_hip'],     self.motors_1['RR_hip'],      self.motors_2['RL_hip']]
        ]

    def reset(self):
        reset_thread = threading.Thread(target=self.motor_position_control)
        reset_thread.start()
        print("Motors Set reset")
        reset_thread.join()
        
    def motor_position_control(self, position=None, kp_kd_list=None):
        if position is None:
            position = [[     3 ,    -3,   -3,     3 ], 
                        [  -1.6 ,   1.6,  1.6,  -1.6 ], 
                        [     0 ,     0,    0,     0]] 

        if kp_kd_list is None:
            kp_kd_list = [ 3, 0.1]
        
        for i, motor_list in enumerate(self.leg_motor_list):
            for j, motor in enumerate(motor_list):
                if j in [0, 2]: 
                    self.motor_control_1.controlMIT(motor, kp_kd_list[0], kp_kd_list[1], position[i][j], 0, 0)
                elif j in [1, 3]:  
                    self.motor_control_2.controlMIT(motor, kp_kd_list[0], kp_kd_list[1], position[i][j], 0, 0)

    def read(self):
        self.refresh_motor()
        self.update_joint_state()
        np.set_printoptions(suppress=True) 
        print(self.joint_positions / math.pi * 180 )
        print(self.joint_positions)

    def refresh_motor(self):
        for i, motor_list in enumerate(self.leg_motor_list):
            for j, motor in enumerate(motor_list):
                if j in [0, 2]: 
                    self.motor_control_1.refresh_motor_status(motor)
                elif j in [1, 3]:  
                    self.motor_control_2.refresh_motor_status(motor)

    def update_joint_state(self):
        for i, motor_list in enumerate(self.leg_motor_list):
            self.joint_positions[i] = [
                motor_list[0].getPosition() , motor_list[1].getPosition() ,
                motor_list[2].getPosition() , motor_list[3].getPosition() 
            ]

            self.joint_velocity[i] = [
                motor_list[0].getVelocity() , motor_list[1].getVelocity() ,
                motor_list[2].getVelocity() , motor_list[3].getVelocity() 
            ]

    def enable_motor(self):
        for i, motor_list in enumerate(self.leg_motor_list):
            for j, motor in enumerate(motor_list):
                if j in [0, 2]: 
                    self.motor_control_1.enable(motor)
                elif j in [1, 3]:  
                    self.motor_control_2.enable(motor)
        print("enable the motor")

    def disable_motor(self):
        for i, motor_list in enumerate(self.leg_motor_list):
            for j, motor in enumerate(motor_list):
                if j in [0, 2]: 
                    self.motor_control_1.disable(motor)
                elif j in [1, 3]:  
                    self.motor_control_2.disable(motor)
        print("Disable the motor")

    def set_zero(self):
        for i, motor_list in enumerate(self.leg_motor_list):
            for j, motor in enumerate(motor_list):
                if j in [0, 2]: 
                    self.motor_control_1.set_zero_position(motor)
                elif j in [1, 3]:  
                    self.motor_control_2.set_zero_position(motor)
        print("Motor zero position set")

    def closeSystem(self):
        """Shut down the system."""
        self.serial_device_1.close()
        self.serial_device_2.close()
        print("System closed")


def main():

    package_path = os.path.dirname(os.path.realpath(__file__))  # Get script directory
    config_path = os.path.join(package_path, "config", "motor_config.yaml")

    rclpy.init()
    motor_manager = MotorManager(config_path)
    
    motor_manager.run()

    # command_dict = {
    #     "r": motor_manager.run,
    #     "stop": motor_manager.stop,
    #     "reset": motor_manager.control_cmd.reset,
    #     "read": motor_manager.control_cmd.read,
    #     "enable": motor_manager.control_cmd.enable_motor,
    #     "disable": motor_manager.control_cmd.disable_motor,
    #     "set" : motor_manager.control_cmd.set_zero,
    # }
    
    # print("Available commands:")
    # print("r - Run all motors")
    # print("stop - Stop all motors")
    # print("reset - Reset all motors")
    # print("read - Read status of all motors")
    # print("enable - Enable all motors")
    # print("disable - Disable all motors")
    # print("exit - Close the system")
    
    # while True:
    #     try:
    #         cmd = input("CMD: ")
    #         if cmd in command_dict:
    #             command_dict[cmd]()
    #         elif cmd == "exit":
    #             motor_manager.stop()
    #             motor_manager.control_cmd.closeSystem()
    #             break
    #         else:
    #             print("Invalid command")
    #     except Exception as e:
    #         traceback.print_exc()
    #         break

if __name__ == '__main__':
    main()