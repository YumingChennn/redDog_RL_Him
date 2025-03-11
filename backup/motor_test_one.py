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

        
class MotorManager:
    def __init__(self, config_path="config/motor_config.yaml"):

        self.control_cmd = DualControlCmd()
        self.lock = threading.Lock()
        self.condition = threading.Condition()
        self.Is_Run = False
        self.run_thread = None

        self.jointAngle_data = {}
        self.joint_angles = None
        self.kp_kd_list = None

        self.motor_turn = True

        with open(config_path, "r") as file:
            self.motor_limits = yaml.safe_load(file)["motor_limits"]
    
    def stop_executor(self):
        self.executor.shutdown()
        self.spin_thread.join()  

    def _run_motor(self):
        prev_time = time.time()
        count = 0

        while self.Is_Run:
            
            # with self.condition:
            #     while not self.motor_turn:  # 只有 motor_turn=True 時才執行
            #         self.condition.wait()
                
            #     self.get_jointAngle_data()
            #     self.control_cmd.motor_position_control(self.joint_angles, self.kp_kd_list)
                
            #     self.motor_turn = False  # **切換權限給 _read_motor()**
            #     self.condition.notify()  # 通知 _read_motor() 可以執行

            # self.get_jointAngle_data()
            self.control_cmd.motor_position_control(np.zeros((1, 1)), self.kp_kd_list)


            count += 1
            print("start",count)
            elapsed_time = time.time() - prev_time
            if elapsed_time >= 1.0:
                print(f"Motor control frequency: {count} Hz")
                count = 0
                prev_time = time.time()


    def run(self):
        if not self.Is_Run:
            self.Is_Run = True
            self.control_cmd.enable_motor()


            self.run_thread = threading.Thread(target=self._run_motor)
            self.run_thread.start()     
            # self.read_thread = threading.Thread(target=self._read_motor) 
            # self.read_thread.start()   

class DualControlCmd:
    """Control two sets of DM_CAN motors using different serial ports."""
    def __init__(self):
        self.setup_serials()
        self.setup_motors()
        self.lock = threading.Lock()
        self.joint_positions = np.zeros((3, 4))

    # [Previous setup_serials, setup_motors, load_config methods remain the same]
    def setup_serials(self):
        self.serial_device_1 = serial.Serial('/dev/ttyRedDogRight', 921600, timeout=0.5)
        self.motor_control_1 = MotorControl(self.serial_device_1)
        
        self.serial_device_2 = serial.Serial('/dev/ttyRedDogLeft', 921600, timeout=0.5)
        self.motor_control_2 = MotorControl(self.serial_device_2)

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
                        [self.motors_1['FR_hip']]
        ]

        # self.leg_motor_list = [
        #                 [self.motors_1['FR_lower'],   self.motors_2['FL_lower'],   self.motors_1['RR_lower'],    self.motors_2['RL_lower']  ],
        #                 [self.motors_1['FR_higher'],  self.motors_2['FL_higher'],  self.motors_1['RR_higher'],   self.motors_2['RL_higher']],
        #                 [self.motors_1['FR_hip'],     self.motors_2['FL_hip'],     self.motors_1['RR_hip'],      self.motors_2['RL_hip']]
        # ]

        
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

    def enable_motor(self):
        for i, motor_list in enumerate(self.leg_motor_list):
            for j, motor in enumerate(motor_list):
                if j in [0, 2]: 
                    self.motor_control_1.enable(motor)
                elif j in [1, 3]:  
                    self.motor_control_2.enable(motor)
        print("enable the motor")

    def closeSystem(self):
        """Shut down the system."""
        self.serial_device_1.close()
        self.serial_device_2.close()
        print("System closed")


def main():
    rclpy.init()
    motor_manager = MotorManager()
    
    command_dict = {
        "r": motor_manager.run,
        "enable": motor_manager.control_cmd.enable_motor,
    }
    
    print("Available commands:")
    print("r - Run all motors")
    print("stop - Stop all motors")
    print("reset - Reset all motors")
    print("read - Read status of all motors")
    print("enable - Enable all motors")
    print("disable - Disable all motors")
    print("exit - Close the system")
    
    while True:
        try:
            cmd = input("CMD: ")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                motor_manager.stop()
                motor_manager.control_cmd.closeSystem()
                break
            else:
                print("Invalid command")
        except Exception as e:
            traceback.print_exc()
            break

if __name__ == '__main__':
    main()