import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import time
import torch
import yaml
import matplotlib.pyplot as plt
import argparse
import threading

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('low_level_info_motor_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            '/low_level_info/joint_states',
            self.joint_state_callback,
            10
        )
        self.get_logger().info('Subscribed to joint state topic.')
        
        self.jointAngle_data = {}
        self.jointVelocity_data = {}

    def joint_state_callback(self, msg):
        joint_names = msg.name
        positions = msg.position
        velocities = msg.velocity

        for i, name in enumerate(joint_names):
            self.jointAngle_data[name] = positions[i]
            self.jointVelocity_data[name] = velocities[i]
    
    def get_jointAngle_data(self):
        return self.jointAngle_data, self.jointVelocity_data
    
class ImuSubscriber(Node):
    def __init__(self):
        super().__init__('low_level_Imu_subscriber')
        self.subscription = self.create_subscription(
            Imu,
            '/low_level_info/imu/data_raw',
            self.joint_state_callback,
            10
        )
        self.get_logger().info('Subscribed to Imu topic.')
        
        self.orientation = None
        self.angular_velocity = None  # Add this

    def joint_state_callback(self, msg):
        self.orientation = msg.orientation
        self.angular_velocity = msg.angular_velocity
    
    def get_Imu_data(self):
        return self.orientation, self.angular_velocity

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.kp_kq_publisher_ = self.create_publisher(Float32MultiArray, 'pid_gain', 10)
        self.timer_joint = self.create_timer(0.02, self.publish_joint_states)
        self.timer_pidgain = self.create_timer(0.02, self.publish_kp_kq)

        self.imu_sub = ImuSubscriber()
        self.joint_state_sub = JointStateSubscriber()

        self.initial_joint_angles = np.array([
            [ 0.0,   0.0,  0.0,   0.0 ],
            [ 1.6, - 1.6, -1.6,   1.6 ], 
            [  -3,     3,    3,    -3 ], 
        ])

        self.default_joint_angles = np.array([
            [  0.0,   0.0,   0.0, 0.0],     # Hip
            [  2.4,  -2.4, -2.4,  2.4],  # Upper leg
            [ -2.7,   2.7,  2.7, -2.7]  # Lower leg

        ])

        self.command_joint_angles = np.array([
            [    0.1,   -0.1,    0.1,   -0.1],   # Hip
            [  0.785, -0.785, -0.785,  0.785],  # Upper leg
            [  -1.57,   1.57,   1.57,  -1.57]  # Lower leg

        ])

        self.current_angles = self.initial_joint_angles.copy()
        self.target_angles = None
        self.start_time = None 
        self.transition_duration = 1.2
        self.command_type = None  
        self.last_command = None

        self.jointAngle_data = {}
        self.jointVelocity_data = {}
        self.dof_pos = []
        self.dof_vel = []
        self.orientation = None
        self.angular_velocity = None  # Add this

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.joint_state_sub)
        executor.add_node(self.imu_sub)
        spin_thread = threading.Thread(target= executor.spin, daemon=True)
        spin_thread.start()
        

    def get_Imu_data(self):
        self.orientation, self.angular_velocity = self.imu_sub.get_Imu_data()
    
    def get_Joint_state_data(self):
        self.jointAngle_data, self.jointVelocity_data = self.joint_state_sub.get_jointAngle_data()
        
        self.dof_pos = [self.jointAngle_data['flh'],  self.jointAngle_data['flu'],  self.jointAngle_data['fld'],  
                        self.jointAngle_data['frh'],  - self.jointAngle_data['fru'],  - self.jointAngle_data['frd'], 
                        self.jointAngle_data['rlh'],  self.jointAngle_data['rlu'],  self.jointAngle_data['rld'],    
                        self.jointAngle_data['rrh'],  - self.jointAngle_data['rru'],  - self.jointAngle_data['rrd']]
    
        self.dof_vel = [self.jointVelocity_data['flh'],  self.jointVelocity_data['flu'],  self.jointVelocity_data['fld'],  
                        self.jointVelocity_data['frh'],  - self.jointVelocity_data['fru'],  - self.jointVelocity_data['frd'], 
                        self.jointVelocity_data['rlh'],  self.jointVelocity_data['rlu'],  self.jointVelocity_data['rld'],    
                        self.jointVelocity_data['rrh'],  - self.jointVelocity_data['rru'],  - self.jointVelocity_data['rrd']]

    def publish_joint_states(self):
        msg = JointState()
        msg.name = [
            'flh', 'frh', 'rlh', 'rrh',   # Hips
            'flu', 'fru', 'rlu', 'rru',  # Upper legs
            'fld', 'frd', 'rld', 'rrd'   # Lower legs
        ]
        msg.position = np.array(self.current_angles).flatten().tolist()
        msg.velocity = []
        msg.effort = []
        self.publisher_.publish(msg)
    
    def convert_joint_angles(self, dof_pos):
        mapping = ['flh', 'frh', 'rlh', 'rrh',  # Hips
                'flu', 'fru', 'rlu', 'rru',  # Upper legs
                'fld', 'frd', 'rld', 'rrd']  # Lower legs

        format1_order = ['flh', 'flu', 'fld',  
                        'frh', 'fru', 'frd', 
                        'rlh', 'rlu', 'rld',    
                        'rrh', 'rru', 'rrd']
        
        index_map = [format1_order.index(name) for name in mapping]

        reordered_dof_pos = [dof_pos[i] for i in index_map]

        reordered_dof_pos = [ reordered_dof_pos[0],   reordered_dof_pos[1], reordered_dof_pos[2], reordered_dof_pos[3],
                            reordered_dof_pos[4], - reordered_dof_pos[5], reordered_dof_pos[6], -reordered_dof_pos[7],
                            reordered_dof_pos[8], - reordered_dof_pos[9], reordered_dof_pos[10], -reordered_dof_pos[11]]

        return reordered_dof_pos
    
    def publish_kp_kq(self):
        msg = Float32MultiArray()
        msg.data = [float(self.kp), float(self.kq)]
        self.kp_kq_publisher_.publish(msg)

    def send_cmd(self):
        self.publish_joint_states()
        self.publish_kp_kq()

    def move_to_default_pos(self):
        self.kp = 3
        self.kq = 0.1
        self.current_angles = self.default_joint_angles.copy()    
        start_time = time.time()  # 記錄開始時間

        while time.time() - start_time < 15:  # 讓它跑 10 秒
            self.send_cmd()
            time.sleep(0.02)  # 50 Hz
            
        print("Moving to default pos completed.")

    def ready_to_standup(self):
        self.target_angles = self.command_joint_angles
        self.start_time = time.time()  # 確保開始時間被設置
        self.kp = 10
        self.kq = 0.4

        while True:
            elapsed_time = time.time() - self.start_time
            phase = np.tanh(elapsed_time / self.transition_duration)

            self.current_angles = phase * self.target_angles + (1 - phase) * self.default_joint_angles
            self.send_cmd()

            if np.allclose(self.current_angles, self.target_angles, atol=0.01):
                self.current_angles = self.target_angles
                self.target_angles = None
                self.command_type = None
                self.get_logger().info("Movement completed, ready for next command.")
                return  
            time.sleep(0.02)  # 50 Hz
        
    def keep_pos(self):
        start_time = time.time()  # 記錄開始時間
        while time.time() - start_time < 5:  # 讓它跑 10 秒
            self.send_cmd()
            time.sleep(0.02)  # 50 Hz
            
        print("Keeping the pos completed.")

        
    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def run(self, config_file):
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print("Config loaded successfully. Keys:", config.keys())

            policy_path = config["policy_path"]
            print("Attempting to load policy from:", policy_path)
            try:
                policy = torch.jit.load(policy_path)
                print("Policy loaded successfully")
            except Exception as e:
                print("Failed to load policy:", e)
                return
            
            print("Loading other config parameters...")
            xml_path = config["xml_path"]
            simulation_duration = config["simulation_duration"]
            simulation_dt = config["simulation_dt"]
            control_decimation = config["control_decimation"]

            kps = np.array(config["kps"], dtype=np.float32)
            kds = np.array(config["kds"], dtype=np.float32)

            default_angles = np.array(config["default_angles"], dtype=np.float32)

            ang_vel_scale = config["ang_vel_scale"]
            dof_pos_scale = config["dof_pos_scale"]
            dof_vel_scale = config["dof_vel_scale"]
            action_scale = config["action_scale"]
            cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            num_actions = config["num_actions"]
            num_obs = config["num_obs"]
            one_step_obs_size = config["one_step_obs_size"]
            obs_buffer_size = config["obs_buffer_size"]
            
            cmd = np.array(config["cmd_init"], dtype=np.float32)
            print("Config parameters loaded")
        
        # target_dof_pos = default_angles.copy()
        # self.current_angles = default_angles
        action = np.zeros(num_actions, dtype=np.float32)
        obs = np.zeros(num_obs, dtype=np.float32)

        lin_vel_data_list = []
        ang_vel_data_list = []
        gravity_b_list = []
        joint_vel_list = []
        action_list = []
    
        self.kp = 10  #10
        self.kq = 0.4 #0.4

        interval = 1.0 / 50  # 150 Hz -> 每次執行間隔 6.67 ms
        
        print("Entering main loop...")
        while True:
                start_time = time.time()

            # create observation
                self.get_Imu_data()
                self.get_Joint_state_data()

                qpos = np.array(self.dof_pos, dtype=np.float32)
                qvel = np.array(self.dof_vel, dtype=np.float32)
                ang_vel_I = np.array([self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z], dtype=np.float32)
                # gravity_b = self.get_gravity_orientation(np.array([self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z], dtype=np.float32))
                gravity_b = np.array([ 0, 0, -1], dtype=np.float32)
                cmd_vel = np.array(config["cmd_init"], dtype=np.float32)

                print(f"cmd_vel: {cmd_vel}")
                print(f"ang_vel_I: {ang_vel_I}")
                print(f"gravity_b: {gravity_b}")
                print(f"qpos: {qpos}")
                print(f"qvel: {qvel}")

                obs_list = []
                obs_list.append(cmd_vel * cmd_scale)
                obs_list.append(ang_vel_I * ang_vel_scale)
                obs_list.append(gravity_b)
                obs_list.append((qpos - default_angles) * dof_pos_scale)
                obs_list.append(qvel * dof_vel_scale)
                obs_list.append(action)
                ## Record Data ##
                ang_vel_data_list.append(ang_vel_I * ang_vel_scale)
                gravity_b_list.append(gravity_b)
                joint_vel_list.append(qvel * dof_vel_scale)
                action_list.append(action)
                ###
                obs_list = [torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs for obs in obs_list]
                obs_tensor_buf = torch.zeros((1, one_step_obs_size * obs_buffer_size))
                # print(f"Before concatenating, obs_list shapes: {[obs.shape for obs in obs_list]}")
                obs = torch.cat(obs_list, dim=0).unsqueeze(0)
                # print(f"After concatenating, obs shape: {obs.shape}")
                obs_tensor = torch.clamp(obs, -100, 100)
                # print(f"obs_tensor_buf shape before: {obs_tensor_buf.shape}")
                # print(f"obs_tensor shape: {obs_tensor.shape}")

                # obs_tensor_buf = torch.cat([
                #     obs_tensor,
                #     obs_tensor_buf[:, :obs_buffer_size * one_step_obs_size - one_step_obs_size]
                # ], dim=1)
                obs_tensor_buf = torch.cat([
                    obs_tensor,
                    obs_tensor_buf[:, :obs_buffer_size * one_step_obs_size - one_step_obs_size]
                ], dim=1)
                # print(f"obs_tensor_buf shape before: {obs_tensor_buf.shape}")
                # print(f"obs_tensor shape: {obs_tensor.shape}")

                # policy inference
                action = policy(obs_tensor_buf).detach().numpy().squeeze()
                # print("action :", action)
                # current_angles = default_angles.copy()
                current_angles = action * action_scale + default_angles
                print(f"action: {action}")
                self.current_angles = self.convert_joint_angles(current_angles)
                self.send_cmd()  

                sleep_time = interval - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)


def main(args=None):
    parser = argparse.ArgumentParser(description='Controller for Quadruped.')
    parser.add_argument('config_file', type=str, help='Path to configuration file.')
    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    
    print(f"Using config file: {args.config_file}")

    controller = MotorController()

    controller.move_to_default_pos()

    controller.ready_to_standup()

    controller.keep_pos()

    while True:
        try:
            controller.run(args.config_file)
        except KeyboardInterrupt:
            break
    

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
