import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CommandSender(Node):
    def __init__(self):
        super().__init__('command_sender')
        self.publisher_ = self.create_publisher(String, 'joint_commands', 10)
        
        self.get_logger().info("Command input node has started. Enter 's' to start stand up, Enter 'r' to return to stand down or 'exit' to quit.")
        self.run()

    def run(self):
        """Handles terminal input and publishes commands."""
        while rclpy.ok():
            command = input("Enter command: ")
            if command.lower() == 'exit':
                break
            if command.strip():  # Ensure input is not empty
                msg = String()
                msg.data = command
                self.publisher_.publish(msg)
                self.get_logger().info(f"Sent command: {command}")

def main(args=None):
    rclpy.init(args=args)
    sender = CommandSender()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
