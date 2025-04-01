#!/bin/bash

# Define the udev rule file path
UDEV_RULE_FILE="/etc/udev/rules.d/99-reddog-serial.rules"

# Create or overwrite the udev rule file with the following content
sudo bash -c "cat > $UDEV_RULE_FILE" <<EOL
# Udev rules for RedDog USB Serial Devices
# Right Port
SUBSYSTEM=="tty", ATTRS{idVendor}=="1d6b", ATTRS{idProduct}=="0002", KERNEL=="ttyACM0", SYMLINK+="ttyRedDogLeft", MODE="0666"

# Left Port
SUBSYSTEM=="tty", ATTRS{idVendor}=="2e88", ATTRS{idProduct}=="4603", KERNEL=="ttyACM1", SYMLINK+="ttyRedDogRight", MODE="0666"

# Imu
SUBSYSTEM=="tty", ATTRS{idVendor}=="2639", ATTRS{idProduct}=="0301", KERNEL=="ttyUSB0", SYMLINK+="ttyIMU", MODE="0666"

EOL

# Reload udev rules to apply the new configurations
sudo udevadm control --reload-rules
sudo udevadm trigger

# Inform the user
echo "Udev rules have been successfully updated."
echo "You can now access the devices using /dev/ttyRedDogRight , /dev/ttyRedDogLeft and /dev/ttyIMU"