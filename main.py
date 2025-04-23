import numpy as np
import os
import time
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class MyRobot(PyBulletRobot):
    """My robot"""

    def __init__(self, sim):
        action_dim = 1  # = number of joints; here, 1 joint, so dimension = 1
        action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)

        # Create the URDF file
        urdf_path = "my_robot.urdf"

        super().__init__(
            sim,
            body_name="my_robot",  # choose the name you want
            file_name=urdf_path,  # the path of the URDF file
            base_position=np.array([0.0, 0.0, 0.5]),  # Raised position to be visible
            action_space=action_space,
            joint_indices=np.array([0]),  # list of the indices, as defined in the URDF
            joint_forces=np.array([10.0]),  # Increased force for better control
        )

    def set_action(self, action):
        self.control_joints(target_angles=action)

    def get_obs(self):
        return self.get_joint_angle(joint=0)

    def reset(self):
        neutral_angle = np.array([0.0])
        self.set_joint_angles(angles=neutral_angle)


def main():
    # Initialize PyBullet directly to have more control
    import pybullet as p
    import pybullet_data

    # Initialize PyBullet with appropriate settings
    sim = PyBullet(render_mode="human")
    sim.step()  # Initialize the simulation properly

    # Access the underlying physics client for better visualization
    physics_client = sim.physics_client

    # Configure visualization and add a ground plane using direct PyBullet access
    try:
        # Set camera view
        physics_client.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )

        # Load a plane directly using PyBullet
        physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = physics_client.loadURDF("plane.urdf")
    except Exception as e:
        print(f"Warning: Could not configure visualization: {e}")

    # Create the robot
    robot = MyRobot(sim)
    robot.reset()  # Ensure the robot starts in a defined position

    # Run the simulation loop
    for i in range(10000):
        # Oscillate the joint for demonstration
        action = np.array([np.sin(i * 0.01)])
        robot.set_action(action)
        sim.step()
        time.sleep(0.01)  # Add slight delay for better visualization


if __name__ == "__main__":
    main()