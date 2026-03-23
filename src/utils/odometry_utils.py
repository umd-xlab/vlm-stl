import math
import numpy as np
from geometry_msgs.msg import Pose

def motion(state, u, dt):
    """
    Vectorized motion model to update the robot's state.
    Args:
        state: Current state [x, y, theta].
        u: Control input [v, omega] (linear and angular velocity).
        dt: Time step duration.
    Returns:
        Updated state [x, y, theta].
    """
    x, y, theta = state
    v, omega = u

    # Update theta (orientation)
    theta_new = theta + omega * dt

    # Update x and y positions based on the new theta and velocities
    x_new = x + v * np.cos(theta_new) * dt
    y_new = y + v * np.sin(theta_new) * dt

    return np.array([x_new, y_new, theta_new])


def euler_from_quaternion(quaternion):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw).

    The quaternion is expected to be a tuple (x, y, z, w).

    Returns:
    A tuple of three elements representing the Euler angles (roll, pitch, yaw) in radians.
    """
    x, y, z, w = quaternion

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw  # in radians

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) into a quaternion.
    
    roll is rotation around x-axis in radians (counterclockwise)
    pitch is rotation around y-axis in radians (counterclockwise)
    yaw is rotation around z-axis in radians (counterclockwise)
    
    Returns:
    A tuple of four elements representing the quaternion (x, y, z, w).
    """
    
    # Compute the quaternion components
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)

def get_min_distance_to_segment(point_cloud, p1, p2):
        
    # vector representation of the trajectory segment from p1 to p2
    vector_traversed = p2 - p1
    # adjust point cloud to be relative to p1
    adjusted_class_points = point_cloud.reshape(-1, 3) - p1
    # use cross product definition of shortest distance from a point to a line
    distances = np.linalg.norm(np.cross(adjusted_class_points, vector_traversed), axis=1) / np.linalg.norm(vector_traversed)
    
    return np.min(distances)

def polar_to_cartesian(distance, angle):
    # Convert polar coordinates to Cartesian coordinates in the robot's frame
    x = distance * math.cos(angle)
    y = distance * math.sin(angle)
    return x, y
    
def transform_to_odom(robot_x, robot_y, robot_th, obstacle_x, obstacle_y):
    # Transform the obstacle position from the robot's frame to the odom frame
    odom_x = robot_x + (obstacle_x * math.cos(robot_th) - obstacle_y * math.sin(robot_th))
    odom_y = robot_y + (obstacle_x * math.sin(robot_th) + obstacle_y * math.cos(robot_th))
    return odom_x, odom_y

def global_to_ego(current_state, traj_state_x, traj_state_y):
    # Transform global coordinates to ego frame (polar coordinates)
    dx = traj_state_x - current_state[0]
    dy = traj_state_y - current_state[1]
    distance = math.sqrt(dx**2 + dy**2)
    angle = current_state[2] - math.atan2(dy, dx)  # Angle relative to robot's heading
    return angle, distance

def odom_traj_to_robot(trajectory, robot_x, robot_y, robot_theta):
    """
    Convert an entire trajectory in the odom frame to the robot frame using vectorized operations.
    `trajectory`: List of states [[x_odom, y_odom, theta_odom], ...]
    """
    # Extract x, y, and theta components from the trajectory
    trajectory = np.array(trajectory)
    x_odom = trajectory[:, 0]
    y_odom = trajectory[:, 1]
    theta_odom = trajectory[:, 2]

    # Vectorized transformation to the robot frame
    x_rob = (x_odom - robot_x) * np.cos(robot_theta) + (y_odom - robot_y) * np.sin(robot_theta)
    y_rob = -(x_odom - robot_x) * np.sin(robot_theta) + (y_odom - robot_y) * np.cos(robot_theta)

    # Combine transformed x and y with theta
    robot_frame_trajectory = np.column_stack((x_rob, y_rob, theta_odom))

    return robot_frame_trajectory