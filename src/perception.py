import numpy as np
import cv2
import torch

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPTokenizer, AutoImageProcessor, AutoModelForDepthEstimation
import open3d as o3d

class PerceptionModule:
    def __init__(self, camera_projection_matrix, camera_offset_x, camera_offset_y, 
                 camera_height, camera_tilt_angle, img_h, img_w):
        # Set device for model computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the CLIPSeg model and processor
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # load depth estimation processor and model
        depth_checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_checkpoint)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_checkpoint).to(self.device)
        
        self.prompts = ["vegetation", "Pavement","grass", "Stop gesture"]  # Prompts for segmentation

        self.prob_thresh = 0.1  # Probability threshold for segmentation masks
        self.cost_values = [0.05,0.95,0.48,0] #high means preferred regions (e.g., pavement), and low means avoiding regions
        self.behav_costmap = None
        self.environment_state = None
        self.point_cloud = None
        
        self.proj_matrix = camera_projection_matrix
        self.camera_offset_x = camera_offset_x
        self.camera_offset_y = camera_offset_y
        self.camera_height = camera_height
        self.camera_tilt_angle = camera_tilt_angle
        self.img_h = img_h
        self.img_w = img_w

    # This function calculates the distance of the closest point from each class 
    # to the segment of the trajectory defined by points p1 and p2
    # do this for all classes at once to benefit more from vectorization
    def get_min_distance_to_classes(self, p1, p2, range_threshold):
        
        # vector representation of the trajectory segment from p1 to p2
        vector_traversed = p2 - p1
        # adjust point cloud to be relative to p1
        adjusted_class_points = self.point_cloud.reshape(-1, 3) - p1
        # use cross product definition of shortest distance from a point to a line
        distances = np.linalg.norm(np.cross(adjusted_class_points, vector_traversed), axis=1) / np.linalg.norm(vector_traversed)
        # alternative distance calculation; may be faster
        # rejection_vectors = adjusted_class_points - np.dot(adjusted_class_points, vector_traversed)[:, np.newaxis] / np.dot(vector_traversed, vector_traversed) * vector_traversed
        # distances = np.linalg.norm(rejection_vectors, axis=1) / np.linalg.norm(vector_traversed)
        
        min_distances = []
        safest_points = []
        for class_id, cost_value in enumerate(self.cost_values):
            class_distances = distances[self.environment_state[class_id].flatten()]
            if class_distances.size > 0:
                class_min_dist = np.min(class_distances)
                
                # if too close to an obstacle of this class, calculate the safest point to steer towards 
                if class_min_dist < range_threshold:
                    closest_point = adjusted_class_points[np.argmin(class_distances)]
                    # will be removed if rejection vector method is faster
                    rejection_vector = closest_point - np.dot(closest_point, vector_traversed) / np.dot(vector_traversed, vector_traversed) * vector_traversed
                    
                    safe_point = closest_point - rejection_vector * (range_threshold / np.linalg.norm(rejection_vector))
                    safest_points.append(safe_point + p1)  # Transform back to global coordinates
                else:
                    safest_points.append(None)
                
                min_distances.append(class_min_dist)
            else:
                min_distances.append(float('inf'))
        
        return min_distances, safest_points

    # TODO: Modify to include information about trajectory compliance
    def get_traj_behav_cost(self, robot_frame_trajectory):
        """
        Project robot frame trajectory points onto the camera image plane, get pixel coordinates,
        and return the maximum cost from the costmap at the trajectory pixel locations.
        """

        # Create a copy of the behavior cost map for visualization purposes
        marked_img = self.behav_costmap.copy()

        # Convert robot_frame_trajectory to NumPy array for efficient vectorized operations
        robot_frame_trajectory = np.array(robot_frame_trajectory)

        # Extract x_rob and y_rob from the trajectory
        x_rob = robot_frame_trajectory[:, 0]
        y_rob = robot_frame_trajectory[:, 1]

        # Vectorized computation for camera frame coordinates
        traj_coords_xyz = np.column_stack((
            -y_rob + self.camera_offset_y,   # Y-axis adjustment
            np.full(x_rob.shape, self.camera_height),  # Constant camera height
            x_rob - self.camera_offset_x  # X-axis adjustment
        ))

        if traj_coords_xyz.shape[0] == 0:
            return marked_img, 0.0

        # Apply rotation for tilt
        alpha = np.deg2rad(self.camera_tilt_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(alpha), -np.sin(alpha)],
                                    [0, np.sin(alpha), np.cos(alpha)]])
        points_rotated = np.dot(traj_coords_xyz, rotation_matrix.T)

        # Homogeneous coordinates for projection
        points_homogeneous = np.hstack((points_rotated, np.ones((points_rotated.shape[0], 1))))

        # Project onto image plane using the projection matrix
        uvw = np.dot(self.proj_matrix, points_homogeneous.T)
        u_vec, v_vec, w_vec = uvw[0], uvw[1], uvw[2]

        # Compute pixel coordinates
        x_vec = (u_vec / w_vec).astype(int)
        y_vec = (v_vec / w_vec).astype(int)

        # Filter out points that are outside the image bounds
        valid_indices = (y_vec >= 0) & (y_vec < self.img_h) & (x_vec >= 0) & (x_vec < self.img_w)
        valid_x = x_vec[valid_indices]
        valid_y = y_vec[valid_indices]

        if len(valid_x) == 0:
            return marked_img, 0.0

        # Extract cost values from the costmap at the valid pixel coordinates
        costs = self.behav_costmap[valid_y, valid_x]
        max_cost = np.max(costs) if costs.size > 0 else 0.0

        if self.publish_outputs:
            # Visualize the trajectory points on the image
            points = np.vstack((valid_x, valid_y)).T
            if len(points) > 1:
                cv2.polylines(marked_img, [points], isClosed=False, color=255, thickness=8)

            # Publish the marked image
            marked_image_msg = self.bridge.cv2_to_imgmsg(marked_img, encoding="mono8")
            self.traj_image_pub.publish(marked_image_msg)

        return marked_img, max_cost