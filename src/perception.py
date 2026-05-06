import numpy as np
import cv2
import torch
import time
from PIL import Image as PILImage
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import os

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPTokenizer, AutoImageProcessor, AutoModelForDepthEstimation
import open3d as o3d

class PerceptionModule:
    def __init__(self, camera_intrinsic_matrix, camera_offset_x, camera_offset_y, 
                 camera_height, camera_tilt_angle, publish_outputs=True):
        # Set device for model computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Setting up Perception Module...")
        # Load the CLIPSeg model and processor
        self.seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # load depth estimation processor and model
        depth_checkpoint = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_checkpoint)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_checkpoint).to(self.device)
        
        self.prompts = ["Road", "Sidewalk", "Stop Sign", "Building", "Grass"]  # Prompts for segmentation

        # self.prob_thresh = 0.1  # Probability threshold for segmentation masks
        self.cost_values = [0, 1, 5, 20, 20, 10]  # Cost values for each class in the same order as prompts
        self.image_costmap = None
        self.environment_state = None
        self.point_cloud = None
        
        self.proj_matrix = camera_intrinsic_matrix  # Projection matrix for camera intrinsics
        self.camera_offset_x = camera_offset_x
        self.camera_offset_y = camera_offset_y
        self.camera_height = camera_height
        self.camera_tilt_angle = camera_tilt_angle
        
        self.publish_outputs = publish_outputs
        
    def process_image(self, image, segmentation_gt=None, depth_gt=None):
        self.img_h, self.img_w, _ = image.shape
        pil_image = PILImage.fromarray(image)

        # Process the text prompts and image input in batches
        print("Preprocessing Image")
        preprocess_start_time = time.time()
        seg_inputs = self.seg_processor(self.prompts, images=[pil_image] * len(self.prompts), padding=True, return_tensors="pt").to(self.device)
        depth_inputs = self.depth_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        print(f"Preprocessing time: {time.time() - preprocess_start_time:.2f} seconds")

        # Perform model inference
        print("Running Inference")
        start_time = time.time()
        if segmentation_gt is None:
            with torch.no_grad():
                outputs = self.seg_model(**seg_inputs, interpolate_pos_encoding=True)
                multilabel_preds = torch.sigmoid(outputs.logits)
                softmax_preds = torch.softmax(outputs.logits, dim=0)
                print(F.one_hot(torch.argmax(softmax_preds, dim=0), num_classes=len(self.prompts)).shape)
                state_one_hot = F.one_hot(torch.argmax(softmax_preds, dim=0), num_classes=len(self.prompts))  # Shape: (batch_size, num_classes, H, W)
                # Batch process segmentation masks and update the cost map
                preds_resized = F.interpolate(multilabel_preds.unsqueeze(1), size=(self.img_h, self.img_w), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
        else:
            segmentation_gt = torch.from_numpy(segmentation_gt).long()
            preds_resized = F.one_hot(segmentation_gt, num_classes=len(self.prompts) + 1).permute(2, 0, 1)  # Shape: (batch_size, num_classes, H, W)
            state_one_hot = F.one_hot(segmentation_gt, num_classes=len(self.prompts) + 1).permute(2, 0, 1)  # Shape: (batch_size, num_classes, H, W)
        
        if depth_gt is None:
            with torch.no_grad():
                depth_outputs = self.depth_model(depth_inputs)
            output_depth = depth_outputs.predicted_depth
            depth_map = F.interpolate(output_depth.unsqueeze(0), size=(self.img_h, self.img_w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        else:
            depth_map = depth_gt.astype(np.float32)
        
        inference_time = time.time() - start_time
        print(f"Inference Time: {inference_time:.2f} seconds")
        
        plt.imshow(depth_map, cmap='inferno')
        plt.colorbar()
        plt.show()
        plt.close()

        # Get prediction logits and apply sigmoid
        point_cloud_start_time = time.time()
        camera_K = o3d.camera.PinholeCameraIntrinsic(self.img_w, self.img_h, np.array(self.proj_matrix[:3, :3]))
        depth_o3d = o3d.geometry.Image(depth_map)  # Convert to mm and uint16
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, camera_K, project_valid_depth_only=False)
        points = np.asarray(point_cloud.points).reshape(self.img_h, self.img_w, 3)
        self.point_cloud = points
        print(f"Point cloud generation time: {time.time() - point_cloud_start_time:.2f} seconds")

        cost_map_start_time = time.time()
        # Initialize cost map (set to 128 for non-segmented areas)
        combined_cost_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        # add cost values based on the priority order
        for i, pred_resized in enumerate(preds_resized):
            combined_cost_map = np.max(np.stack([combined_cost_map, pred_resized * self.cost_values[i]]), axis=0)  # Update only segmented regions
            
        combined_cost_map[combined_cost_map == 0] = 1  # Set non-segmented areas to 128
        print(f"Cost map generation time: {time.time() - cost_map_start_time:.2f} seconds")
        
        
        self.environment_state = state_one_hot.cpu().numpy()  # Store the one-hot encoded environment state for later use
        
        # Clip and convert cost map to 8-bit for visualization
        combined_cost_map = np.clip(combined_cost_map, 0, 255).astype(np.uint8)
        self.image_costmap = combined_cost_map
        

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
    def get_traj_behav_cost(self, robot_frame_trajectory, full_trajectory=False):
        """
        Project robot frame trajectory points onto the camera image plane, get pixel coordinates,
        and return the maximum cost from the costmap at the trajectory pixel locations.
        """

        # Create a copy of the behavior cost map for visualization purposes
        marked_img = self.image_costmap.copy()
        
        traj_lengths = np.linalg.norm(np.diff(robot_frame_trajectory, axis=0), axis=1)
        total_traj_length = np.sum(traj_lengths)

        # Convert robot_frame_trajectory to NumPy array for efficient vectorized operations
        robot_frame_trajectory = np.array(robot_frame_trajectory)

        # Extract x_rob and y_rob from the trajectory
        x_rob = robot_frame_trajectory[:, 0]
        y_rob = robot_frame_trajectory[:, 1]

        # Vectorized computation for camera frame coordinates
        traj_coords_xyz = np.column_stack((
            -y_rob + self.camera_offset_y,   # Y-axis adjustment
            np.full(x_rob.shape, -self.camera_height),  # Constant camera height
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
        valid_y = self.img_h - 1 - y_vec[valid_indices]  # Flip y-axis for image coordinates
        
        link_costs = []
        for i in range(len(valid_x) - 1):
            mask_img = np.zeros_like(marked_img)
            cv2.line(mask_img, (valid_x[i], valid_y[i]), (valid_x[i+1], valid_y[i+1]), color=255, thickness=1)
        
            traj_costs = self.image_costmap[mask_img > 0]
            link_cost = np.sum(traj_costs) / np.sum(mask_img > 0) if np.sum(mask_img > 0) > 0 else 0
            world_link_cost = link_cost * traj_lengths[i]  # Scale by the length of the trajectory segment
            link_costs.append(world_link_cost)

        if len(valid_x) == 0:
            return marked_img, 0.0, 0, []

        # Extract cost values from the costmap at the valid pixel coordinates
        max_cost = np.max(link_costs)
        total_cost = np.sum(link_costs)  # Average cost along the trajectory

        if self.publish_outputs and full_trajectory:
            # Visualize the trajectory points on the image
            points = np.vstack((valid_x, valid_y)).T
            return marked_img, max_cost, total_cost, points
            # if len(points) > 1:
            #     cv2.polylines(marked_img, [points], isClosed=False, color=0, thickness=8)

            # Publish the marked image
            # marked_image_msg = self.bridge.cv2_to_imgmsg(marked_img, encoding="mono8")
            # self.traj_image_pub.publish(marked_image_msg)

        return marked_img, max_cost, total_cost, []
    
def single_image_test(intrinsic_matrix, offset_x, offset_y, height, tilt_angle, image_path, segmentation_gt, depth_gt):
    setup_start_time = time.time()
    perception_module = PerceptionModule(intrinsic_matrix, offset_x, offset_y, height, tilt_angle)
    print(f"Perception Module setup time: {time.time() - setup_start_time:.2f} seconds")
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL

    processing_start_time = time.time()
    perception_module.process_image(image, segmentation_gt=segmentation_gt, depth_gt=depth_gt)
    print(f"Image processing time: {time.time() - processing_start_time:.2f} seconds")
    
    sample_trajectory = [[5, 0], [10, -2], [15, 0], [20, 0], [25, 0]] 
    
    cost_start_time = time.time()
    marked_img, max_cost, total_cost, points = perception_module.get_traj_behav_cost(sample_trajectory, full_trajectory=True)
    print(f"Trajectory cost calculation time: {time.time() - cost_start_time:.2f} seconds")
    
    print(f"Max Cost along trajectory: {max_cost}")
    print(f"Total Cost along trajectory: {total_cost}")
    
    os.makedirs("output_images", exist_ok=True)
    marked_img = (marked_img / np.max(marked_img) * 255).astype(np.uint8)  # Normalize for better visualization
    marked_color_img = cv2.applyColorMap(marked_img, cv2.COLORMAP_JET)
    cv2.polylines(marked_color_img, [points], isClosed=False, color=(255, 0, 255), thickness=8)
    
    combined_img = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, marked_color_img, 0.5, 0)
    cv2.imwrite("output_images/marked_image.png", combined_img)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    points = perception_module.point_cloud.reshape(-1, 3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, c=points[:, 2], cmap='inferno')
    plt.show()
    plt.close()
    
    
    # Sample trajectory points in robot frame
    
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Test the PerceptionModule with a single image and trajectory.")
    parser.add_argument("--intrinsic_matrix", type=str, required=True, help="Path to the camera intrinsic matrix (json file).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the .npz file containing ground truth segmentation and depth maps.")
    
    args = parser.parse_args()
    
    program_start_time = time.time()
    
    print("Loading camera intrinsic parameters...")
    
    with open(args.intrinsic_matrix, 'r') as f:
        json_dict = json.load(f)
        
    intrinsic_params = json_dict["cam_mat_intr"]
    intrinsic_matrix = np.eye(3)
    intrinsic_matrix[0, 0] = intrinsic_params["f_x"]
    intrinsic_matrix[1, 1] = intrinsic_params["f_y"]
    intrinsic_matrix[0, 2] = intrinsic_params["c_x"]
    intrinsic_matrix[1, 2] = intrinsic_params["c_y"]
    
    print("Loading ground truth segmentation and depth maps...")

    gt_maps = np.load(args.gt_path)
    segmentation = gt_maps['segmentation_masks']
    depth = gt_maps['depth_map']
    extrinsic_matrix = gt_maps['extrinsic_mat']
    
    rot_mat = Rotation.from_matrix(extrinsic_matrix[:3, :3])
    euler_angles = rot_mat.as_euler('xyz', degrees=True)
    
    camera_height = 0.559221 + 0.503693 
    
    rot_mat = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    
    proj_matrix = intrinsic_matrix @ np.hstack((rot_mat, np.zeros((3, 1))))
    
    single_image_test(proj_matrix, 0, 0, extrinsic_matrix[0, 3], -euler_angles[0] + 90, args.image_path, segmentation, depth)
    
    print(f"Total execution time: {time.time() - program_start_time:.2f} seconds")