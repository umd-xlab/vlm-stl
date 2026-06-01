import numpy as np
import cv2
import torch
import time
from PIL import Image as PILImage
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

from utils.gemini_utils import parse_segmentation_masks

import os

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPTokenizer, AutoImageProcessor, AutoModelForDepthEstimation
from transformers import AutoProcessor, GroupViTModel
import open3d as o3d
from google import genai
from google.genai import types

class PerceptionModule:
    def __init__(self, camera_intrinsic_matrix, camera_offset_x, camera_offset_y, 
                 camera_height, camera_tilt_angle, segmentation_model='clipseg', planar_costmap_scale=0.1, publish_outputs=True):
        # Set device for model computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Setting up Perception Module...")
        # Load the CLIPSeg model and processor
        if segmentation_model == 'clipseg':
            self.seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
            self.prompts = ["Road", "Sidewalk", "Stop sign", "Building", "Grass"]  # Prompts for segmentation
        elif segmentation_model == 'groupvit':
            self.seg_model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc", output_segmentation=True).to(self.device)
            self.seg_processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
            self.prompts = ["A photo of a road", "A photo of a sidewalk", "A photo of a stop sign", "A photo of a building", "A photo of grass"]  # Prompts for segmentation
        elif segmentation_model == 'gemini':
            api_key = os.getenv("GOOGLE_API_KEY")
            self.seg_model = genai.Client(api_key=api_key)
            self.safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
            ]
            self.model_id = "gemini-2.5-flash"
            self.classes = ["Road", "Sidewalk", "Stop sign", "Building", "Grass"]
            self.prompt = "Give the segmentation masks for the following objects in the given image: " + ", ".join(self.classes) + ". Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\" as a base64 png string starting with \"data:image/png;base64\", and the text label in the key \"label\". The label should be provided exactly as given in the prompt."
            

        # load depth estimation processor and model
        depth_checkpoint = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_checkpoint)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_checkpoint).to(self.device)
        
        self.prompts = ["A photo of a road", "A photo of a sidewalk", "A photo of a stop sign", "A photo of a building", "A photo of grass"]  # Prompts for segmentation

        # self.prob_thresh = 0.1  # Probability threshold for segmentation masks
        self.cost_values = [1, 5, 20, 20, 10]  # Cost values for each class in the same order as prompts
        self.image_costmap = None
        self.environment_state = None
        self.point_cloud = None
        self.seg_model_name = segmentation_model
        
        self.proj_matrix = camera_intrinsic_matrix  # Projection matrix for camera intrinsics
        self.camera_offset_x = camera_offset_x
        self.camera_offset_y = camera_offset_y
        self.camera_height = camera_height
        self.camera_tilt_angle = camera_tilt_angle
        self.planar_costmap_scale = planar_costmap_scale
        self.publish_outputs = publish_outputs
        
    def process_image(self, image, segmentation_gt=None, depth_gt=None):
        if segmentation_gt is not None:
            self.cost_values = [0] + self.cost_values  # Add cost for background class if using ground truth segmentation
        self.img_h, self.img_w, _ = image.shape
        pil_image = PILImage.fromarray(image)

        # Process the text prompts and image input in batches
        print("Preprocessing Image")
        preprocess_start_time = time.time()
        if self.seg_model_name == 'groupvit':
            seg_inputs = self.seg_processor(text=self.prompts, images=pil_image, padding=True, return_tensors="pt").to(self.device)
        elif self.seg_model_name == 'clipseg':
            seg_inputs = self.seg_processor(text=self.prompts, images=[pil_image] * len(self.prompts), padding=True, return_tensors="pt").to(self.device)
            
        depth_inputs = self.depth_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        print(f"Preprocessing time: {time.time() - preprocess_start_time:.2f} seconds")

        # Perform model inference
        print("Running Inference")
        start_time = time.time()
        if segmentation_gt is None:
            with torch.no_grad():
                if self.seg_model_name == 'groupvit':
                    outputs = self.seg_model(**seg_inputs, interpolate_pos_encoding=True)
                    pred_logits = F.interpolate(outputs.segmentation_logits, size=(self.img_h, self.img_w), mode="bilinear", align_corners=False).squeeze()
                    pred_probs = torch.softmax(pred_logits, dim=0)  # Shape: (num_classes, H', W')
                elif self.seg_model_name == 'clipseg':
                    outputs = self.seg_model(**seg_inputs, interpolate_pos_encoding=True)
                    pred_logits = F.interpolate(outputs.logits.unsqueeze(0), size=(self.img_h, self.img_w), mode="bilinear", align_corners=False).squeeze()
                    pred_probs = torch.softmax(pred_logits, dim=0)  # Shape: (num_classes, H', W')
                elif self.seg_model_name == 'gemini':
                    response = self.seg_model.models.generate_content(
                        model=self.model_id,
                        contents=[self.prompt, pil_image],
                        config = types.GenerateContentConfig(
                            temperature=0.5,
                            safety_settings=self.safety_settings,
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=0
                            )
                        )
                    )
                    
                    segmentation_masks = parse_segmentation_masks(response.text, img_height=pil_image.size[1], img_width=pil_image.size[0])
                    pred_probs = np.zeros((len(self.prompts), self.img_h, self.img_w), dtype=np.float32)
                    for mask in segmentation_masks:
                        class_index = self.classes.index(mask.label)
                        pred_probs[class_index] = np.maximum(pred_probs[class_index], mask.mask / 255.0)  # Normalize mask to [0, 1] and take max if multiple masks for the same class
                    pred_probs = torch.from_numpy(pred_probs).to(self.device)
                    pred_logits = torch.log(pred_probs + 1e-6) 
                
                preds = torch.argmax(pred_probs, dim=0).long() + 1
        
                state_one_hot = F.one_hot(preds, num_classes=len(self.prompts) + 1)  # Shape: (batch_size, num_classes, H, W)
                # Batch process segmentation masks and update the cost map
                preds_resized = pred_probs.cpu().numpy()
        else:
            segmentation_gt = torch.from_numpy(segmentation_gt).long()
            preds = segmentation_gt  # Use ground truth segmentation for cost map generation
            preds_resized = F.one_hot(segmentation_gt, num_classes=len(self.prompts) + 1).permute(2, 0, 1)  # Shape: (batch_size, num_classes, H, W)
            state_one_hot = F.one_hot(segmentation_gt, num_classes=len(self.prompts) + 1)  # Shape: (batch_size, H, W, num_classes)
            pred_logits = preds_resized.float()  # Use one-hot encoded ground truth as "logits" for consistency in cost map generation
        
        if depth_gt is None:
            with torch.no_grad():
                depth_outputs = self.depth_model(depth_inputs)
            output_depth = depth_outputs.predicted_depth
            depth_map = F.interpolate(output_depth.unsqueeze(0), size=(self.img_h, self.img_w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        else:
            depth_map = depth_gt.astype(np.float32)
        
        inference_time = time.time() - start_time
        print(f"Inference Time: {inference_time:.2f} seconds")

        # Get prediction logits and apply sigmoid
        point_cloud_start_time = time.time()
        camera_K = o3d.camera.PinholeCameraIntrinsic(self.img_w, self.img_h, np.array(self.proj_matrix[:3, :3]))
        depth_o3d = o3d.geometry.Image(depth_map)  # Convert to mm and uint16
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, camera_K, project_valid_depth_only=False)
        tilt_rad = np.deg2rad(self.camera_tilt_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
                                    [0, np.sin(tilt_rad), np.cos(tilt_rad)]])
        points = np.asarray(point_cloud.points) @ rotation_matrix.T  # Apply rotation
        points = points.reshape(self.img_h, self.img_w, 3) / self.planar_costmap_scale  # convert to arbitrary scale
        self.point_cloud = points
        print(f"Point cloud generation time: {time.time() - point_cloud_start_time:.2f} seconds")

        cost_map_start_time = time.time()
        # Initialize cost map (set to 128 for non-segmented areas)
        combined_cost_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        # add cost values based on the priority order
        for i, pred_resized in enumerate(preds_resized):
            combined_cost_map = np.max(np.stack([combined_cost_map, pred_resized * self.cost_values[i]]), axis=0)  # Update only segmented regions
            
        combined_cost_map[combined_cost_map == 0] = 128  # Set non-segmented areas to 128
        print(f"Cost map generation time: {time.time() - cost_map_start_time:.2f} seconds")

        self.environment_state = state_one_hot.cpu().numpy()  # Store the one-hot encoded environment state for later use
                
        # Clip and convert cost map to 8-bit for visualization
        self.image_costmap = combined_cost_map
        
        self.get_top_down_environment_state()  # Generate top-down environment state map for visualization
        
        return pred_logits.cpu().numpy()
        
    def get_top_down_environment_state(self):
        
        # Get the class with the highest probability for each pixel
        class_indices = np.argmax(self.environment_state, axis=2)  # Shape: (H, W)
        
        points_with_class = np.concatenate((self.point_cloud, class_indices[..., np.newaxis]), axis=-1)
        points_with_class = points_with_class.reshape(-1, 4)  # Reshape to (num_points, 4) where columns are (X, Y, Z, Class)
        points_with_class = points_with_class[~np.isnan(points_with_class).any(axis=1)]  # Filter out points with NaN values
        points_with_class = points_with_class[np.flip(points_with_class[:, 1].argsort(), axis=0)]  # reversed Sort by y-coordinate
        
        width = int(np.nanmax(points_with_class[:, 0]) - np.nanmin(points_with_class[:, 0])) + 1
        height = int(np.nanmax(points_with_class[:, 2]) - np.nanmin(points_with_class[:, 2])) + 1
        ground_plane_state_map = np.zeros((height, width), dtype=np.int32) - 1  # Initialize with -1 for unknown areas
        ground_plane_state_map[(points_with_class[:, 2] - np.nanmin(points_with_class[:, 2])).astype(int), 
                              (points_with_class[:, 0] - np.nanmin(points_with_class[:, 0])).astype(int)] = points_with_class[:, 3].astype(int)  # Assign class values to the ground plane state map
        
        data_points = np.where(ground_plane_state_map >= 0)
        values = ground_plane_state_map[data_points]
        fill_points = np.where(ground_plane_state_map < 0)
        interpolated_values = griddata(data_points, values, fill_points, method='nearest', fill_value=-1)  # Fill in missing values with nearest neighbor interpolation
        ground_plane_state_map[fill_points] = interpolated_values.astype(int)
        
        ground_plane_state_map = np.flip(ground_plane_state_map, axis=0)  # Flip vertically for correct orientation
        
        ground_plane_state_map = F.one_hot(torch.from_numpy(ground_plane_state_map.copy()).long(), num_classes=len(self.prompts) + 1).permute(2, 0, 1).numpy()  # Convert back to one-hot encoding for consistency
        
        class_colors = [(0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0)]
        segmentation_color_map = ListedColormap(np.array(class_colors) / 255.0)
        
        plt.imshow(np.argmax(ground_plane_state_map, axis=0), cmap=segmentation_color_map, vmin=0, vmax=len(self.prompts) + 1)  # Visualize the top-down environment state map
        plt.title("Top-Down Environment State Map")
        plt.xlabel("X (1/10 meters)")
        plt.ylabel("Z (1/10 meters)")
        plt.show()
        return ground_plane_state_map
    
    def get_top_down_costmap(self):
        points_with_cost = np.concatenate((self.point_cloud, self.image_costmap[..., np.newaxis]), axis=-1)
        points_with_cost = points_with_cost.reshape(-1, 4)  # Reshape to (num_points, 4) where columns are (X, Y, Z, Cost)
        points_with_cost = points_with_cost[~np.isnan(points_with_cost).any(axis=1)]  # Filter out points with NaN values
        points_with_cost = points_with_cost[np.flip(points_with_cost[:, 1].argsort(), axis=0)]  # reversed Sort by y-coordinate
        
        width = int(np.nanmax(points_with_cost[:, 0]) - np.nanmin(points_with_cost[:, 0])) + 1
        height = int(np.nanmax(points_with_cost[:, 2]) - np.nanmin(points_with_cost[:, 2])) + 1
        ground_plane_cost_map = np.zeros((height, width), dtype=np.float32)
        ground_plane_cost_map[(points_with_cost[:, 2] - np.nanmin(points_with_cost[:, 2])).astype(int), 
                              (points_with_cost[:, 0] - np.nanmin(points_with_cost[:, 0])).astype(int)] = points_with_cost[:, 3]  # Assign cost values to the ground plane cost map
                
        data_points = np.where(ground_plane_cost_map > 0)
        values = ground_plane_cost_map[data_points]
        fill_points = np.where(ground_plane_cost_map == 0)
        interpolated_values = griddata(data_points, values, fill_points, method='linear', fill_value=0)  # Fill in missing values with nearest neighbor interpolation
        ground_plane_cost_map[fill_points] = interpolated_values
        
        ground_plane_cost_map = np.flip(ground_plane_cost_map, axis=0)  # Flip vertically for correct orientation
        
        return ground_plane_cost_map

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
        marked_img[marked_img == 128] = 0  # Set non-segmented areas to black for better visualization
        
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
    class_colors = [(255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0)]
    segmentation_color_map = ListedColormap(np.array(class_colors) / 255.0)
    
    setup_start_time = time.time()
    true_perception_module = PerceptionModule(intrinsic_matrix, offset_x, offset_y, height, tilt_angle)
    print(f"Perception Module setup time: {time.time() - setup_start_time:.2f} seconds")
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL

    processing_start_time = time.time()
    true_perception_module.process_image(image, segmentation_gt=segmentation_gt, depth_gt=depth_gt)
    print(f"Image processing time: {time.time() - processing_start_time:.2f} seconds")
    
    # sample_trajectory = [[5, 0], [10, -2], [15, 0], [20, 0], [25, 0]] 
    sample_trajectory = [[5, 0], [10, -2], [12, -4], [13, -4], [16, -3]] 
    
    cost_start_time = time.time()
    marked_img, max_cost, total_cost, points = true_perception_module.get_traj_behav_cost(sample_trajectory, full_trajectory=True)
    print(f"Trajectory cost calculation time: {time.time() - cost_start_time:.2f} seconds")
    
    print(f"Max True Cost along trajectory: {max_cost}")
    print(f"Total True Cost along trajectory: {total_cost}")
    
    os.makedirs("output_images", exist_ok=True)
    drawn_marked_img = (marked_img / np.max(marked_img) * 255).astype(np.uint8)  # Normalize for better visualization
    marked_color_img = cv2.applyColorMap(drawn_marked_img, cv2.COLORMAP_JET)
    cv2.polylines(marked_color_img, [points], isClosed=False, color=(255, 0, 255), thickness=8)
    
    gt_path_mask = cv2.polylines(np.zeros_like(marked_img), [points], isClosed=False, color=(255), thickness=10)
    gt_path_costs = marked_img[gt_path_mask > 0]
    
    combined_img = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, marked_color_img, 0.5, 0)
    cv2.imwrite("output_images/gt_marked_image.png", combined_img)
    
    pred_perception_module = PerceptionModule(intrinsic_matrix, offset_x, offset_y, height, tilt_angle, segmentation_model='clipseg')
    pred_logits = pred_perception_module.process_image(image)
    
    marked_img_pred, max_cost_pred, total_cost_pred, points_pred = pred_perception_module.get_traj_behav_cost(sample_trajectory, full_trajectory=True)
    print(f"Predicted Max Cost along trajectory: {max_cost_pred}")
    print(f"Predicted Total Cost along trajectory: {total_cost_pred}")
    
    drawn_marked_img_pred = (marked_img_pred / np.max(marked_img_pred) * 255).astype(np.uint8)  # Normalize for better visualization
    marked_color_img_pred = cv2.applyColorMap(drawn_marked_img_pred, cv2.COLORMAP_JET)
    cv2.polylines(marked_color_img_pred, [points_pred], isClosed=False, color=(255, 0, 255), thickness=8)
    
    pred_path_mask = cv2.polylines(np.zeros_like(marked_img_pred), [points_pred], isClosed=False, color=(255), thickness=10)
    pred_path_costs = marked_img_pred[pred_path_mask > 0]
    
    combined_img_pred = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, marked_color_img_pred, 0.5, 0)
    cv2.imwrite("output_images/pred_marked_image.png", combined_img_pred)
    
    cost_error = np.abs(gt_path_costs - pred_path_costs)
    print(gt_path_costs.dtype, pred_path_costs.dtype, cost_error.dtype)
    print(min(gt_path_costs - pred_path_costs), max(gt_path_costs - pred_path_costs))
    print(max(cost_error))
    print(max(gt_path_costs), max(pred_path_costs))
    print(min(gt_path_costs), min(pred_path_costs))
    print(f"Total cost error along trajectory: {np.sum(cost_error)}")
    print(f"Average cost error along trajectory: {np.mean(cost_error)}")
    print(f"Max cost error along trajectory: {np.max(cost_error)}")
    
    error_map = np.zeros_like(marked_img)
    error_map[pred_path_mask > 0] = cost_error
    drawn_error_map = (error_map / np.max(error_map) * 255).astype(np.uint8)  # Normalize for better visualization
    error_color_map = cv2.applyColorMap(drawn_error_map, cv2.COLORMAP_INFERNO)
    combined_error_img = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, error_color_map, 0.5, 0)
    cv2.imwrite("output_images/cost_error_map.png", combined_error_img)
    
    pred_logits = np.concatenate((np.zeros((1, pred_logits.shape[1], pred_logits.shape[2]), dtype=np.float32), pred_logits), axis=0)  # Add background class with zero probability

    loss_img = F.cross_entropy(torch.from_numpy(pred_logits).unsqueeze(0), torch.from_numpy(segmentation_gt).long().unsqueeze(0), reduction='none').squeeze()
    drawn_loss_img = (loss_img.squeeze().cpu().numpy() / np.max(loss_img.cpu().numpy()) * 255).astype(np.uint8)  # Normalize for better visualization
    loss_color_map = cv2.applyColorMap(drawn_loss_img, cv2.COLORMAP_JET)
    combined_loss_img = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, loss_color_map, 0.5, 0)
    cv2.imwrite("output_images/loss_map.png", combined_loss_img)
    
    traj_loss = np.zeros_like(marked_img, dtype=np.float32)
    traj_loss[pred_path_mask > 0] = loss_img.squeeze().cpu().numpy()[pred_path_mask > 0]
    drawn_traj_loss = (traj_loss / np.max(traj_loss) * 255).astype(np.uint8)  # Normalize for better visualization
    traj_loss_color_map = cv2.applyColorMap(drawn_traj_loss, cv2.COLORMAP_INFERNO)
    traj_loss_img = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, traj_loss_color_map, 0.5, 0)
    cv2.imwrite("output_images/traj_loss_map.png", traj_loss_img)
    
    print(f"Average loss across image: {torch.mean(loss_img).item()}")
    print(f"Average loss along trajectory: {np.mean(loss_img.squeeze().cpu().numpy()[pred_path_mask > 0])}")
    print(f"Max loss along trajectory: {np.max(loss_img.squeeze().cpu().numpy()[pred_path_mask > 0])}")
    print(f"Total loss along trajectory: {np.sum(loss_img.squeeze().cpu().numpy()[pred_path_mask > 0])}")
    
    pred_env_state = pred_perception_module.environment_state
    pred_segmentation = np.argmax(pred_env_state, axis=2)
    colored_pred_segmentation = segmentation_color_map(pred_segmentation / (len(true_perception_module.prompts) + 1))[:, :, :3]  # Normalize for colormap and convert to RGB
    colored_pred_segmentation = (colored_pred_segmentation * 255).astype(np.uint8)
    combined_segmentation_img = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, colored_pred_segmentation, 0.5, 0)
    cv2.imwrite("output_images/pred_segmentation.png", combined_segmentation_img)
    
    colored_gt_segmentation = segmentation_color_map(segmentation_gt / (len(true_perception_module.prompts) + 1))[:, :, :3]  # Normalize for colormap and convert to RGB
    colored_gt_segmentation = (colored_gt_segmentation * 255).astype(np.uint8)
    combined_gt_segmentation_img = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, colored_gt_segmentation, 0.5, 0)
    cv2.imwrite("output_images/gt_segmentation.png", combined_gt_segmentation_img)
    
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
    
    camera_height = 0.559221 + 0.503693 # for the blender image test
    
    proj_matrix = intrinsic_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    single_image_test(proj_matrix, 0, 0, extrinsic_matrix[0, 3], -euler_angles[0] + 90, args.image_path, segmentation, depth)
    
    print(f"Total execution time: {time.time() - program_start_time:.2f} seconds")