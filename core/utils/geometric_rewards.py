"""
Geometric reward calculation based on vanishing point detection.
Adapted from GlobustVP for DDPO-PyTorch integration.
"""
import numpy as np
import cv2
import torch
from typing import Optional

from core.utils.line_processing import detect_and_format_lines
from core.utils.geometry import (
    normalize_lines,
    compute_backprojection_normals,
    compute_line_uncertainties
)

def get_intersection_vp(line1_normal: np.ndarray, line2_normal: np.ndarray) -> Optional[np.ndarray]:
    """
    Geometric Reward Function Part 1: Finding Line Intersections.
    
    In this codebase, lines are represented as 3D normals of the plane passing through
    the camera center and the line segment.
    
    The intersection of two such planes defines the direction of the Vanishing Point (VP).
    Geometrically, this is simply the CROSS PRODUCT of the two line normals.
    
    Parameters:
        line1_normal: Shape (3,) unit vector representing the first line.
        line2_normal: Shape (3,) unit vector representing the second line.
        
    Returns:
        candidate_vp: Shape (3,) unit vector representing the intersection direction.
                      Returns None if lines are identical/parallel (cross prod ~ 0).
    """
    # 1. Compute cross product
    vp_direction = np.cross(line1_normal, line2_normal)
    
    # 2. Check for degeneracies (lines are parallel in 3D or identical)
    norm = np.linalg.norm(vp_direction)
    if norm < 1e-8:
        return None
        
    # 3. Normalize to get a valid direction vector
    return vp_direction / norm


def calculate_line_support_reward(
    candidate_vp: np.ndarray, 
    all_lines: np.ndarray, 
    threshold_c: float = 0.03
) -> int:
    """
    Geometric Reward Function Part 2: Max parallel line grouping.
    
    Calculates a reward for a candidate VP based on how many lines in the scene
    are parallel to it (form a group with it).
    
    Parameters:
        candidate_vp: Shape (3,) Unit vector of the VP hypothesis.
        all_lines: Shape (N, 3) Normals of all detected lines in the image.
        threshold_c: Threshold for consistency (default 0.03).
        
    Returns:
        reward: Integer count of lines supporting this VP.
    """
    if candidate_vp is None:
        return 0
        
    # 1. Compute dot product (cosine of angle from 90 deg)
    # Ideally, line_normal is perpendicular to VP, so dot product should be 0.
    consistency = np.abs(np.dot(all_lines, candidate_vp))
    
    # 2. Find lines that satisfy the geometric constraint
    inliers = np.sum(consistency < threshold_c)
    
    return int(inliers)


class ImageGeometricReward:
    """
    Reward calculator for geometric structure based on vanishing point detection.
    """
    
    def __init__(self, K: Optional[np.ndarray] = None, min_length: float = 20.0):
        """
        Initialize the reward calculator.
        
        Args:
            K: Camera intrinsic matrix (3x3). If None, will be estimated from image size.
            min_length: Minimum line segment length for detection.
        """
        self.K_provided = K
        self.K = K
        self.min_length = min_length
            
    def _process_image(self, image: np.ndarray):
        """
        Helper: Line detection and normalization pipeline.
        
        Returns:
            lines_2D: (6, N) detected lines
            normalized_lines: (N, 4) normalized line endpoints
            para_lines: (N, 3) 3D normals
            uncertainty: (N, 1) uncertainty weights
            line_labels: (N,) cluster indices (0..k-1)
        """
        # Ensure grayscale
        if image.ndim == 3:
            # Assume BGR (default for cv2) or RGB
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image[:, :, 0]
            h, w = image.shape[:2]
        else:
            gray = image
            h, w = image.shape

        # Update K if not provided
        if self.K_provided is None:
            # Estimate: focal length roughly 1.0 * max(w, h)
            # Center at image center
            f = max(w, h)
            self.K = np.array([
                [f, 0, w/2],
                [0, f, h/2],
                [0, 0, 1]
            ])
            
        # --- Step 1: Detect Lines (LSD + Grouping) ---
        try:
            # Now returns labels as well
            lines_2D, line_labels = detect_and_format_lines(gray, min_length=self.min_length, angle_cluster_k=3)
        except ValueError:
            return None, None, None, None, None
            
        # --- Step 2: Normalize Lines ---
        normalized_lines = normalize_lines(self.K, lines_2D)
        
        # --- Step 3: Back-project to Normals ---
        # Note: input expects (N, 4), which is normalized_lines.T
        para_lines = compute_backprojection_normals(normalized_lines.T)
        
        # --- Step 4: Compute Uncertainty ---
        uncertainty = compute_line_uncertainties(normalized_lines.T, self.K, use_uncertainty=False)
        
        return lines_2D, normalized_lines.T, para_lines, uncertainty, line_labels

    def get_intersection_sampling_reward(
        self, 
        image: np.ndarray, 
        num_samples: int = 50,
        threshold_c: float = 0.03,
        min_support: int = 3
    ) -> float:
        """
        Calculates BINARY reward based on finding vanishing points via random sampling.
        Legacy method.
        """
        _, _, para_lines, _, _ = self._process_image(image)
        if para_lines is None or len(para_lines) < 2:
            return 0.0
            
        best_score = 0
        N = len(para_lines)
        
        for _ in range(num_samples):
            # 1. Random Sample 2 lines
            idx1, idx2 = np.random.choice(N, 2, replace=False)
            line1, line2 = para_lines[idx1], para_lines[idx2]
            
            # 2. Find Intersection VP (Geometry)
            vp = get_intersection_vp(line1, line2)
            if vp is None: 
                continue
            
            # 3. Calculate Support (Reward)
            score = calculate_line_support_reward(vp, para_lines, threshold_c=threshold_c)
            
            if score > best_score:
                best_score = score
        
        return 1.0 if best_score >= min_support else 0.0

    def get_max_consensus_reward(
        self, 
        image: np.ndarray, 
        solver_timeout: float = 3.0,
        threshold_c: float = 0.03,
        max_iters: int = 100
    ) -> float:
        """
        Calculates BINARY reward using the full SDP solver (synchronous).
        """
        if globustvp is None:
            return 0.0
            
        _, norm_lines, para_lines, uncertainty, _ = self._process_image(image)
        if para_lines is None: 
            return 0.0
        
        N = para_lines.shape[0]
        if N < 3: 
            return 0.0
        
        param = {
            "line_num": N,
            "vanishing_point_num": 1,
            "c": threshold_c,
            "sample_line_num": min(8, N),
            "is_fast_solver": True,
            "eigen_threshold": 1,
            "solver": "SCS",
            "solver_opts": {"eps_abs": 1e-3, "eps_rel": 1e-3, "max_iters": max_iters}
        }
        
        try:
            status, est_vps, est_corrs = globustvp(norm_lines, para_lines, uncertainty, param)
            if status and len(est_corrs) > 0:
                return 1.0
            else:
                return 0.0
        except Exception:
            return 0.0

    def get_algebraic_intersection_reward(
        self,
        image: np.ndarray,
        num_samples: int = 5,
        threshold_c: float = 0.03,
        epsilon: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> float:
        """
        Calculates a continuous reward based on the algebraic intersection quality of lines,
        PER CLUSTER (Group).

        Algorithm:
        1. Detect Lines (LSD) & Partition into K groups (Hough/Angle Voting).
        2. For EACH Group k:
           - Move lines to GPU.
           - Sample pairs within the group.
           - Find intersection VP_k.
           - Compute mean algebraic error of lines IN THE SAME GROUP to VP_k.
           - Compute Group Reward = 1 / (1 + error).
        3. Final Reward = Average(Group Rewards).

        This robustly handles multiple VPs (Left, Right, Vertical) by rewarding each structure independently.
        """
        # 1. Detect and Group Lines
        _, _, para_lines_cpu, _, labels_cpu = self._process_image(image)
        
        if para_lines_cpu is None or len(para_lines_cpu) < 2:
            return 0.0
            
        # Convert to tensors
        all_lines = torch.from_numpy(para_lines_cpu).float().to(device)
        labels = torch.from_numpy(labels_cpu).long().to(device)
        
        unique_labels = torch.unique(labels)
        group_rewards = []
        
        # 2. Iterate over each group (Voting Cluster)
        for label in unique_labels:
            # Select lines belonging to this group
            group_indices = (labels == label).nonzero(as_tuple=True)[0]
            group_lines = all_lines[group_indices]
            
            N_group = group_lines.shape[0]
            
            # Need at least 2 lines to find an intersection
            if N_group < 2:
                continue
                
            current_group_best_reward = 0.0
            
            # Sample pairs WITHIN this group
            # If group is small, we can just test all pairs or fewer samples
            samples_to_run = min(num_samples, N_group * (N_group - 1) // 2)
            if samples_to_run < 1: 
                samples_to_run = 1
            
            for _ in range(samples_to_run):
                # Pick 2 random lines
                idx = torch.randperm(N_group, device=device)[:2]
                l1 = group_lines[idx[0]]
                l2 = group_lines[idx[1]]
                
                # Intersection (VP direction)
                vp = torch.linalg.cross(l1, l2)
                vp_norm = torch.norm(vp)
                
                if vp_norm < 1e-6: # Parallel lines
                    continue
                    
                vp = vp / vp_norm
                
                # Compute error of LINES IN THIS GROUP relative to this VP
                # (We don't care if lines in other groups mismatch this VP)
                residuals = torch.abs(torch.matmul(group_lines, vp))
                mean_error = torch.mean(residuals)
                
                # Reward for this sample
                # Using a sharper scale factor (20.0) since we expect inliers to be very good
                sample_reward = 1.0 / (1.0 + 20.0 * mean_error)
                
                if sample_reward > current_group_best_reward:
                    current_group_best_reward = sample_reward.item()
            
            group_rewards.append(current_group_best_reward)
            
        if not group_rewards:
            return 0.0
            
        # 3. Final Reward is Average of Group Rewards
        # (Rewarding the presence of multiple good structures)
        final_reward = sum(group_rewards) / len(group_rewards)
        return final_reward

