"""
Geometric utilities for line normalization and back-projection.
Adapted from GlobustVP for DDPO-PyTorch integration.
"""
import numpy as np


def normalize_lines(K: np.ndarray, lines_2D: np.ndarray) -> np.ndarray:
    """
    Normalize 2D line segments by applying the inverse of the camera intrinsic matrix.

    Parameters:
        K : np.ndarray
            Camera intrinsic matrix, shape (3, 3).
        lines_2D : np.ndarray
            2D line segments in homogeneous image coordinates,
            where each column is [x1, y1, 1, x2, y2, 1]^T, shape (6, N).

    Returns:
        np.ndarray
            Normalized line endpoints in camera coordinates,
            where each column is [x1, y1, x2, y2]^T, shape (4, N).
    """
    K_inv = np.linalg.inv(K)

    # Apply inverse calibration to both endpoints
    pts1_h = K_inv @ lines_2D[0:3, :]  # First endpoint in homogeneous coords
    pts2_h = K_inv @ lines_2D[3:6, :]  # Second endpoint

    # Convert to inhomogeneous coordinates
    pts1 = pts1_h[:2, :] / pts1_h[2:, :]
    pts2 = pts2_h[:2, :] / pts2_h[2:, :]

    return np.vstack((pts1, pts2))


def compute_backprojection_normals(lines_2D: np.ndarray) -> np.ndarray:
    """
    Compute 3D plane normals by back-projecting 2D image line segments.

    Parameters:
        lines_2D : np.ndarray
            Normalized 2D line segments,
            where each row is [x1, y1, x2, y2], shape (N, 4).

    Returns:
        para_lines : np.ndarray
            Unit-norm 3D normals of the back-projection planes, shape (N, 3).
    """
    assert lines_2D.shape[1] == 4, "Input must have shape (N, 4)."

    x1 = lines_2D[:, 0:2]  # shape (N, 2)
    x2 = lines_2D[:, 2:4]  # shape (N, 2)

    # Convert to homogeneous coordinates
    x1_h = np.hstack([x1, np.ones((x1.shape[0], 1))])  # shape (N, 3)
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])  # shape (N, 3)

    # Cross product to get plane normals
    para_lines = np.cross(x1_h, x2_h)
    para_lines /= np.linalg.norm(para_lines, axis=1, keepdims=True)

    return para_lines


def skew(v: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix of a 3D vector.

    The resulting matrix [v]_x satisfies [v]_x @ w == np.cross(v, w) for any 3D vector w.

    Parameters:
        v : np.ndarray
            Input 3D vector, shape (3,).

    Returns:
        np.ndarray
            A 3x3 skew-symmetric matrix corresponding to the input vector.
    """
    assert v.shape == (3,), "Input vector must be 3-dimensional."
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0]
    ])


def line_uncertainty(
    K: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray
) -> float:
    """
    Compute the uncertainty of a line defined by two image points, 
    propagated through the camera intrinsics to 3D space.

    Parameters:
        K : np.ndarray
            Camera intrinsic matrix, shape (3, 3).
        start_point : np.ndarray
            Start point of the line, shape (2,).
        end_point : np.ndarray
            End point of the line, shape (2,).

    Returns:
        uncertainty : float
            Inverse trace-based uncertainty measure.
    """
    # Homogeneous coordinates
    p1_h = np.append(start_point, 1.0)
    p2_h = np.append(end_point, 1.0)

    # Isotropic 2D point covariance (lifted to 3D)
    Sigma_2D = 2 * np.eye(2)
    Sigma_h = np.zeros((3, 3))
    Sigma_h[:2, :2] = Sigma_2D

    # Transform to normalized camera coordinates
    K_inv = np.linalg.inv(K)
    Sigma_1_h = K_inv @ Sigma_h @ K_inv.T
    Sigma_2_h = Sigma_1_h  # Same covariance for both points

    # 3D line from cross product
    l_3d = np.cross(p1_h, p2_h)
    norm_l = np.linalg.norm(l_3d)
    l_3d_normalized = l_3d / norm_l

    # Covariance propagation for cross product
    Sigma_l = (
        skew(p2_h) @ Sigma_1_h @ skew(p2_h).T +
        skew(p1_h) @ Sigma_2_h @ skew(p1_h).T
    )

    # Jacobian projection matrix for normalization
    J = (np.eye(3) - np.outer(l_3d_normalized, l_3d_normalized)) / norm_l
    Sigma_l_normalized = J @ Sigma_l @ J.T

    # Uncertainty from trace
    uncertainty = 1.0 / np.trace(Sigma_l_normalized)
    return uncertainty


def compute_line_uncertainties(
    lines_2D: np.ndarray,
    K: np.ndarray,
    use_uncertainty: bool = False
) -> np.ndarray:
    """
    Compute normalized uncertainty weights for a set of 2D line segments.

    Parameters:
        lines_2D : np.ndarray
            Normalized 2D line segments,
            where each row is [x1, y1, x2, y2], shape (N, 4).
        K : np.ndarray
            Camera intrinsic matrix, shape (3, 3).
        use_uncertainty : bool, optional, default=False
            Whether to compute uncertainty based on geometry.
            If False, returns uniform weights (all ones).

    Returns:
        uncertainty : np.ndarray
            Normalized uncertainty weights, shape (N, 1).
    """
    total_num = lines_2D.shape[0]
    uncertainty = np.ones((total_num, 1))

    if use_uncertainty:
        for i in range(total_num):
            u = line_uncertainty(K, lines_2D[i, :2], lines_2D[i, 2:4])
            uncertainty[i] = u
        # Normalize to [0.1, 0.3] for numerical stability
        min_u, max_u = np.min(uncertainty), np.max(uncertainty)
        uncertainty = 0.1 + (uncertainty - min_u) * 0.2 / (max_u - min_u + 1e-8)

    return uncertainty
