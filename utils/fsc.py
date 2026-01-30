import numpy as np
import math
import time
from scipy.special import comb


def FSC(point1, point2, transform_model, error_t, mode='matrix'):
    """
    Fast Sample Consensus (FSC) for robust point matching.
    """
    M, N = point1.shape
    # Determine minimum number of samples based on the transformation model
    n = {'similarity': 2, 'affine': 3, 'perspective': 4}.get(transform_model, 2)
    
    # Calculate max possible iterations (combination of M points taken n at a time)
    iterations = 2000  # Default iteration limit

    most_consensus_number = 0
    point1_new, point2_new = None, None

    np.random.seed(0)
    # Convert points to homogeneous coordinates (3, M)
    match1_xy = np.concatenate((point1[:, :2].T, np.ones((1, M))))
    match2_xy = np.concatenate((point2[:, :2].T, np.ones((1, M))))

    for i in range(int(iterations)):
        # Randomly select n unique indices
        indices = np.random.choice(M, n, replace=False)
        point1_sample = point1[indices, :2]
        point2_sample = point2[indices, :2]

        # Estimate parameters using Least Squares Method (LSM)
        parameters, _ = LSM(point1_sample, point2_sample, transform_model)
        solution = np.array([[parameters[0], parameters[1], parameters[4]],
                             [parameters[2], parameters[3], parameters[5]],
                             [parameters[6], parameters[7], 1]])

        if transform_model == 'perspective':
            # Apply perspective transformation
            match1_test_trans = np.dot(solution, match1_xy)
            match1_test_trans /= (match1_test_trans[2, :] + 1e-10)
            diff_match2_xy = np.linalg.norm(match1_test_trans[:2, :].T - point2[:, :2], axis=1)
        else:
            # Apply linear transformation (affine/similarity)
            t_match1_xy = np.dot(solution, match1_xy)
            diff_match2_xy = np.linalg.norm(t_match1_xy[:2, :] - match2_xy[:2, :], axis=0)

        # Find inliers based on the error threshold
        index_in = np.flatnonzero(diff_match2_xy < error_t)
        consensus_num = index_in.size

        # Update best model if current consensus is higher
        if consensus_num > most_consensus_number:
            most_consensus_number = consensus_num
            point1_new = point1[index_in, :]
            point2_new = point2[index_in, :]

    if point1_new is None:
        return None, np.inf

    # Remove duplicate matching point pairs
    for _ in range(2):
        unil = point1_new[:, :2]
        _, unique_idx = np.unique(unil, return_index=True, axis=0)
        point1_new = point1_new[unique_idx, :]
        point2_new = point2_new[unique_idx, :]
        
        unil = point2_new[:, :2]
        _, unique_idx = np.unique(unil, return_index=True, axis=0)
        point1_new = point1_new[unique_idx, :]
        point2_new = point2_new[unique_idx, :]

    # Re-calculate final parameters using all inliers
    parameters, rmse = LSM(point1_new[:, :2], point2_new[:, :2], transform_model)
    solution = np.array([[parameters[0], parameters[1], parameters[4]],
                         [parameters[2], parameters[3], parameters[5]],
                         [parameters[6], parameters[7], 1]])
    
    if mode != 'matrix':
        return solution, rmse, point1_new, point2_new
    else:
        return solution, rmse


def LSM(match1, match2, transform_model):
    """
    Least Squares Method for estimating geometric transformations.
    """
    match1_xy = match1[:, :2]  # Extract x, y coordinates
    match2_xy = match2[:, :2]

    num_points = match1_xy.shape[0]  # Number of points
    target_vector = match2_xy.flatten()  # Target vector b

    if transform_model == 'affine':
        A_left = np.zeros((num_points * 2, 4))
        # Map [x, y] to rows for affine estimation
        A_left[::2, :2] = match1_xy    # Even rows, first two columns
        A_left[1::2, 2:] = match1_xy   # Odd rows, last two columns

        A_right = np.tile(np.eye(2), (num_points, 1))
        A = np.hstack((A_left, A_right))

        parameters, _, _, _ = np.linalg.lstsq(A, target_vector, rcond=None)
        parameters = np.append(parameters, [0, 0])

        # Transformation matrix M and translation vector
        M = np.array([[parameters[0], parameters[1]], [parameters[2], parameters[3]]])
        translation = parameters[4:6]

        # Calculate RMSE
        match1_trans = (M @ match1_xy.T).T + translation
        errors = match1_trans - match2_xy
        rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

    elif transform_model == 'perspective':
        A_left = np.zeros((num_points * 2, 4))
        A_left[::2, :2] = match1_xy
        A_left[1::2, 2:] = match1_xy

        A_right = np.tile(np.eye(2), (num_points, 1))
        A = np.hstack((A_left, A_right))

        temp_coords = -match1_xy.repeat(2, axis=0)
        temp_targets = target_vector.repeat(2).reshape(-1, 2)
        temp_cross_terms = temp_coords * temp_targets
        A = np.hstack((A, temp_cross_terms))

        parameters, _, _, _ = np.linalg.lstsq(A, target_vector, rcond=None)

        # Construct transformation matrix M
        match1_homo = np.vstack((match1_xy.T, np.ones(num_points)))
        M = np.array([[parameters[0], parameters[1], parameters[4]],
                      [parameters[2], parameters[3], parameters[5]],
                      [parameters[6], parameters[7], 1]])

        # Normalize perspective coordinates
        match1_trans_homo = M @ match1_homo
        match1_trans_norm = match1_trans_homo[:2, :] / (match1_trans_homo[2, :] + 1e-10)

        # Calculate RMSE
        errors = match1_trans_norm.T - match2_xy
        rmse = np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))

    elif transform_model == 'similarity':
        A = np.zeros((2 * num_points, 4))

        # Fill A matrix for similarity transformation
        A[::2, 0] = match1_xy[:, 0]  # x
        A[::2, 1] = match1_xy[:, 1]  # y
        A[::2, 2] = 1                # 1
        # Row 1, Col 4 is 0

        A[1::2, 0] = match1_xy[:, 1]  # y
        A[1::2, 1] = -match1_xy[:, 0] # -x
        # Row 2, Col 3 is 0
        A[1::2, 3] = 1                # 1

        parameters, _, _, _ = np.linalg.lstsq(A, target_vector, rcond=None)

        # Pad to match standard 8-parameter array format
        parameters = np.pad(parameters, (0, max(0, 8 - parameters.shape[0])), 'constant')

        # Rearrange parameters to follow transformation rules
        parameters[4:6], parameters[2:4] = parameters[2:4].copy(), parameters[4:6].copy()
        parameters[2], parameters[3] = -parameters[1], parameters[0]

        M = np.array([[parameters[0], parameters[1]], [parameters[2], parameters[3]]])
        translation = parameters[4:6]
        
        # Calculate RMSE
        match1_trans = (M @ match1_xy.T).T + translation
        errors = match1_trans - match2_xy
        rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

    return parameters, rmse
