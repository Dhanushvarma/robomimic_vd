import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import pdb
import matplotlib.pyplot as plt
import torch
import wandb
from wandb import plot
import cv2


def calculate_max_column_variance(subgoal_data):
    """
    Calculate the maximum of the column-wise variances for a given (N, 3) numpy array.

    Args:
        subgoal_data (np.array): An (N, 3) numpy array.

    Returns:
        float: The maximum variance value among the three columns.
    """
    # Calculate variance along each column (axis=0)
    variances = np.var(subgoal_data, axis=0)

    # Return the maximum of the three variance values
    return np.max(variances)



def subgoal_ee_pos(subgoal_samples_dict):
    """
    Extracts the position of the end-effector (eef) of robot0 from a dictionary of subgoal samples
    and converts it to a NumPy array in the desired shape.

    Args:
        subgoal_samples_dict (dict): A dictionary containing various subgoal samples. It should
                                     include the key 'robot0_eef_pos', which is a PyTorch tensor.

    Returns:
        numpy.ndarray: A NumPy array containing the end-effector positions, reshaped to [..., 3].

    Raises:
        KeyError: If 'robot0_eef_pos' is not a key in the input dictionary.
        TypeError: If the value associated with 'robot0_eef_pos' is not a PyTorch tensor.
    """
    try:
        # Ensure 'robot0_eef_pos' is in the dictionary
        eef_pos_tensor = subgoal_samples_dict['robot0_eef_pos']

        # Check if it's a PyTorch tensor
        if not isinstance(eef_pos_tensor, torch.Tensor):
            raise TypeError("The 'robot0_eef_pos' entry must be a PyTorch tensor.")

        # Move tensor to CPU, detach from the computation graph, convert to NumPy array, and reshape
        return eef_pos_tensor.cpu().detach().numpy().reshape(-1, 3)

    except KeyError:
        raise KeyError("The key 'robot0_eef_pos' was not found in the input dictionary.")



def choose_subgoal(sg_proposals, choose_subgoal_index):
    """
    Selects a specific subgoal for each key in the provided proposals based on the given index.

    Args:
        sg_proposals (dict): A dictionary containing subgoal proposals, where each key maps to a list of tensors.
        choose_subgoal_index (int): The index of the subgoal to be selected for each key.

    Returns:
        dict: A dictionary with the chosen subgoal for each key.

    Raises:
        IndexError: If the `choose_subgoal_index` is out of range for any of the subgoal lists.
    """
    chosen_subgoal = {}

    for key in sg_proposals:
        if not (0 <= choose_subgoal_index < len(sg_proposals[key][0])):
            raise IndexError(f"Index {choose_subgoal_index} is out of range for key '{key}'.")

        chosen_subgoal[key] = sg_proposals[key][0][choose_subgoal_index].unsqueeze(0)

    return chosen_subgoal


def gaze_subgoal_index(pp_sgs, gaze_input):
    """
    Find the index of the point in pp_sgs that is closest to the single gaze_input point in Euclidean distance.

    Args:
        pp_sgs (np.ndarray): Point positions, an array of shape [N, 2].
        gaze_input (np.ndarray): Gaze input array of shape [1, 2].

    Returns:
        int: Index of the closest point in pp_sgs.
    """
    # Calculate Euclidean distances between gaze_input and each point in pp_sgs
    # Since gaze_input is [1, 2], it will broadcast over [N, 2]
    distances = np.linalg.norm(pp_sgs - gaze_input, axis=1)

    # Find the index of the minimum distance
    min_distance_idx = np.argmin(distances)

    return min_distance_idx