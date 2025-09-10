

import torch
import os


def get_device_safe_threading():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["NUMBA_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    return device



def safe_recursive_update(original_dict, update_dict):
    """
    Recursively updates a dictionary with values from another, but only for keys
    that already exist in the original dictionary at each level.

    Args:
        original_dict (dict): The dictionary to be updated (e.g., your base config).
        update_dict (dict): The dictionary containing the new values.

    Returns:
        dict: The updated original dictionary.
    """
    for key, value in update_dict.items():
        # Check if the key exists in the original dictionary
        if key in original_dict:
            # If the key's value is a dictionary in both, recurse
            if isinstance(original_dict.get(key), dict) and isinstance(value, dict):
                safe_recursive_update(original_dict[key], value)
            else:
                # Otherwise, update the value directly
                original_dict[key] = value
    return original_dict