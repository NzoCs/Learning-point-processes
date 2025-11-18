import copy
import json
import pickle

from tqdm import tqdm


def py_assert(condition, exception_type, msg):
    """An assert function that ensures the condition holds, otherwise throws a message.

    Args:
        condition (bool): a formula to ensure validity.
        exception_type (_StandardError): Error type, such as ValueError.
        msg (str): a message to throw out.

    Raises:
        exception_type: throw an error when the condition does not hold.
    """
    if not condition:
        raise exception_type(msg)


def load_pickle(file_dir):
    """Load from pickle file.

    Args:
        file_dir (BinaryIO): dir of the pickle file.

    Returns:
        any type: the loaded data.
    """
    with open(file_dir, "rb") as file:
        try:
            data = pickle.load(file, encoding="latin-1")
        except Exception:
            data = pickle.load(file)

    return data


def save_json(data, file_dir):
    """
    Save data to a JSON file.

    Args:
        data: The data to be saved. It should be JSON serializable (e.g., a dictionary or list).
        file_dir (str): The path to the file where the data will be saved.

    Raises:
        IOError: If the file cannot be opened or written to.
    """
    with open(file_dir, "w") as outfile:
        json.dump(data, outfile, indent=4)
    print(f"Data successfully saved to {file_dir}")


def dict_deep_update(target, source, is_add_new_key=True):
    """Update 'target' dict by 'source' dict deeply, and return a new dict copied from target and source deeply.

    Args:
        target: dict
        source: dict
        is_add_new_key: bool, default True.
            Identify if add a key that in source but not in target into target.

    Returns:
        New target: dict. It contains the both target and source values, but keeps the values from source when the key
        is duplicated.
    """
    # deep copy for avoiding to modify the original dict
    result = copy.deepcopy(target) if target is not None else {}

    if source is None:
        return result

    for key, value in source.items():
        if key not in result:
            if is_add_new_key:
                result[key] = value
            continue
        # both target and source have the same key
        base_type_list = [int, float, str, tuple, bool]
        if type(result[key]) in base_type_list or type(source[key]) in base_type_list:
            result[key] = value
        else:
            result[key] = dict_deep_update(
                result[key], source[key], is_add_new_key=is_add_new_key
            )
    return result


def format_multivariate_simulations(simulations: list[dict], dim_process) -> list[dict]:
    """
    Formats the raw simulation results into a list of dictionaries, one per sequence.

    Each dictionary follows a structure similar to Hugging Face datasets,
    containing event times, time deltas, event types, sequence length, etc.

    Args:
        simulations (List[Dict]): A list where each dict contains tensors
                                    ('time_seq', 'time_delta_seq', 'event_seq')
                                    for a single simulated sequence.
        dim_process (Optional[int]): The number of event types (dimensionality) in the process.

    Returns:
        List[Dict]: A list of dictionaries, each representing a formatted sequence.
    """
    formatted_data = []

    for seq_idx, sim in enumerate(tqdm(simulations, desc="Formatting sequences")):
        times = sim["time_seq"]
        events = sim["event_seq"]
        time_deltas = sim["time_delta_seq"]

        times = times - times[0]

        times_list = times.cpu().tolist()
        events_list = events.cpu().long().tolist()
        time_deltas_list = time_deltas.cpu().tolist()

        seq_dict = {
            "dim_process": dim_process if dim_process is not None else -1,
            "seq_len": len(times_list),
            "seq_idx": seq_idx,
            "time_since_start": times_list,
            "time_since_last_event": time_deltas_list,
            "type_event": events_list,
        }
        formatted_data.append(seq_dict)

    return formatted_data
