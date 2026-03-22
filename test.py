
import os
import pickle
def get_case_data_by_id(dataset_path, sample_id):
    """
    Inverted approach: Load the specific modality data from disk 
    using the ID found in the case study JSON.
    """
    file_path = os.path.join(dataset_path, f"{sample_id}_real_real.pkl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data found for ID {sample_id} at {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # data will contain: 'data_node', 'data_log', 'data_edge', 
    # 'groundtruth_cls', 'groundtruth_real'
    return data

data = get_case_data_by_id("data/MSDS-save (new)", "0")
print(data)