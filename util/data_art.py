import gc
import os
import logging
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# Configure logging to see output in console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimizedArtDataProcess:
    """
    Metrics 
        - service names are the cmdb_id in csvs (46 services)
        - feature names are the files themselves 
            - organized through 3 different folders (container, node, jvm)
    """
    def __init__(self, rawdata_path, dataset_path, window_size=10, step=1):
        self.rawdata_path = rawdata_path
        self.dataset_path = dataset_path
        self.window_size = window_size
        self.step = step
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

    def _load_csv_dir(self, path):
        """Helper to load and merge CSVs in a folder efficiently."""
        if not os.path.exists(path):
            logging.warning(f"Path not found: {path}")
            return pd.DataFrame()
        
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        if not files:
            return pd.DataFrame()
        
        # Generator expression inside concat saves memory
        return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    def run(self, debug_mode=False):
        """Main loop that processes data day-by-day."""
        logging.info("Starting sequential memory-optimized loading...")
        
        # Load labels once
        gt_path = os.path.join(os.path.dirname(self.rawdata_path), 'groundtruth', 'groundtruth-all.csv')
        try:
            df_gt = pd.read_csv(gt_path)
        except FileNotFoundError:
            logging.error(f"Ground truth not found at {gt_path}")
            return

        dates = sorted([d for d in os.listdir(self.rawdata_path) if os.path.isdir(os.path.join(self.rawdata_path, d))])
        
        if debug_mode:
            dates = dates[:1] # Process only the first day for testing
            logging.info("--- DEBUG MODE ACTIVE: Processing 1 day only ---")

        service_hash = None
        for i, date_dir in tqdm(enumerate(dates), desc="Processing days"):

            day_path = os.path.join(self.rawdata_path, date_dir)
            
            # 1. Load Metrics
            df_metric = self._load_csv_dir(os.path.join(day_path, 'metric/container'))
            if service_hash is None:
                service_hash = self.build_custom_service_hash(df_metric)
            if df_metric.empty: continue
            df_metric = self.process_metric(df_metric, service_hash)

            # 2. Load Logs
            df_log = self._load_csv_dir(os.path.join(day_path, 'log'))

            # 3. Load Traces
            df_trace = self._load_csv_dir(os.path.join(day_path, 'trace'))

            # 4. Transform and save immediately
            logging.info(f"Transforming windows for {date_dir}...")
            self._transform_and_stream(df_metric, df_log, df_trace, df_gt, date_dir)
            
            # 5. Manual Cleanup
            del df_metric, df_log, df_trace
            gc.collect() 


    def build_custom_service_hash(self, df_metric):
        # Strip the "node-X." prefix from all cmdb_ids in the metric dataframe
        # node-6.emailservice-0 -> emailservice-0
        cleaned_names = df_metric['cmdb_id'].apply(lambda x: x.split('.')[-1]).unique()
        
        all_services = sorted(cleaned_names.tolist())
        service_to_idx = {name: i for i, name in enumerate(all_services)}
        
        # IMPORTANT: Force num_services to match your mapping
        self.num_services = len(service_to_idx) 
        print(f"Total unique services found in metrics: {self.num_services}")
        return service_to_idx

    def process_metric(self, df_metric, service_map):
        # 1. Map names to indices and get unique KPIs/Timestamps
        df_metric['cmdb_id'] = df_metric['cmdb_id'].apply(lambda x: x.split('.')[-1])
        df_metric['node_idx'] = df_metric['cmdb_id'].map(service_map)
        # Drop data for services we don't care about to save space
        df_metric = df_metric.dropna(subset=['node_idx'])

        
        unique_timestamps = np.sort(df_metric['timestamp'].unique())
        unique_kpis = np.sort(df_metric['kpi_name'].unique())
        
        # Create helper maps for coordinates
        time_map = {t: i for i, t in enumerate(unique_timestamps)}
        kpi_map = {k: i for i, k in enumerate(unique_kpis)}
        
        num_times = len(unique_timestamps)
        num_services = len(service_map)  # Fixed based on D1
        num_features = len(unique_kpis)

        logging.info(f"Allocating tensor of shape: ({num_times}, {num_services}, {num_features})")
        
        # 2. Pre-allocate NumPy array with float32 (uses half the RAM of default float64)
        # This acts as your "ensured" container where missing services are automatically 0
        data_tensor = np.zeros((num_times, num_services, num_features), dtype=np.float32)

        # 3. Get the integer coordinates for every row in df_metric
        # We do this vectorized for speed
        time_coords = df_metric['timestamp'].map(time_map).values
        service_coords = df_metric['node_idx'].astype(int).values
        kpi_coords = df_metric['kpi_name'].map(kpi_map).values
        values = df_metric['value'].astype(np.float32).values

        # 4. The "Magic" Step: Direct assignment into the tensor
        # This is lightning fast and doesn't create extra copies of data
        data_tensor[time_coords, service_coords, kpi_coords] = values

        return data_tensor

    def _transform_and_stream(self, df_metric, df_log, df_trace, df_gt, date_label):
        """
        Processes the dataframes into sliding windows and saves 
        each window as a separate pickle file.
        """
        # --- Logic Check ---
        # ArtData column names often differ from MSDS. 
        # MSDS uses 'now', ArtData might use 'timestamp'. 
        # Standardize here if necessary:
        # df_metric.rename(columns={'timestamp': 'now'}, inplace=True)
        
        # Example of saving (to be filled with your specific windowing logic):
        # for i, window_data in enumerate(windows):
        #    save_path = os.path.join(self.dataset_path, f"{date_label}_win_{i}.pkl")
        #    with open(save_path, 'wb') as f:
        #        pickle.dump(window_data, f)
        pass

if __name__ == "__main__":
    processor = OptimizedArtDataProcess(
        rawdata_path='./data/ArtData/aiops-dataset/Aiops-Dataset/data',
        dataset_path='./data/ArtData-processed',
        window_size=10,
        step=1
    )
    # Run with debug_mode=True first to verify it works on 1 day
    processor.run(debug_mode=True)




"""

# preprocessed data combines all of metrics, logs, traces in 1 vector for each service node
# so i need to build my own preprocessing that saves metrics, logs, traces separately

import dgl
import dgl.heterograph
import sys

# Patch every possible location pickle might look
dgl.heterograph.DGLHeteroGraph = dgl.DGLGraph
if hasattr(dgl, 'DGLGraph'):
    sys.modules['dgl.heterograph'] = dgl.heterograph
    setattr(dgl.heterograph, 'DGLHeteroGraph', dgl.DGLGraph)

# Now try loading your data
train_path = "/home/db2003/Desktop/Amr/Tests/ART/data/D1/samples/train_samples.pkl"
test_path = "/home/db2003/Desktop/Amr/Tests/ART/data/D1/samples/test_samples.pkl"

with open(train_path, "rb") as f:
    train_data = pickle.load(f)

with open(test_path, "rb") as f:
    test_data = pickle.load(f)

print("Train Data Keys:", train_data.keys())
print("Test Data Keys:", test_data.keys())
"""