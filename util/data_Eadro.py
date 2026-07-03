from torch.utils.data import Dataset, DataLoader
import torch
import util.util as util
class chunkDataset(Dataset): #[node_num, T, else]
    def __init__(self, chunks, node_num, edges, window_size):
        self.dataset = []
        self.logs = []
        self.metrics = []
        self.traces = []
        self.losses = []
        self.idx2id = {}
        self.window = window_size
        self.percent = 0.5
        self.first_graph = None
        for idx, chunk_id in enumerate(chunks.keys()):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]
            #graph = dgl.graph((edges[0], edges[1]), num_nodes=node_num)
            graph = self.create_graph(edges, node_num)
            graph["logs"] = torch.FloatTensor(chunk["logs"])
            graph["metrics"] = torch.FloatTensor(chunk["metrics"])
            graph["traces"] = torch.FloatTensor(chunk["traces"])
            #print(node_num, chunk["logs"].shape, chunk["metrics"].shape, chunk["traces"].shape)
            #self.data.append((graph, chunk["culprit"]))
            self.logs.append(torch.FloatTensor(chunk["logs"])) #<-- no window dim (why ?), aggregated over all window
            self.metrics.append(torch.FloatTensor(chunk["metrics"]))#
            self.traces.append(torch.FloatTensor(chunk["traces"]))
            hot_encode_label = torch.zeros(node_num)
            if chunk["culprit"] >=0:
                hot_encode_label[chunk["culprit"]] = 1.0
            chunk["culprit"] = hot_encode_label
            self.losses.append(chunk["culprit"])
            if self.first_graph is None:
                self.first_graph = graph["adjacency_matrix"]
        #---logs---
        self.logs_array = torch.stack(self.logs)#(B, N,log_dim=1)<-- was aggregated over all window, so now just repeat it
        logs_expanded = np.expand_dims(self.logs_array, axis=1)# Expand (Insert) the window dimension at index 1 --> (B, 1, N, log_dim)
        target_shape = self.logs_array.shape[0], self.window, self.logs_array.shape[1], self.logs_array.shape[2] #Broadcast the single dimension to the required window_size
        self.logs_array = np.broadcast_to(logs_expanded, target_shape) #(B, window, N, 1)

        #---metrics---
        self.metrics_array = torch.stack(self.metrics)#(B, N,window,metric_dim=7)
        self.metrics_array = self.metrics_array.permute(0,2,1,3)#change to (B,window,N,metric_dim)

        #---traces---
        self.traces_array = torch.stack(self.traces)#(B,N,window,trace_dim=2)
        traces_permuted = self.traces_array.permute(0, 2, 1, 3) 
        # 2. Unsqueeze (Add) a new dimension at index 3 (the second 'N' slot)
        # Shape changes to (B, window, N, 1, dim)
        traces_expanded = traces_permuted.unsqueeze(3)
        # 3. Expand (Broadcast) the new dimension (index 3) to be N, 
        # repeating the data N times.
        # self.traces_array.size(1) holds the original N size.
        N_size = self.traces_array.size(1) 
        self.traces_array = traces_expanded.expand(-1, -1, -1, N_size, -1)

        #---losses---
        self.losses_array = torch.stack(self.losses)#(B,N)

        #--labels---
        self.label = np.eye(2)[self.losses_array.numpy().astype(int)]#(B,N,2)
        self.label_mask = self.losses_array.numpy().copy()#(B,N,3)
        times = np.zeros((node_num, 2))  
        for idx in range(self.label.shape[0]):
            if idx < self.window:
                continue
            times += self.label[idx]
            mask = times[self.label[idx] == 1] %10 >= 10*self.percent
            self.label_mask[idx, mask] = 2
        self.label_mask = np.eye(3)[self.label_mask.astype(int)]

        self.dataset = []
        for i in range(self.logs_array.shape[0]):
            record = {}
            record['data_log'] = self.logs_array[i] #(window, N, log_dim)
            record['data_node'] = self.metrics_array[i] #(window, N, metric_dim)
            record['data_edge'] = self.traces_array[i] #(window, N, N, trace_dim)
            record['groundtruth_real'] = self.label[i] #(N, 2)
            record['groundtruth_cls'] = self.label_mask[i] #(N, 3)
            self.dataset.append(record)

        """
        processed.dataset[0]["data_node"].shape --> #(10, 5, 3)
        processed.dataset[0]["data_log"].shape --> #(10, 5, 256)
        processed.dataset[0]["groundtruth_cls"].shape --> #(5, 3) --> label_mask
        processed.dataset[0]["groundtruth_real"].shape --> #(5, 2)  --> label
        processed.dataset[0]["data_edge"].shape --> #(10, 5, 5, 7)
        """


    def create_graph(self, edges, node_num):
        # Convert lists to NumPy arrays for easier processing
        source_nodes = np.array(edges[0])
        destination_nodes = np.array(edges[1])

        # Determine the total number of nodes (num_nodes) by finding the max ID and adding 1
        all_nodes = np.concatenate((source_nodes, destination_nodes))
        num_nodes = all_nodes.max() + 1

        # Initialize the adjacency matrix with zeros
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        # Populate the adjacency matrix: A[u, v] = 1 for an edge from u to v
        for u, v in zip(source_nodes, destination_nodes):
            adjacency_matrix[u, v] = 1
        return {"adjacency_matrix"  :adjacency_matrix}

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]
    def __get_chunk_id__(self, idx):
        return self.idx2id[idx]

from util.utils_Eadro import *

#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("--random_seed", default=42, type=int)
#
#### Training params
#parser.add_argument("--gpu", default=False, type=lambda x: x.lower() == "true")
#parser.add_argument("--epoches", default=5, type=int)
#parser.add_argument("--batch_size", default=256, type=int)
#parser.add_argument("--lr", default=0.001, type=float)
#parser.add_argument("--patience", default=10, type=int)
#
###### Fuse params
#parser.add_argument("--self_attn", default=True, type=lambda x: x.lower() == "true")
#parser.add_argument("--fuse_dim", default=128, type=int)
#parser.add_argument("--alpha", default=0.5, type=float)
#parser.add_argument("--locate_hiddens", default=[64], type=int, nargs='+')
#parser.add_argument("--detect_hiddens", default=[64], type=int, nargs='+')
#
###### Source params
#parser.add_argument("--log_dim", default=16, type=int)
#parser.add_argument("--trace_kernel_sizes", default=[2], type=int, nargs='+')
#parser.add_argument("--trace_hiddens", default=[64], type=int, nargs='+')
#parser.add_argument("--metric_kernel_sizes", default=[2], type=int, nargs='+')
#parser.add_argument("--metric_hiddens", default=[64], type=int, nargs='+')
#parser.add_argument("--graph_hiddens", default=[64], type=int, nargs='+')
#parser.add_argument("--attn_head", default=4, type=int, help="For gat or gat-v2")
#parser.add_argument("--activation", default=0.2, type=float, help="use LeakyReLU, shoule be in (0,1)")
#
###### Data params
#parser.add_argument("--data", type=str,  default="SN")
#parser.add_argument("--result_dir", default="../result/")
#
#params = vars(parser.parse_args())

import logging
def get_device(gpu):
    if gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")
    

def collate(data):
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    return batched_graph , torch.tensor(labels)

def assert_shapes(shapes, dataset):
    #assert select_metirc.shape == (self.window, len(MSDS_pod), 5), f"Worng kpi"
    batch = dataset.metrics_array.shape[0]
    assert dataset.metrics_array.shape == (batch, shapes["window_size"], shapes["node_num"], shapes["metric_dim"]), f"Wrong metrics shape"
    assert dataset.logs_array.shape == (batch, shapes["window_size"], shapes["node_num"], shapes["event_num"]), f"Wrong logs shape"
    assert dataset.traces_array.shape == (batch, shapes["window_size"], shapes["node_num"], shapes["node_num"], shapes["trace_dim"]), f"Wrong traces shape"
    assert dataset.label_mask.shape == (batch, shapes["node_num"],3), f"Wrong losses shape"

def run(data="TT"):
    if util.is_raspberry_pi():
        data_dir = os.path.join("/home/db2003/Desktop/MicroService_Twin_Original/data", data)
    else:
        data_dir = os.path.join("/home/db2003/Desktop/Amr/Tests/Eadro/codes/chunks", data)


    metadata = read_json(os.path.join(data_dir, "metadata.json"))
    event_num, node_num, metric_num =  metadata["event_num"], metadata["node_num"], metadata["metric_num"]
    edges = metadata["edges"]
    chunk_lenth = 10

    train_chunks, test_chunks = load_chunks(data_dir)

    train_data = chunkDataset(train_chunks, node_num, edges,chunk_lenth)
    test_data = chunkDataset(test_chunks, node_num, edges, chunk_lenth)
    
    shapes = {
        "metric_dim": metric_num,
        "trace_dim": 2,
        "window_size": chunk_lenth,
        "node_num": node_num,
        "event_num": event_num
    }
    assert_shapes(shapes,train_data)


    #train_dl = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True, collate_fn=collate, pin_memory=True)
    #test_dl = DataLoader(test_data, batch_size=params["batch_size"], shuffle=False, collate_fn=collate, pin_memory=True)


    #logging.info("Current hash_id {}".format(hash_id))
    return train_data, test_data
if "__main__" == __name__:
    run()


"""
processed.dataset[0]["data_node"].shape --> #(10, 5, 3)
processed.dataset[0]["data_log"].shape --> #(10, 5, 256)
processed.dataset[0]["groundtruth_cls"].shape --> #(5, 3) --> label_mask
processed.dataset[0]["groundtruth_real"].shape --> #(5, 2)  --> label
processed.dataset[0]["data_edge"].shape --> #(10, 5, 5, 7)
"""