import os
import numpy as np
from datetime import datetime
import locale
# Force the locale to Chinese (Simplified) for Windows
try:
    locale.setlocale(locale.LC_ALL, 'Chinese_China.936') 
except locale.Error:
    # Fallback for systems where the above string isn't recognized
    locale.setlocale(locale.LC_ALL, 'chs')
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import *
from tqdm import tqdm
import torch
import random
import logging

from dataset import LOG_DATASET, METRIC_DATASET, TRACE_DATASET
import dataset.utils as U
from dataset.config import CONFIG_DICT

class Process:
    def __init__(self, **kwargs):

        self.window = kwargs['window']
        self.step = kwargs['step']
        self.dataset_path = kwargs['dataset_path']
        self.rawdata_path = kwargs["data_path"]

        self.log_len = kwargs['log_len']
        self.metric_len = kwargs['raw_node']
        self.num_node = kwargs['num_nodes']
        self.percent = kwargs['label_percent']
        self.set, self.dataset, self.trace_type = {}, [], []

        config = CONFIG_DICT["aiops22"]
        time = datetime.now().strftime("%Y年%m月%d日%H时%M分%S秒")
        U.set_logger(config, time)
        self.__config__ = config
        self.__device__ = U.get_device(config)
        self.__log_data__ = self.__load_dataset__(
            LOG_DATASET[config["dataset"]](config)
        )
        self.__metric_data__ = self.__load_dataset__(
            METRIC_DATASET[config["dataset"]](config)
        )
        self.__trace_data__ = self.__load_dataset__(
            TRACE_DATASET[config["dataset"]](config)
        )

        logging.info("Tranform data into timewindows")
        proccessed_train, proccessed_test = self.load_raw()
        self.proccessed_train = proccessed_train
        self.proccessed_test = proccessed_test
        ==> they need to be saved as their own datasets 
    def __load_dataset__(self, dataset):
        if self.__config__["use_tmp"] and os.path.exists(dataset.get_dataset_path()):
            print(f"Use: cached dataset")
            dataset.load_from_tmp()
        else:
            dataset.load()
            dataset.save_to_tmp()
        return dataset


    def __get_loader__(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        X_train, X_test, y_train, y_test = train_test_split(
            range(len(self.__log_data__)),
            self.__log_data__.y,
            test_size=0.2,
            # stratify=self.__log_data__.y,
        )
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            # stratify=y_train
        )
        data = list(
            zip(
                self.__log_data__.X,
                self.__metric_data__.X,
                self.__trace_data__.X,
                self.__log_data__.y,
            )
        )
        train_data = []
        for index in X_train:
            train_data.append(data[index])
        #train_data = self.__data_enhance__(train_data)
        print("The number of train samples: ", len(train_data))
        print("The number of eval  samples: ", len(X_eval))
        print("The number of test  samples: ", len(X_test))
        return (
            DataLoader(
                train_data,
                batch_size=self.__config__["batch_size"],
                shuffle=True,
                collate_fn=self.__collat_fn__,
            ),
            DataLoader(
                data,
                batch_size=self.__config__["batch_size"],
                sampler=X_eval,
                collate_fn=self.__collat_fn__,
            ),
            DataLoader(
                data,
                batch_size=self.__config__["batch_size"],
                sampler=X_test,
                collate_fn=self.__collat_fn__,
            ),
        )


    def __data_enhance__(self, data):
        y_dict = {}
        for index, sample in enumerate(data):
            if y_dict.get(sample[3], None) is None:
                y_dict[sample[3]] = []
            y_dict[sample[3]].append(index)
        enhance_num = max([len(val) for val in y_dict.values()])
        scheduler = tqdm(total=enhance_num * len(y_dict.keys()), desc="Data enhancing")
        new_data = []
        for _, indices in y_dict.items():
            cnt = len(indices)
            scheduler.update(cnt)
            while cnt < enhance_num:
                new_data.append(self.__fake__(indices, data))
                cnt += 1
                scheduler.update(1)
        scheduler.close()
        data.extend(new_data)
        return data

    def __collat_fn__(self, batch):
        log = []
        metric = []
        trace = []
        y = []
        for _log, _metric, _trace, _y in batch:
            log.append(_log)
            metric.append(_metric)
            trace.append(_trace)
            y.append(_y)
        return (
            (
                torch.tensor(log, dtype=torch.float),
                torch.tensor(metric, dtype=torch.float),
                torch.tensor(trace, dtype=torch.float),
            ),
            torch.tensor(y, dtype=torch.long),
        )

    def __fake__(self, indices, data):
        choices = random.choices(indices, k=2)
        sample1 = data[choices[0]]
        sample2 = data[choices[1]]
        return (
            random.choice([sample1[0], sample2[0]]),
            random.choice([sample1[1], sample2[1]]),
            random.choice([sample1[2], sample2[2]]),
            sample1[3],
        )


    def load_raw(self):
        train_loader, eval_loader, test_loader = self.__get_loader__()
        self.num_node = 0
        def process_loader(loader):
            # history counter for mask generation
            times = None
            dataset = []
            for inputs, labels in tqdm(loader, desc="Processing Aiops data"):
                # -------------------------
                # Original inputs
                logs = inputs[0]       # [B, window, log_dim]
                metric = inputs[1]     # [B, window, T, metric_dim]
                traces = inputs[2]     # [B, trace_dim]
                label_raw = np.array(labels)     # [B]

                B = logs.size(0)
                num_nodes = metric.shape[1]
                if self.num_node == 0:
                    self.num_node = num_nodes
                window = metric.shape[2]
                device = logs.device

                # -------------------------
                # Metric → [B, window, num_node, metric_dim]
                metric = metric.permute(0, 2, 1, 3)

                # -------------------------
                # Logs → [B, window, num_node, log_dim]
                logs = logs.unsqueeze(2).repeat(1, 1, num_nodes, 1)

                # -------------------------
                # Traces → [B, window, num_node, num_node, trace_dim]
                traces = traces.unsqueeze(1) \
                                .unsqueeze(2) \
                                .unsqueeze(3) \
                                .repeat(1, window, num_nodes, num_nodes, 1)

             
                label = np.eye(2)[label_raw.astype(int)]
                label_mask = label_raw.copy()
                times = np.zeros((num_nodes, 2))  
                for idx in range(label.shape[0]):
                    if idx < self.window:
                        continue
                    times += label[idx]
                    mask = times[label[idx] == 1] %10 >= 10*self.percent
                    label_mask[idx, mask] = 2
                label_mask = np.eye(3)[label_mask.astype(int)]

                # Save
                record = {}
                record["logs"]=logs
                record["metric"]=metric
                record["traces"]=traces
                record["label_raw"]=label
                record["label_mask"]=label_mask

                dataset.append(record)
            return dataset
            # Concatenate over batches
            #for k in data:
            #    data[k] = torch.cat(data[k], dim=0)

         
        proccessed_train = process_loader(train_loader)
        proccessed_test = process_loader(test_loader)
        self.graph = self.build_graph()

        return proccessed_train, proccessed_test

    def build_graph(self):
        """
        Mimics the MSDS 5x5 topology for 42 nodes.
        Node 0 (Gateway) connects to all, and all have self-loops.
        """
        num_nodes = self.num_node
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # 1. Self-loops (diagonal = 1)
        np.fill_diagonal(adj_matrix, 1.0)
        
        # 2. Gateway (Node 0) connects to all services
        adj_matrix[0, :] = 1.0 
        
        # 3. All services can reply to Gateway
        adj_matrix[:, 0] = 1.0
        
        return adj_matrix