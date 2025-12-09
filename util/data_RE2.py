import os
import pickle
import sys
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

class Process:
    def __init__(self, **kwargs):
        # Single pod for RQ2
        self.num_node = 1

        self.window = kwargs['window']
        self.step = kwargs['step']
        self.dataset_path = kwargs['dataset_path']
        self.rawdata_path = kwargs["data_path"]

        self.log_len = kwargs['log_len']
        self.metric_len = kwargs['raw_node']
        self.percent = kwargs['label_percent']

        self.set = {}
        self.dataset = []
        self.trace_type = []

        # Load raw data and process
        self._load_raw()
        self._process_label_mask()
        self.dataset = self._transform()
        self.save_data(self.dataset_path)
# time	adservice_container-cpu-system-seconds-total	cartservice_container-cpu-system-seconds-total	checkoutservice_container-cpu-system-seconds-total	currencyservice_container-cpu-system-seconds-total	emailservice_container-cpu-system-seconds-total	frontend_container-cpu-system-seconds-total	paymentservice_container-cpu-system-seconds-total	productcatalogservice_container-cpu-system-seconds-total	recommendationservice_container-cpu-system-seconds-total	redis_container-cpu-system-seconds-total	shippingservice_container-cpu-system-seconds-total	adservice_container-cpu-usage-seconds-total	cartservice_container-cpu-usage-seconds-total	checkoutservice_container-cpu-usage-seconds-total	currencyservice_container-cpu-usage-seconds-total	emailservice_container-cpu-usage-seconds-total	frontend_container-cpu-usage-seconds-total	paymentservice_container-cpu-usage-seconds-total	productcatalogservice_container-cpu-usage-seconds-total	recommendationservice_container-cpu-usage-seconds-total	redis_container-cpu-usage-seconds-total	shippingservice_container-cpu-usage-seconds-total	adservice_container-cpu-user-seconds-total	cartservice_container-cpu-user-seconds-total	checkoutservice_container-cpu-user-seconds-total	currencyservice_container-cpu-user-seconds-total	emailservice_container-cpu-user-seconds-total	frontend_container-cpu-user-seconds-total	paymentservice_container-cpu-user-seconds-total	productcatalogservice_container-cpu-user-seconds-total	recommendationservice_container-cpu-user-seconds-total	redis_container-cpu-user-seconds-total	shippingservice_container-cpu-user-seconds-total	adservice_container-spec-cpu-quota	cartservice_container-spec-cpu-quota	checkoutservice_container-spec-cpu-quota	currencyservice_container-spec-cpu-quota	emailservice_container-spec-cpu-quota	frontend_container-spec-cpu-quota	paymentservice_container-spec-cpu-quota	productcatalogservice_container-spec-cpu-quota	recommendationservice_container-spec-cpu-quota	redis_container-spec-cpu-quota	shippingservice_container-spec-cpu-quota	adservice_container-memory-cache	cartservice_container-memory-cache	checkoutservice_container-memory-cache	currencyservice_container-memory-cache	emailservice_container-memory-cache	frontend_container-memory-cache	paymentservice_container-memory-cache	productcatalogservice_container-memory-cache	recommendationservice_container-memory-cache	redis_container-memory-cache	shippingservice_container-memory-cache	adservice_container-memory-failures-total	cartservice_container-memory-failures-total	checkoutservice_container-memory-failures-total	currencyservice_container-memory-failures-total	emailservice_container-memory-failures-total	frontend_container-memory-failures-total	paymentservice_container-memory-failures-total	productcatalogservice_container-memory-failures-total	recommendationservice_container-memory-failures-total	redis_container-memory-failures-total	shippingservice_container-memory-failures-total	adservice_container-memory-mapped-file	cartservice_container-memory-mapped-file	checkoutservice_container-memory-mapped-file	currencyservice_container-memory-mapped-file	emailservice_container-memory-mapped-file	frontend_container-memory-mapped-file	paymentservice_container-memory-mapped-file	productcatalogservice_container-memory-mapped-file	recommendationservice_container-memory-mapped-file	redis_container-memory-mapped-file	shippingservice_container-memory-mapped-file	adservice_container-memory-max-usage-bytes	cartservice_container-memory-max-usage-bytes	checkoutservice_container-memory-max-usage-bytes	currencyservice_container-memory-max-usage-bytes	emailservice_container-memory-max-usage-bytes	frontend_container-memory-max-usage-bytes	paymentservice_container-memory-max-usage-bytes	productcatalogservice_container-memory-max-usage-bytes	recommendationservice_container-memory-max-usage-bytes	redis_container-memory-max-usage-bytes	shippingservice_container-memory-max-usage-bytes	adservice_container-memory-rss	cartservice_container-memory-rss	checkoutservice_container-memory-rss	currencyservice_container-memory-rss	emailservice_container-memory-rss	frontend_container-memory-rss	paymentservice_container-memory-rss	productcatalogservice_container-memory-rss	recommendationservice_container-memory-rss	redis_container-memory-rss	shippingservice_container-memory-rss	adservice_container-memory-swap	cartservice_container-memory-swap	checkoutservice_container-memory-swap	currencyservice_container-memory-swap	emailservice_container-memory-swap	frontend_container-memory-swap	paymentservice_container-memory-swap	productcatalogservice_container-memory-swap	recommendationservice_container-memory-swap	redis_container-memory-swap	shippingservice_container-memory-swap	adservice_container-memory-usage-bytes	cartservice_container-memory-usage-bytes	checkoutservice_container-memory-usage-bytes	currencyservice_container-memory-usage-bytes	emailservice_container-memory-usage-bytes	frontend_container-memory-usage-bytes	paymentservice_container-memory-usage-bytes	productcatalogservice_container-memory-usage-bytes	recommendationservice_container-memory-usage-bytes	redis_container-memory-usage-bytes	shippingservice_container-memory-usage-bytes	adservice_container-memory-working-set-bytes	cartservice_container-memory-working-set-bytes	checkoutservice_container-memory-working-set-bytes	currencyservice_container-memory-working-set-bytes	emailservice_container-memory-working-set-bytes	frontend_container-memory-working-set-bytes	paymentservice_container-memory-working-set-bytes	productcatalogservice_container-memory-working-set-bytes	recommendationservice_container-memory-working-set-bytes	redis_container-memory-working-set-bytes	shippingservice_container-memory-working-set-bytes	adservice_container-spec-memory-limit-bytes	cartservice_container-spec-memory-limit-bytes	checkoutservice_container-spec-memory-limit-bytes	currencyservice_container-spec-memory-limit-bytes	emailservice_container-spec-memory-limit-bytes	frontend_container-spec-memory-limit-bytes	paymentservice_container-spec-memory-limit-bytes	productcatalogservice_container-spec-memory-limit-bytes	recommendationservice_container-spec-memory-limit-bytes	redis_container-spec-memory-limit-bytes	shippingservice_container-spec-memory-limit-bytes	adservice_container-blkio-device-usage-total	emailservice_container-blkio-device-usage-total	recommendationservice_container-blkio-device-usage-total	redis_container-blkio-device-usage-total	adservice_container-fs-writes-total	emailservice_container-fs-writes-total	recommendationservice_container-fs-writes-total	redis_container-fs-writes-total	adservice_container-fs-writes-bytes-total	emailservice_container-fs-writes-bytes-total	recommendationservice_container-fs-writes-bytes-total	redis_container-fs-writes-bytes-total	adservice_container-fs-reads-total	emailservice_container-fs-reads-total	recommendationservice_container-fs-reads-total	redis_container-fs-reads-total	adservice_container-fs-reads-bytes-total	emailservice_container-fs-reads-bytes-total	recommendationservice_container-fs-reads-bytes-total	redis_container-fs-reads-bytes-total	adservice_container-sockets	cartservice_container-sockets	checkoutservice_container-sockets	currencyservice_container-sockets	emailservice_container-sockets	frontend_container-sockets	paymentservice_container-sockets	productcatalogservice_container-sockets	recommendationservice_container-sockets	redis_container-sockets	shippingservice_container-sockets	adservice_container-network-transmit-packets-total	cartservice_container-network-transmit-packets-total	checkoutservice_container-network-transmit-packets-total	currencyservice_container-network-transmit-packets-total	emailservice_container-network-transmit-packets-total	frontend_container-network-transmit-packets-total	loadgenerator_container-network-transmit-packets-total	paymentservice_container-network-transmit-packets-total	productcatalogservice_container-network-transmit-packets-total	recommendationservice_container-network-transmit-packets-total	redis_container-network-transmit-packets-total	shippingservice_container-network-transmit-packets-total	adservice_container-network-transmit-packets-dropped-total	cartservice_container-network-transmit-packets-dropped-total	checkoutservice_container-network-transmit-packets-dropped-total	currencyservice_container-network-transmit-packets-dropped-total	emailservice_container-network-transmit-packets-dropped-total	frontend_container-network-transmit-packets-dropped-total	loadgenerator_container-network-transmit-packets-dropped-total	paymentservice_container-network-transmit-packets-dropped-total	productcatalogservice_container-network-transmit-packets-dropped-total	recommendationservice_container-network-transmit-packets-dropped-total	redis_container-network-transmit-packets-dropped-total	shippingservice_container-network-transmit-packets-dropped-total	adservice_container-network-transmit-bytes-total	cartservice_container-network-transmit-bytes-total	checkoutservice_container-network-transmit-bytes-total	currencyservice_container-network-transmit-bytes-total	emailservice_container-network-transmit-bytes-total	frontend_container-network-transmit-bytes-total	loadgenerator_container-network-transmit-bytes-total	paymentservice_container-network-transmit-bytes-total	productcatalogservice_container-network-transmit-bytes-total	recommendationservice_container-network-transmit-bytes-total	redis_container-network-transmit-bytes-total	shippingservice_container-network-transmit-bytes-total	adservice_container-network-transmit-errors-total	cartservice_container-network-transmit-errors-total	checkoutservice_container-network-transmit-errors-total	currencyservice_container-network-transmit-errors-total	emailservice_container-network-transmit-errors-total	frontend_container-network-transmit-errors-total	loadgenerator_container-network-transmit-errors-total	paymentservice_container-network-transmit-errors-total	productcatalogservice_container-network-transmit-errors-total	recommendationservice_container-network-transmit-errors-total	redis_container-network-transmit-errors-total	shippingservice_container-network-transmit-errors-total	adservice_container-network-receive-packets-total	cartservice_container-network-receive-packets-total	checkoutservice_container-network-receive-packets-total	currencyservice_container-network-receive-packets-total	emailservice_container-network-receive-packets-total	frontend_container-network-receive-packets-total	loadgenerator_container-network-receive-packets-total	paymentservice_container-network-receive-packets-total	productcatalogservice_container-network-receive-packets-total	recommendationservice_container-network-receive-packets-total	redis_container-network-receive-packets-total	shippingservice_container-network-receive-packets-total	adservice_container-network-receive-packets-dropped-total	cartservice_container-network-receive-packets-dropped-total	checkoutservice_container-network-receive-packets-dropped-total	currencyservice_container-network-receive-packets-dropped-total	emailservice_container-network-receive-packets-dropped-total	frontend_container-network-receive-packets-dropped-total	loadgenerator_container-network-receive-packets-dropped-total	paymentservice_container-network-receive-packets-dropped-total	productcatalogservice_container-network-receive-packets-dropped-total	recommendationservice_container-network-receive-packets-dropped-total	redis_container-network-receive-packets-dropped-total	shippingservice_container-network-receive-packets-dropped-total	adservice_container-network-receive-bytes-total	cartservice_container-network-receive-bytes-total	checkoutservice_container-network-receive-bytes-total	currencyservice_container-network-receive-bytes-total	emailservice_container-network-receive-bytes-total	frontend_container-network-receive-bytes-total	loadgenerator_container-network-receive-bytes-total	paymentservice_container-network-receive-bytes-total	productcatalogservice_container-network-receive-bytes-total	recommendationservice_container-network-receive-bytes-total	redis_container-network-receive-bytes-total	shippingservice_container-network-receive-bytes-total	adservice_container-network-receive-errors-total	cartservice_container-network-receive-errors-total	checkoutservice_container-network-receive-errors-total	currencyservice_container-network-receive-errors-total	emailservice_container-network-receive-errors-total	frontend_container-network-receive-errors-total	loadgenerator_container-network-receive-errors-total	paymentservice_container-network-receive-errors-total	productcatalogservice_container-network-receive-errors-total	recommendationservice_container-network-receive-errors-total	redis_container-network-receive-errors-total	shippingservice_container-network-receive-errors-total	adservice_istio-request-total	cartservice_istio-request-total	checkoutservice_istio-request-total	currencyservice_istio-request-total	emailservice_istio-request-total	frontend_istio-request-total	frontend-external_istio-request-total	paymentservice_istio-request-total	productcatalogservice_istio-request-total	recommendationservice_istio-request-total	shippingservice_istio-request-total	currencyservice_istio-error-total	frontend_istio-error-total	frontend-external_istio-error-total	productcatalogservice_istio-error-total	adservice_istio-latency-50	cartservice_istio-latency-50	checkoutservice_istio-latency-50	currencyservice_istio-latency-50	emailservice_istio-latency-50	frontend_istio-latency-50	paymentservice_istio-latency-50	productcatalogservice_istio-latency-50	recommendationservice_istio-latency-50	shippingservice_istio-latency-50	adservice_istio-latency-90	cartservice_istio-latency-90	checkoutservice_istio-latency-90	currencyservice_istio-latency-90	emailservice_istio-latency-90	frontend_istio-latency-90	paymentservice_istio-latency-90	productcatalogservice_istio-latency-90	recommendationservice_istio-latency-90	shippingservice_istio-latency-90	adservice_istio-latency-95	cartservice_istio-latency-95	checkoutservice_istio-latency-95	currencyservice_istio-latency-95	emailservice_istio-latency-95	frontend_istio-latency-95	paymentservice_istio-latency-95	productcatalogservice_istio-latency-95	recommendationservice_istio-latency-95	shippingservice_istio-latency-95	adservice_istio-latency-99	cartservice_istio-latency-99	checkoutservice_istio-latency-99	currencyservice_istio-latency-99	emailservice_istio-latency-99	frontend_istio-latency-99	paymentservice_istio-latency-99	productcatalogservice_istio-latency-99	recommendationservice_istio-latency-99	shippingservice_istio-latency-99	adservice_istio-bytes-50	cartservice_istio-bytes-50	checkoutservice_istio-bytes-50	currencyservice_istio-bytes-50	emailservice_istio-bytes-50	frontend_istio-bytes-50	paymentservice_istio-bytes-50	productcatalogservice_istio-bytes-50	recommendationservice_istio-bytes-50	shippingservice_istio-bytes-50	adservice_istio-bytes-90	cartservice_istio-bytes-90	checkoutservice_istio-bytes-90	currencyservice_istio-bytes-90	emailservice_istio-bytes-90	frontend_istio-bytes-90	paymentservice_istio-bytes-90	productcatalogservice_istio-bytes-90	recommendationservice_istio-bytes-90	shippingservice_istio-bytes-90	adservice_istio-bytes-95	cartservice_istio-bytes-95	checkoutservice_istio-bytes-95	currencyservice_istio-bytes-95	emailservice_istio-bytes-95	frontend_istio-bytes-95	paymentservice_istio-bytes-95	productcatalogservice_istio-bytes-95	recommendationservice_istio-bytes-95	shippingservice_istio-bytes-95	adservice_istio-bytes-99	cartservice_istio-bytes-99	checkoutservice_istio-bytes-99	currencyservice_istio-bytes-99	emailservice_istio-bytes-99	frontend_istio-bytes-99	paymentservice_istio-bytes-99	productcatalogservice_istio-bytes-99	recommendationservice_istio-bytes-99	shippingservice_istio-bytes-99	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-cpu-seconds-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-cpu-seconds-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-cpu-seconds-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-cpu-seconds-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-memory-active-bytes	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-memory-active-bytes	gke-gke-cluster-default-pool-2e1807ce-w819_node-memory-active-bytes	gke-gke-cluster-default-pool-2e1807ce-xte3_node-memory-active-bytes	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-memory-inactive-bytes	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-memory-inactive-bytes	gke-gke-cluster-default-pool-2e1807ce-w819_node-memory-inactive-bytes	gke-gke-cluster-default-pool-2e1807ce-xte3_node-memory-inactive-bytes	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-disk-reads-completed-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-disk-reads-completed-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-disk-reads-completed-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-disk-reads-completed-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-disk-writes-completed-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-disk-writes-completed-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-disk-writes-completed-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-disk-writes-completed-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-disk-read-bytes-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-disk-read-bytes-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-disk-read-bytes-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-disk-read-bytes-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-disk-written-bytes-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-disk-written-bytes-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-disk-written-bytes-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-disk-written-bytes-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-network-receive-packets-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-network-receive-packets-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-network-receive-packets-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-network-receive-packets-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-network-transmit-packets-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-network-transmit-packets-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-network-transmit-packets-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-network-transmit-packets-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-network-receive-errs-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-network-receive-errs-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-network-receive-errs-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-network-receive-errs-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-network-transmit-errs-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-network-transmit-errs-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-network-transmit-errs-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-network-transmit-errs-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-network-receive-drop-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-network-receive-drop-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-network-receive-drop-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-network-receive-drop-total	gke-gke-cluster-default-pool-2e1807ce-0e4z_node-network-transmit-drop-total	gke-gke-cluster-default-pool-2e1807ce-cx8g_node-network-transmit-drop-total	gke-gke-cluster-default-pool-2e1807ce-w819_node-network-transmit-drop-total	gke-gke-cluster-default-pool-2e1807ce-xte3_node-network-transmit-drop-total

    def safe_memory_available(self, gb_required=1.0):
            """Return True if at least gb_required GB of RAM is available."""
            mem = psutil.virtual_memory()
            return mem.available > gb_required * (1024**3)

    def _load_raw(self):
        raw = self.rawdata_path

        #-- get number of pods ---
        pod_file = raw + "/pod-node-1.csv"
        pod_df = pd.read_csv(pod_file)
        pod_list = pod_df['POD'].unique().tolist()
        self.num_node = len(pod_list)
        self.pod_list = pod_list

        # --- METRICS ---
        print("Loading metric data...")
        metric_file = raw + "/metrics.csv"
        metric = pd.read_csv(metric_file)
        if 'now' not in metric.columns:
            metric = metric.rename(columns={metric.columns[0]: 'now'})
        metric = metric.set_index('now').sort_index()
        metric = (metric - metric.min()) / (metric.max() - metric.min() + 1e-6)
        metric = metric.fillna(0)
        timestart, timeend = metric.index.min(), metric.index.max()
        self.time_list = [item for item in range(int(timestart), int(timeend)+1, 1)]
        # remove columns with name starting with 'gke-gke-cluster'
        metric = metric.loc[:, ~metric.columns.str.startswith('gke-gke-cluster')]
        # choose the first 33 columns, as they cover all the 11 pods
        metric = metric.iloc[:, :11*33].values.reshape(len(metric), 11, -1) # reshape to (N, pods, -1)
        empty_pod_data = np.zeros((len(metric), 1, metric.shape[2]))  # shape (N, 1, features)
        metric = np.concatenate((metric, empty_pod_data), axis=1)  # shape (N, 12, features)
        self.set['metric'] = metric
    
        

        # --- LOGS ---
        log_file = raw + "/logs.csv"
        log = pd.read_csv(log_file)
        # rename columns
        log = log.rename(columns={'container_name': 'Hostname', 'log_template': 'templateid', 'timestamp': '@timestamp'})
        log = log.sort_values(by='@timestamp', ascending=True)
        log_record = {}
        max_record = np.zeros(self.log_len)
        min_record = np.ones(self.log_len)
        i = 0
        for timestamp, data in tqdm(log.groupby(['@timestamp'])):
            i+=1
            if i % 1000 ==0:
                if not self.safe_memory_available(0.5):
                    print("[SAFE EXIT] Not enough RAM during log processing.")
                    sys.exit(1)
            
            new = np.zeros((self.num_node, self.log_len))
            for idx, item in data.groupby(['Hostname', 'templateid']):
                if idx[0] not in pod_list:
                    continue
                new[pod_list.index(idx[0]), idx[1] - 1] = item.shape[0]
            log_record[timestamp] = new
            new = new.max(axis=0)
            max_record = np.where(new > max_record, new, max_record)
            min_record = np.where(new < min_record, new, min_record)

        if len(log_record) != (timeend - timestart + 1):
            for item in self.time_list:
                if item not in log_record:
                    log_record[item] = np.zeros((self.num_node, self.log_len))
            min_record = np.zeros(self.log_len)
        
        dis = max_record - min_record + 1e-6
        for name, item in log_record.items():
            log_record[name] = (item - min_record) / dis

        self.set['log'] = log_record
        # --- TRACE ---
        print("Loading trace (pre-aggregated) data...")
        self.load_trace_full()
        aa=1
    def load_trace_full(self):
        trace_raw = pd.read_csv(os.path.join(self.rawdata_path, "traces.csv"))

        # --- Rename columns to match old code ---
        trace_raw = trace_raw.rename(columns={
            'serviceName': 'cmbd_id',
            'parentSpanID': 'fatherpod',
            'methodName': 'stats',
            'startTime': 'start_time'
        })

        # --- Create end_time = start_time + duration ---
        # Your timestamps are in microseconds → convert to seconds (old code expects seconds)
        trace_raw['end_time'] = (trace_raw['start_time'] + trace_raw['duration']) // 1000

        # --- Sort like original code ---
        trace_raw = trace_raw.sort_values(by='end_time', ascending=True)

        # --- MSDS_pod contains 1 element ---
        MSDS_pod = self.pod_list

        # --- Extend trace types ---
        self.trace_type.extend(trace_raw['stats'].fillna("unknown").unique().tolist())

        # --- Allocate tensor ---
        trace_a = np.zeros((len(MSDS_pod), len(MSDS_pod), len(self.trace_type), len(self.time_list)))

        # --- Same groupby keys as original code ---
        i=0
        for name, item in tqdm(trace_raw.groupby(['cmbd_id', 'fatherpod', 'stats', 'end_time'])):
            src, dst, stype, t = name
            i+=1
            if i % 1000 ==0:
                if not self.safe_memory_available(0.5):
                    print("[SAFE EXIT] Not enough RAM during trace processing.")
                    sys.exit(1)

            if src not in MSDS_pod or dst not in MSDS_pod or t > self.time_list[-1]:
                continue

            trace_a[
                MSDS_pod.index(src),
                MSDS_pod.index(dst),
                self.trace_type.index(stype),
                int(t - self.time_list[0])
            ] = item['duration'].sum()

        # --- Same normalization step ---
        trace = trace_a.transpose(3, 0, 1, 2) / (trace_a.mean(axis=-1) * 10 + 1e-6)
        self.set['trace'] = trace
        print("[OK] Trace tensor ready:", trace.shape)

    def _process_label_mask(self):
        inject_file = self.rawdata_path + "/inject_time.txt"
        inject_time = float(open(inject_file).read().strip()) if os.path.isfile(inject_file) else 0

        times = np.array(self.time_list) #self.set['metric'].index.to_numpy()
        N = len(times)

        # initialize integer labels first: 0 = normal, 1 = anomaly
        label_raw = np.zeros((N, self.num_node), dtype=int)
        label_raw[times >= inject_time, :] = 1

        # convert to one-hot for 2 classes: shape (N, pods, 2)
        label_onehot = np.eye(2)[label_raw]  # automatically expands dims

        self.set['label'] = label_onehot

        mask = np.zeros((len(times), self.num_node, 3))
        mask[label_raw == 0, 0] = 1
        mask[label_raw == 1, 2] = 1
        self.set['mask'] = mask

    def save_data(self, dataset_path):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
        for record in tqdm(self.dataset, desc="Saving dataset"):
            file_name = os.path.join(dataset_path, f"{record['name']}.pkl")
            rec_copy = record.copy()
            del rec_copy['name']
            with open(file_name, 'wb') as f:
                pickle.dump(rec_copy, f)
        logger.info(f"Saved {len(self.dataset)} records to {dataset_path}")

    def read_data(self, dataset_path):
        if not os.path.exists(dataset_path):
            logger.warning(f"No dataset found in {dataset_path}")
            return
        files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".pkl")], key=lambda x: int(x.split(".")[0]))
        self.dataset = []
        for f in tqdm(files, desc="Loading dataset"):
            with open(os.path.join(dataset_path, f), 'rb') as file:
                record = pickle.load(file)
                record['name'] = f.split(".")[0]
                self.dataset.append(record)
        logger.info(f"Loaded {len(self.dataset)} records from {dataset_path}")

    def _transform(self):
        data_list = []
        metric = self.set['metric']# (N,417)
        self.metric_len = self.set['metric'].shape[-1]
        log = self.set['log']# dict of N-1, of time steps, each of size (1,256)
        trace = self.set['trace']# (N,1,1,17)
        label = self.set['label']# (N,1,2)
        mask = self.set['mask']# (N,1,3)

        starttime, endtime = metric.index.min(), metric.index.max()
        num = 0
        count1, count2, count3 = 0, 0, 0
        data_list = []
            
        while starttime + (self.window - 1) * self.step <= endtime:
            # ---- MEMORY CHECK ----
            if not self.safe_memory_available(0.3):
                print("\n❌ MEMORY LOW — stopping window generation to avoid freeze.\n"
                    "   Saved partial data_list and exiting gracefully.")
                break
            record = {}
            if num % 500 == 0:
                logging.info(f"Processing window {num} starting at time {starttime}...")
            
            #metric
            select_metric = metric[
                (metric.index >= starttime) &
                (metric.index <= starttime + (self.window - 1) * self.step)
            ]
            # do NOT copy unnecessarily
            mvals = select_metric.values
            record['data_node'] = mvals.reshape(self.window, self.num_node, self.metric_len)

            # log
            start = int(starttime)# 1. Define the timestamps needed
            end   = int(starttime + (self.window - 1) * self.step + 1)
            needed_times = range(start, end, self.step)

            log_record = {}# 2. Pre-fill missing timestamps
            zero_frame = np.zeros((self.num_node, self.log_len))
            for t in needed_times:
                log_record.setdefault(t, zero_frame)
            
            log_record_array = np.stack([log_record[t] for t in needed_times], axis=0)# 3. Stack safely and assign
            record['data_log'] = np.nan_to_num(log_record_array)
            #assert log_record.shape == (self.window, len(MSDS_pod), self.log_len), f"Worng log"

            #label
            select_label = label[num + self.window - 1, :]
            select_mask = mask[num + self.window - 1, :]

            record['groundtruth_cls'] = select_mask
            record['groundtruth_real'] = select_label
            count1 += 1 if record['groundtruth_cls'].sum(axis=0)[1] > 0 else 0
            count2 += 1 if record['groundtruth_real'].sum(axis=0)[1] > 0 else 0
            #assert record['groundtruth_cls'].shape == (len(MSDS_pod), 3), f"Worng label"    

            # trace
            select_trace = trace[num : num + self.window]
            count3 += 1 if select_trace.sum() > 0 else 0
            #assert select_trace.shape == (self.window, len(MSDS_pod), len(MSDS_pod), len(self.trace_type)), f"Worng Trace"
            record['data_edge'] = select_trace
            record['name'] = f'{num}'
            num += 1
            data_list.append(record)
            starttime += self.step
            del record
        logging.info(f"deal ...{num}...error see:{count1}...error real:{count2}...")
        return data_list

        logger.info(f"Created {len(data_list)} sliding windows")
        return data_list
