import json
import logging
import os
import time
import copy
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from adabelief_pytorch import AdaBelief
import psutil

from tqdm import tqdm
import util.util as util

class Base(nn.Module):
    def __init__(self, model, **args):
        super(Base, self).__init__()

        self.model = model
        self.use_gpu = args['gpu']
        # Training
        self.epoches = args['epochs']
        self.learning_rate = args['learning_rate']
        self.weight_decay = args['weight_decay']
        self.patience = args['patience']  # > 0: use early stop
        self.model_save_dir = args['result_dir']
        self.learning_change = args['learning_change']
        self.learning_gamma = args['learning_gamma']
        self.rec_down = args['rec_down']
        self.para_low = args['para_low']
        self.True_list = {'normal': 1, 'abnormal': args['abnormal_weight']}

        if args['evaluate']:
            self.load_model(args['model_path'])
        else:
            logging.info('model : init weight')
            self.init_weight()

        if args['gpu'] and torch.cuda.is_available():
            logging.info("Using GPU...")
            torch.cuda.empty_cache()
            self.model.cuda()
        else:
            logging.info("Using CPU...")

    # Model init
    def init_weight(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    #  Put Data into GPU/CPU
    def input2device(self, batch_input, use_gpu):
        if isinstance(batch_input, dict):
            if use_gpu:
                for name, data in batch_input.items():
                    if torch.any(torch.isnan(data)):
                        data = torch.where(torch.isnan(data), torch.full_like(data, 0), data)
                    batch_input[name] = torch.tensor(data, dtype=torch.float32, requires_grad=True).cuda()
            else:
                for name, data in batch_input.items():
                    if torch.any(torch.isnan(data)):
                        data = torch.where(torch.isnan(data), torch.full_like(data, 0), data)
                    batch_input[name] = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        else:
            if use_gpu:
                if torch.any(torch.isnan(batch_input)):
                        data = torch.where(torch.isnan(batch_input), torch.full_like(batch_input, 0), batch_input)
                batch_input = torch.tensor(batch_input, dtype=torch.float32, requires_grad=True).cuda()
            else:
                if torch.any(torch.isnan(batch_input)):
                        data = torch.where(torch.isnan(batch_input), torch.full_like(batch_input, 0), batch_input)
                batch_input = torch.tensor(batch_input, dtype=torch.float32, requires_grad=True)
        return batch_input

    # Loading modal paras
    def load_model(self, model_save_file="", name='loss'):
        if model_save_file == ' ':
            logging.info(f'No {self.model.name} statue file')
        else:
            logging.info(f'{self.model.name} on {model_save_file} loading...')
            self.model.load_state_dict(torch.load(
                    os.path.join(model_save_file, f"{self.model.name}_{name}_stage.ckpt")))

    # Saving modal paras
    def save_model(self, best_dict, model_save_dir="", name='loss'):
        file_status = os.path.join(model_save_dir, f"{self.model.name}_{name}_stage.ckpt")
        if best_dict['state'] is None:
            logging.info(f'No {self.model.name} - {name} statue file')
        else: 
            logging.info(f'{self.model.name} - {name}  best score:{best_dict["score"]} at epoch {best_dict["epoch"]}')
            torch.save(best_dict['state'], file_status)

class MY(Base):
    def __init__(self, model, **args):
        super().__init__(model, **args)

    def fit(self, train_loader, test_loader, **args):
        optimizer = AdaBelief(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.learning_change, self.learning_gamma)

        best = {"loss":{"score": float("inf"), "state": None, "epoch": 0},
                "f1":{"score": 0, "state": None, "epoch": 0}}

        pre_loss, worse_count, isWrong = float("inf"), 0, False

        label_weight = torch.tensor(
            np.array(list(self.True_list.values())), dtype=torch.float).cuda()
        losser = nn.BCEWithLogitsLoss(reduce='mean', weight=label_weight)
        logging.info('optimizer : using AdaBelief')

        training_epoch_time_list = []
        for epoch in range(0, self.epoches):
            lr = optimizer.param_groups[0]['lr']
            para = torch.tensor(1 / (epoch // self.rec_down + 1))
            para = para if para > self.para_low else self.para_low

            logging.info('-' * 100)
            logging.info(
                f'{epoch}/{self.epoches} starting... lr: {lr} para:{para}')
            self.model.train()
            epoch_cls_loss, epoch_rec_loss, epoch_loss = [], [], []
            epoch_time_start = time.time()
            with tqdm(train_loader) as tbar:
                for batch_input in tbar:
                    batch_input = self.input2device(batch_input, self.use_gpu)
                    optimizer.zero_grad()
                    raw_loss, cls_result, cls_label = self.model(batch_input)

                    rec_loss = sum(raw_loss)
                    if cls_result.shape[0] == 0:
                        cls_loss = torch.tensor(0, dtype=torch.float).cuda()
                    else:
                        cls_loss = losser(cls_result, cls_label)

                    loss = (1 - para) * cls_loss + para * rec_loss

                    if torch.isnan(loss):
                        isWrong = True
                        logging.info(f"loss is nan")
                        break

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                    optimizer.step()

                    epoch_cls_loss.append(cls_loss.item())
                    epoch_rec_loss.append(rec_loss.item())
                    epoch_loss.append(loss.item())
                    tbar.set_postfix(
                        loss=f'{loss.item():.8f},{cls_loss.item():.8f},{rec_loss.item():.8f}')

            # show the result about this epoch
            epoch_time_elapsed = time.time() - epoch_time_start
            training_epoch_time_list.append(epoch_time_elapsed)
            epoch_loss = torch.mean(torch.tensor(epoch_loss)).item()
            epoch_cls_loss = torch.mean(torch.tensor(epoch_cls_loss)).item()
            epoch_rec_loss = torch.mean(torch.tensor(epoch_rec_loss)).item()

            if isWrong:
                logging.info("calculate error in epoch {}".format(epoch))
                break

            if epoch_loss <= best["loss"]["score"] or epoch == self.rec_down:
                worse_count = 0
                best["loss"]["score"] = epoch_loss
                best["loss"]["state"] = copy.deepcopy(self.model.state_dict())
                best["loss"]["epoch"] = epoch
            elif epoch_loss <= pre_loss:
                pass
            elif epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break

            pre_loss = epoch_loss
            logging.info(
                "Epoch {}/{}, all_loss:{:.5f} cls_loss:{:.5f} rec_loss:{:.5f} [{:.2f}s]; best loss:{:.5f}, patience : {}"
                .format(epoch, self.epoches, epoch_loss, epoch_cls_loss, epoch_rec_loss, epoch_time_elapsed, best["loss"]['score'], worse_count))
            
            if epoch > self.rec_down:
                try:
                    result = self.evaluate(train_loader)
                    self.evaluate(test_loader)
                except:
                    logging.info("Error in evaluation during training")
                    continue
                if float(result['f1']) >= best["f1"]["score"]:
                    best["f1"]["score"] = float(result['f1'])
                    best["f1"]["state"] = copy.deepcopy(self.model.state_dict())
                    best["f1"]["epoch"] = epoch
            scheduler.step()

        logging.info('saving model...')
        self.save_model(best['loss'], self.model_save_dir, name='loss')
        self.save_model(best['f1'], self.model_save_dir, name='f1')
        avg_training_time_per_epoch = np.mean(training_epoch_time_list)
        logging.info(f'Average training time per epoch: {avg_training_time_per_epoch:.2f} seconds')
        return avg_training_time_per_epoch
    def evaluate(self, test_loader, isFinall=False,final_evaluation=False):
        def run_inference(use_gpu_flag):
            self.model.eval()
            predict_list, label_list = [], []

            is_gpu = use_gpu_flag and torch.cuda.is_available()
            device = torch.device("cuda" if is_gpu else "cpu")
            self.model.to(device)  

            if is_gpu:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
            else:
                start_time = time.time()
                process = psutil.Process()

            with torch.no_grad():
                for batch_input in tqdm(test_loader, desc=f"Running on {'GPU' if is_gpu else 'CPU'}"):
                    try:
                        batch_input = self.input2device(batch_input, use_gpu_flag)
                        raw_result, _ = self.model(batch_input, evaluate=True)

                        predict_list.append(raw_result)
                        label_list.append(batch_input['groundtruth_real'])
                    except Exception as e:
                        logging.error(f"Error during inference: {e}")
                        continue

            if is_gpu:
                end_event.record()
                torch.cuda.synchronize()
                inference_time_ms = start_event.elapsed_time(end_event)
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                inference_time_ms = (time.time() - start_time) * 1000
                peak_memory_mb = process.memory_info().rss / (1024 ** 2)

            predict_all = torch.concat(predict_list, dim=0).cpu()
            label_all = torch.concat(label_list, dim=0).cpu()
            info, result = util.calc_index(predict_all, label_all)

            return {
                "info": info,
                "result": result,
                "inference_time_total_ms": inference_time_ms,
                "inference_time_per_sample_ms": inference_time_ms / len(test_loader.dataset),
                "peak_memory_mb": peak_memory_mb,
            }

        # 1. Run on GPU
        gpu_metrics = run_inference(use_gpu_flag=True)

        if isFinall:
            # 2. Run on CPU
            cpu_metrics = run_inference(use_gpu_flag=False)

            # Combine both
            performance = {
                "GPU": {
                    "inference_time_per_sample_ms": gpu_metrics["inference_time_per_sample_ms"],
                    "peak_memory_mb": gpu_metrics["peak_memory_mb"]
                },
                "CPU": {
                    "inference_time_per_sample_ms": cpu_metrics["inference_time_per_sample_ms"],
                    "peak_memory_mb": cpu_metrics["peak_memory_mb"]
                }
            }

            logging.info(f"Performance Summary:\n{performance}")

        # Return only GPU `info` as main output
        if isFinall:
            return gpu_metrics["info"], performance
        else:
            return gpu_metrics["result"]

    
    def collect_case_study(
        self,
        test_loader,
        use_gpu=True,
        primary=False,
        case_json: Optional[str] = None,
        record_json: Optional[str] = None,
        top_k: int = 10,
        store_pred: bool = True
    ):
        """
        primary=True  → define case difficulty and save case IDs
        primary=False → load case IDs and save per-model records

        case_json   : path to save/load case IDs
        record_json : path to save per-model case records (for plotting)
        """

        self.model.eval()
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)

        sample_records = []
        global_idx = 0  # deterministic dataset index

        # ------------------------------------------------------------
        # Full deterministic pass over test set
        # ------------------------------------------------------------
        with torch.no_grad():
            for batch_input in test_loader:
                gt = batch_input["groundtruth_real"]
                batch_size = gt.size(0)

                batch_input = self.input2device(batch_input, use_gpu)
                raw_result, _ = self.model(batch_input, evaluate=True)
                gt = gt.to(raw_result.device)
                error = torch.abs(raw_result - gt)

                for i in range(batch_size):
                    rec = {
                        "id": global_idx + i,
                        "gt": gt[i].cpu().tolist(),              # JSON-safe
                        "error": error[i].mean().item()
                    }

                    if store_pred:
                        rec["pred"] = raw_result[i].cpu().tolist()

                    sample_records.append(rec)

                global_idx += batch_size

        # ------------------------------------------------------------
        # Primary model: define and save case IDs
        # ------------------------------------------------------------
        if primary:
            sorted_cases = sorted(sample_records, key=lambda x: x["error"])

            cases = {
                "easy":   [c["id"] for c in sorted_cases[:top_k]],
                "hard":   [c["id"] for c in sorted_cases[-top_k:]],
                "middle": [c["id"] for c in
                        sorted_cases[len(sorted_cases)//2 - top_k//2 :
                                        len(sorted_cases)//2 + top_k//2]]
            }

            if case_json is not None:
                with open(case_json, "w") as f:
                    json.dump(cases, f, indent=2)

            # Optionally save full primary records
            if record_json is not None:
                with open(record_json, "w") as f:
                    json.dump(sample_records, f, indent=2)

            return {
                "cases": cases,
                "records": sample_records
            }

        # ------------------------------------------------------------
        # Follower models: load fixed cases & save records
        # ------------------------------------------------------------
        else:
            if case_json is None:
                raise ValueError("case_json must be provided when primary=False")

            with open(case_json) as f:
                cases = json.load(f)

            selected_ids = set(sum(cases.values(), []))

            filtered_records = [
                r for r in sample_records if r["id"] in selected_ids
            ]

            if record_json is not None:
                with open(record_json, "w") as f:
                    json.dump(filtered_records, f, indent=2)

            return {
                "cases": cases,
                "records": filtered_records
            }