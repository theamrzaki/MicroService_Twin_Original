import hashlib
import json
import logging
import os
import time
import pickle
import random
import numpy as np
import torch
from sklearn.metrics import *
from util.constant import *
from pathlib import Path

def calc_index(predict, actual):
    """
    calculate f1 score by predict and actual.
    """

    # Ensure predict and actual are 2D tensors
    if predict.dim() != 2:
        predict = predict.reshape(-1, predict.shape[-1])
    if actual.dim() != 2:
        actual = actual.reshape(-1, actual.shape[-1])

    # Detach from graph and move to CPU before converting to numpy
    predict_np = predict.detach().cpu().numpy()
    actual_np = actual.detach().cpu().numpy()

    ap = average_precision_score(actual_np, predict_np, average='macro')#.tolist()
    auc = roc_auc_score(actual_np, predict_np, average='macro')#.tolist()

    if predict.shape[-1] == 2 and actual.shape[-1] == 2:
        actual_cls = torch.argmax(actual, dim=-1)
        predict_cls = torch.argmax(predict, dim=-1)
    else:
        actual_cls = actual
        predict_cls = predict

    # Convert classification tensors to numpy as well
    actual_cls_np = actual_cls.detach().cpu().numpy()
    predict_cls_np = predict_cls.detach().cpu().numpy()

    ps = precision_score(actual_cls_np, predict_cls_np, average="binary")#.tolist()
    rs = recall_score(actual_cls_np, predict_cls_np, average="binary")#.tolist()
    effection = f1_score(actual_cls_np, predict_cls_np, average="binary", zero_division=1)#.tolist()

    pred = np.bincount(predict_cls_np)
    actu = np.bincount(actual_cls_np)

    if pred.shape[0] == 1:
        information = f'pr:{ps:.4f}  rc:{rs:.4f}  auc:{auc:.4f} ap:{ap:.4f} f1: {effection:.4f} pred_right: {pred[0]} pred_wrong: 0  actu_right: {actu[0]} actu_wrong: {actu[1]}'
    else:
        information = f'pr:{ps:.4f}  rc:{rs:.4f}  auc:{auc:.4f} ap:{ap:.4f} f1: {effection:.4f} pred_right: {pred[0]} pred_wrong:{pred[1]} actu_right: {actu[0]} actu_wrong: {actu[1]}'
    logging.info(information)
    return information, {'pr': ps, 'rc': rs, 'auc': auc, 'ap': ap, 'f1': effection}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def json_pretty_dump(obj, filename):
    
    with open(filename, "w") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4,
                  separators=(",", ": "), ensure_ascii=False, cls=NumpyEncoder)


def dump_params(args):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")).hexdigest()[0:8]
    if args['experiment_name'] == "RQ1_main":
        save_path = os.path.join(args['result_dir'], args['FREQ_DOMAIN'] + '-' +args['data_source'] + '-Epochs' + str(args['epochs']) + '-Seed' + str(args['random_seed']) + 'ExpName' + str(args["experiment_name"]) + '-lowshow-simplegraph')
    
    
    elif args['experiment_name'] == "RQ2_basis_comparison":
        save_path = os.path.join(args['result_dir']+"/server_only_exps/Basis_updatedFreDF", args['FREQ_DOMAIN'] + '-' +args['data_source'] + '-Epochs' + str(args['epochs']) + '-Seed' + str(args['random_seed']) + '-ExpName' + str(args["experiment_name"])) + '-BasisType' + str(args["basis_type"])+ '-lowshow-simplegraph'
    
    
    elif args['experiment_name'] == "RQ2_ablations_components":
        save_path = os.path.join(args['result_dir']+"/server_only_exps/Ablations_updatedFreDF", args['FREQ_DOMAIN'] + '-' +args['data_source'] + '-Epochs' + str(args['epochs']) + '-Seed' + str(args['random_seed']) + '-ExpName' + str(args["experiment_name"]))  + '-LPF' + str(args["filter_used"]) + '-Attn' + str(args["modules_attn"]) + '-NormLin' + str(args["use_normlin"]) + '-RecLambda' + str(args["rec_lambda"]) + '-AuxiLambda' + str(args["auxi_lambda"])   + '-lowshow-simplegraph'
    
    
    elif args['experiment_name'] == "RQ2_architecture":
        save_path = os.path.join(args['result_dir']+"/server_only_exps/Architecture_updatedFreDF", args['FREQ_DOMAIN'] + '-' +args['data_source'] + '-Epochs' + str(args['epochs']) + '-Seed' + str(args['random_seed']) + '-ExpName' + str(args["experiment_name"]))  + '-LPF' + str(args["filter_used"]) + '-Attn' + str(args["modules_attn"]) + '-NormLin' + str(args["use_normlin"]) + '-RecLambda' + str(args["rec_lambda"]) + '-AuxiLambda' + str(args["auxi_lambda"])+ '-lowshow-simplegraph'

    elif args['experiment_name'] == "RQ2z5_basis_degree_comparison":
        #same as RQ2_ablations_components (except adding degree and basis type)
        save_path = os.path.join(args['result_dir']+"/server_only_exps/Ablations_updatedFreDF", args['FREQ_DOMAIN'] + '-' +args['data_source'] + '-Epochs' + str(args['epochs']) + '-Seed' + str(args['random_seed']) + '-ExpName' + str(args["experiment_name"]))  + '-LPF' + str(args["filter_used"]) + '-Attn' + str(args["modules_attn"]) + '-NormLin' + str(args["use_normlin"]) + '-RecLambda' + str(args["rec_lambda"]) + '-AuxiLambda' + str(args["auxi_lambda"])   + '-lowshow-simplegraph' + '-Degree' + str(args["degree"]) + '-BasisType' + str(args["basis_type"])

    elif args['experiment_name'] == "RQ3_sensitivity":
        save_path = os.path.join(args['result_dir']+"/server_only_exps/Sensitivity", args['FREQ_DOMAIN'] + '-' +args['data_source'] + '-Epochs' + str(args['epochs']) + '-Seed' + str(args['random_seed']) + '-ExpName' + str(args["experiment_name"]))  + '-LPF' + str(args["filter_used"]) + '-Attn' + str(args["modules_attn"]) + '-NormLin' + str(args["use_normlin"]) + '-RecLambda' + str(args["rec_lambda"]) + '-AuxiLambda' + str(args["auxi_lambda"])  + '-Degree' + str(args["degree"]) + '-lowshow-simplegraph'
    
    elif args['experiment_name'] == "RQ3_sensitivity_auxi_lambda":
        #same as RQ3_sensitivity 
        save_path = os.path.join(args['result_dir']+"/server_only_exps/Sensitivity", args['FREQ_DOMAIN'] + '-' +args['data_source'] + '-Epochs' + str(args['epochs']) + '-Seed' + str(args['random_seed']) + '-ExpName' + str(args["experiment_name"]))  + '-LPF' + str(args["filter_used"]) + '-Attn' + str(args["modules_attn"]) + '-NormLin' + str(args["use_normlin"]) + '-RecLambda' + str(args["rec_lambda"]) + '-AuxiLambda' + str(args["auxi_lambda"])  + '-Degree' + str(args["degree"]) + '-lowshow-simplegraph'

    #if args["raspberry_pi_smaller_model"] == 'true':
    #    save_path = save_path + '-RaspberryPiSmallerModel'

    elif args['experiment_name'] == "RQ3_case_study":
        save_path = os.path.join(args['result_dir']+"/server_only_exps/CaseStudy", args['FREQ_DOMAIN'] + '-' +args['data_source'] + '-Epochs' + str(args['epochs']) + '-Seed' + str(args['random_seed']) + '-ExpName' + str(args["experiment_name"]))  + '-LPF' + str(args["filter_used"]) + '-Attn' + str(args["modules_attn"]) + '-NormLin' + str(args["use_normlin"]) + '-RecLambda' + str(args["rec_lambda"]) + '-AuxiLambda' + str(args["auxi_lambda"])  + '-Degree' + str(args["degree"]) + '-lowshow-simplegraph'
    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, "running.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return hash_id, save_path


def read_params(args):
    filename = os.path.join(args['model_path'], "params.json")
    with open(filename) as f:
        dict_json = json.load(fp=f)
   
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    return dict_json


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model, verbose=True):
    reconstruction_params = 0
    common_params = 0
    recon_lines = []
    common_lines = []
    found_cutoff = False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_count = param.numel()
        shape_str = str(tuple(param.shape))

        if not found_cutoff:
            if name == "show.0.weight":
                found_cutoff = True
        if not found_cutoff:
            reconstruction_params += param_count
            if verbose:
                recon_lines.append(f"{name:40} | shape: {shape_str:25} | params: {param_count:,}")
        else:
            common_params += param_count
            if verbose:
                common_lines.append(f"{name:40} | shape: {shape_str:25} | params: {param_count:,}")

    total_params = reconstruction_params + common_params

    if verbose:
        logging.info("Trainable Parameters Breakdown:")
        logging.info("-" * 80)
        logging.info("Reconstruction-Specific Parameters:")
        for line in recon_lines:
            logging.info(line)
        logging.info(f"Subtotal: {reconstruction_params:,}")
        logging.info("-" * 80)
        logging.info("Common Parameters:")
        for line in common_lines:
            logging.info(line)
        logging.info(f"Subtotal: {common_params:,}")
        logging.info("-" * 80)
        logging.info(f"Total Trainable Parameters: {total_params:,}\n")

    return {
        "total_params": total_params,
        "reconstruction_params": reconstruction_params,
        "common_params": common_params
    }

def write_results(args, info_dict,total_params,avg_training_time_per_epoch,performance,benchmark_result, file_name='result.csv'):
    file_path = file_name
    #infodict = {'pr':ps, 'rc':rs, 'auc':auc, 'ap':ap, 'f1':effection}
    """
            performance = {
            "GPU": {
                    "inference_time_per_sample_ms": gpu_metrics["inference_time_per_sample_ms"],
                    "throughput_samples_per_sec": gpu_metrics["throughput_samples_per_sec"],
                    "peak_memory_mb": gpu_metrics["peak_memory_mb"],
                    "energy_per_sample_joules": gpu_metrics["energy_per_sample_joules"]
            },
            "CPU": {
                    "inference_time_per_sample_ms": cpu_metrics["inference_time_per_sample_ms"],
                    "throughput_samples_per_sec": cpu_metrics["throughput_samples_per_sec"],
                    "peak_memory_mb": cpu_metrics["peak_memory_mb"],
                    "energy_per_sample_joules": cpu_metrics["energy_per_sample_joules"]
            }
        }
    """
    row = {
        
        'model': args['main_model'],
        'hash_id': args['hash_id'],
        'datasource': args['data_source'],

        'FREQ_DOMAIN': args['FREQ_DOMAIN'],
        'basis_type': args['basis_type'],
        'MULTI_FITS': args['MULTI_FITS'],
        'req_loss_approach': args['req_loss_approach'],
        'rec_lambda': args['rec_lambda'],
        'auxi_lambda': args['auxi_lambda'],
        'filter_used': args['filter_used'],
        'modules_attn':  args['modules_attn'],

        'info_dict': info_dict['f1'],

        'total_params': total_params['total_params'],
        'reconstruction_params': total_params['reconstruction_params'],
        'common_params': total_params['common_params'],

        'training_time_per_epoch': avg_training_time_per_epoch,
        'inference_time_per_sample_ms (GPU)': performance['GPU']['inference_time_per_sample_ms'] if 'GPU' in performance else None,
        'peak_memory_mb (GPU)': performance['GPU']['peak_memory_mb'] if 'GPU' in performance else None,
        'inference_time_per_sample_ms (CPU)': performance['CPU']['inference_time_per_sample_ms'] if 'CPU' in performance else None,
        'peak_memory_mb (CPU)': performance['CPU']['peak_memory_mb'] if 'CPU' in performance else None,
        'throughput_samples_per_sec (GPU)': performance['GPU']['throughput_samples_per_sec'] if 'GPU' in performance else None,
        'throughput_samples_per_sec (CPU)': performance['CPU']['throughput_samples_per_sec'] if 'CPU' in performance else None,
        'energy_per_sample_joules (GPU)': performance['GPU']['energy_per_sample_joules'] if 'GPU' in performance else None,
        'energy_per_sample_joules (CPU)': performance['CPU']['energy_per_sample_joules'] if 'CPU' in performance else None,

        '(benchmark_result) parameters_million': benchmark_result["parameters_million"],
        '(benchmark_result) flops_million': benchmark_result["flops_million"],
        '(benchmark_result) inference_time_ms': benchmark_result["inference_time_ms"],
        '(benchmark_result) inference_memory_mb': benchmark_result["inference_memory_mb"],

        'fewshot_ratio': args['fewshot_ratio'],
        'experiment_name': args['experiment_name'],
        'random_seed': args['random_seed'],
        'use_normlin': args['use_normlin'],

        'degree': args['degree'],
        'linear_attn_dim': args['linear_attn_dim'],
    }
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(','.join(row.keys()) + '\n')
    with open(file_path, 'a') as f:
        f.write(','.join([str(value) for value in row.values()]) + '\n')



def dump_pickle(obj, file_path):
    logging.info("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    logging.info("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)


def is_raspberry_pi():
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        return "Raspberry Pi" in model_file.read_text(errors="ignore")
    return False

