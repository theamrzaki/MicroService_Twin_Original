import os
# Restrict PyTorch to a single thread execution profile to minimize internal memory overhead
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
print("00000 tessting")

import util.util as util

print("11111 tessting")
import util.train as train

import util.data_MSDS as data_loads
import util.data_RE2 as data_loads_RE2
import util.data_Eadro as data_Eadro
import util.data_art as data_ART
print("3333 tessting")
import src.model as model
print("444444 tessting")
from torch.utils.data import DataLoader
print("555555 tessting")
import warnings
import logging
import os
import sys
import torch 
torch.set_num_threads(1)
import argparse
print("66666 tessting")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('/code')
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MutliModel Time-Series Anomaly Detection')
    parser.add_argument("--data_source", default="MSDS", type=str,
                        help='the data source: MSDS or TT or SN')
    
    if parser.parse_known_args()[0].data_source == "MSDS":
        print("Using MSDS parser")
        from util.parser_MSDS_MSDS import *
    elif parser.parse_known_args()[0].data_source == "SN":
        print("Using SN parser")
        from util.parser_SN import *
    elif parser.parse_known_args()[0].data_source == "TT":
        print("Using TT parser")
        from util.parser_TT import *
    elif parser.parse_known_args()[0].data_source == "ART":
        print("Using ART parser")
        from util.parser_ART import *
        
    util.seed_everything(args['random_seed'])
    if args['evaluate']:
        dict_json = util.read_params(args)
        for key in dict_json.keys():
            args[key] =  args[key] if key in ['model_path','evaluate', 'result_dir', 'data_path', 'dataset_path'] else dict_json[key]
        args['result_dir'] = args['model_path']
    else:
        args['hash_id'], args['result_dir'] = util.dump_params(args)
        util.json_pretty_dump(args, os.path.join(args['result_dir'], "params.json"))
        args['model_path'] = args['result_dir']

    logging.info("---- Model: ----" + args['main_model'] +"-" + args['hash_id'] 
        + "----" + f"random_seed : {args['random_seed']}"\
        + "----" + f"req_loss_approach : {args['req_loss_approach']}"\
        + "----" + f"rec_lambda : {args['rec_lambda']}"\
        + "----" + f"auxi_lambda : {args['auxi_lambda']}"\
        + "----" + f"Freq Domain : {args['FREQ_DOMAIN']}"\
        + "----" + f"Multi Fits : {args['MULTI_FITS']}"\
        + "----" + f"train : {not args['evaluate']}"\
        + "----" + f"evaluate : {args['evaluate']}")

    # dealing & loading data
    if args["data_source"] == "MSDS":
        processed = data_loads.Process(**args)
    elif args["data_source"] == "RQ2_OB":
        processed = data_loads_RE2.Process(**args)
    elif args["data_source"] == "SN" or args["data_source"] == "TT":
        print("data source is ", args["data_source"])
        # on Raspberry Pi, only load testing data, no training data
        #processed_train,  processed_test = data_Eadro.run(args["data_source"])
        processed_train = []
        processed_test = data_Eadro.run(args["data_source"])
    elif args["data_source"] == "ART":
        processed = data_ART.Process(**args)
    #train_dl = DataLoader(processed.dataset[:int(len(processed.dataset)*0.7)],
    #                      batch_size=args['batch_size'],
    #                      shuffle=True, pin_memory=False, drop_last=True)
    # Calculate full train set
    if args["data_source"] not in ["SN", "TT"]:
        full_train_data = processed.dataset[:int(len(processed.dataset) * 0.7)]
    else:
        full_train_data = processed_train

    # Few-shot percentage (default = 1.0 → use full data)
    fewshot_ratio = args.get("fewshot_ratio", 1.0)  # e.g., 0.1 for 10%
    logging.info(f"Fewshot: training with {fewshot_ratio} of the 70% of the data")
    fewshot_size = int(len(full_train_data) * fewshot_ratio)

    # Create few-shot subset
    fewshot_indices = list(range(fewshot_size))  # or use random.sample(...) for random sampling
    fewshot_dataset = torch.utils.data.Subset(full_train_data, fewshot_indices)

    # here for Raspberry Pi, only load testing only, no training
    train_dl = None #DataLoader(fewshot_dataset,
                    #         batch_size=args['batch_size'],
                    #        shuffle=True, pin_memory=False, drop_last=True)
    test_dl = DataLoader(processed.dataset if args["data_source"] not in ["SN", "TT"] else processed_test,
                        batch_size=args['batch_size'],
                        num_workers=0,  # as to avoid async randomness
                        shuffle=False, pin_memory=False, drop_last=False if args["case_study"] else True)
    
    
    # declear model and train
    if args["data_source"] not in ["SN", "TT"]:
        graph = processed.graph
    else:
        graph = processed_test.first_graph
    models = model.MyModel(graph, **args)
    total_params = util.count_parameters(models)
    print(total_params)
    logging.info(f"Trainable Parameters: {total_params['total_params']}, reconstruction: {total_params['reconstruction_params']}, common: {total_params['common_params']}")
    sys = train.MY(models, **args)  

    #Training
    avg_training_time_per_epoch = 0
    if not args['evaluate']:
        avg_training_time_per_epoch = sys.fit(train_loader=train_dl, test_loader=test_dl)


    # Evaluating
    logging.info('calculate scores...')
    with open('./result.log', 'a+') as file:
        file.writelines(f"\n {args['main_model']}-{args['hash_id']} --weight_decay:{args['weight_decay']}   --learning_change:{args['learning_change']} \n")
        info_dict = {}
        for statue in ['loss', 'f1']:
            logging.info(f'calculate label with {statue}...')
            sys.load_model(args['model_path'], name=statue)
            info, performance,benchmark_result = sys.evaluate(test_dl, isFinall=True)
            info_dict[statue] = info
            if info is not None:
                logging.info(f"Performance with {statue}: {info}")
                file.writelines(statue + '   ' + info + '\n')
            else:
                logging.info(f"Performance with {statue}: No GPU metrics available.")
                file.writelines(statue + '   ' + ",,," + '\n')
    #if args.get("case_study", False):
    exp_name = ""
    if args["experiment_name"]=="RQ1_main":
        exp_name = "RQ1_benchmark_memOptimized_Larger"
    elif args["experiment_name"] == "RQ1_main_Edge":
        if not util.is_raspberry_pi():
            raise RuntimeError(
                "RQ1_main_Edge can only be run on a Raspberry Pi."
            )
        else:
            print("Running on Raspberry Pi, proceeding with RQ1_main_Edge experiment.")
        exp_name = "RQ1_main_Edge"
    elif args["experiment_name"]=="RQ2_ablations_components":
        exp_name = "RQ2_ablations_withNorm_accMem"
    elif args["experiment_name"]=="RQ2_basis_comparison":
        exp_name = "RQ2_basis_withNorm_accMem"
    elif args["experiment_name"]=="RQ3_case_study":
        exp_name = "RQ3_case_study"
    #results_path = f'./output/result_msds_{exp_name}.csv'#msds
    #if args["data_source"] == "TT":
    #    results_path = f'./output/result_TT_{exp_name}.csv'
    #elif args["data_source"] == "SN":
    #    results_path = f'./output/result_SN_{exp_name}.csv'
    #elif args["data_source"] == "ART":
    #    results_path = f'./output/result_ART_{exp_name}.csv'
    results_path = f'./result_journal/result_{exp_name}.csv'
    util.write_results(args,info_dict,total_params,avg_training_time_per_epoch,performance,benchmark_result,results_path)
    #else:
    #    util.write_results(args,info_dict,total_params,avg_training_time_per_epoch,performance,'./result_casestudy.csv')
    #logging.info("^^^^^^ Current Model: ----" + args['main_model'] + "-" * 4 + args['hash_id'] + " ^^^^^")

    # For case study
    if args.get("case_study", False):
        logging.info("Collecting case-study samples...")
        case_path = os.path.join( "case_ids.json")
        #include num of epochs in the titl
        case_output = os.path.join(f"case_output_{args['FREQ_DOMAIN']}_{args['epochs']}.json")

        #if model = encoder-decoder type, primary = True
        if args['FREQ_DOMAIN'] in ['encoder-decoder','Eadro','Art']:
            primary = True
            print("####---> Primary case study collection for encoder-decoder model.")
            case_data = sys.collect_case_study(
                test_dl,
                primary=primary,
                case_json=case_path,
                top_k=10,
                dataset_path=args['dataset_path']
            )
        else:
            primary = False
            print("@@@@---> Secondary case study collection for other model types.")
            case_data = sys.collect_case_study(
                test_dl,
                primary=False,
                case_json=case_path,
                dataset_path=args['dataset_path']
            )
        util.json_pretty_dump(case_data, case_output)