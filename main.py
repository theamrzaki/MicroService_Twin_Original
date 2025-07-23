import util.util as util
import util.train as train
import util.data_MSDS as data_loads
from util.parser_MSDS import *
import src.model as model
from torch.utils.data import DataLoader
import warnings
import logging
import os
import sys
import torch 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('/code')
warnings.filterwarnings("ignore")

util.seed_everything(args['random_seed'])

if __name__ == '__main__':
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
    processed = data_loads.Process(**args)
    #train_dl = DataLoader(processed.dataset[:int(len(processed.dataset)*0.7)],
    #                      batch_size=args['batch_size'],
    #                      shuffle=True, pin_memory=False, drop_last=True)
    # Calculate full train set
    full_train_data = processed.dataset[:int(len(processed.dataset) * 0.7)]

    # Few-shot percentage (default = 1.0 → use full data)
    fewshot_ratio = args.get("fewshot_ratio", 1.0)  # e.g., 0.1 for 10%
    logging.info(f"Fewshot: training with {fewshot_ratio} of the 70% of the data")
    fewshot_size = int(len(full_train_data) * fewshot_ratio)

    # Create few-shot subset
    fewshot_indices = list(range(fewshot_size))  # or use random.sample(...) for random sampling
    fewshot_dataset = torch.utils.data.Subset(full_train_data, fewshot_indices)

    train_dl = DataLoader(fewshot_dataset,
                        batch_size=args['batch_size'],
                        shuffle=True, pin_memory=False, drop_last=True)
    test_dl = DataLoader(processed.dataset[int(len(processed.dataset)*0.7):],
                        batch_size=args['batch_size'],
                        shuffle=False, pin_memory=False, drop_last=True)
    # declear model and train
    models = model.MyModel(processed.graph, **args)
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
            info, performance = sys.evaluate(test_dl, isFinall=True)
            info_dict[statue] = info
            file.writelines(statue + '   ' + info + '\n')
    util.write_results(args,info_dict,total_params,avg_training_time_per_epoch,performance,'./result.csv')
    logging.info("^^^^^^ Current Model: ----" + args['main_model'] + "-" * 4 + args['hash_id'] + " ^^^^^")

