import torch
from torch.utils.data import DataLoader


import yaml
from argparse import ArgumentParser
from adversarials import Attacker
from utils import *

def main(args):
    
    with open(args.config, 'r') as f:
        configs = yaml.load(f, yaml.FullLoader)

    # prepare required stuff
    model, transform = prepare_model(configs['model'])
    model.name = configs['model']
    print(f'using model: {model.name}')
    dataset = build_dataset(configs['root'], transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=configs['workers'])
    criterion = prepare_loss(configs['criterion'])

    if configs['method'] == 'FGSM':
        attacker = Attacker(device=configs['device'], loader=loader, 
                        model=model, criterion=criterion, epsilons=configs['eps'], 
                        save_samples=configs['save_samples'], img_path=configs['save_sample_root'], process_perturbed=configs['process_perturbed'])
        result = attacker.find_adversarial()
        plot_result(configs['save_fig_root'], 'FGSM', result, configs['eps'], model.name)
    elif configs['method'] == 'iter':
        attacker = Attacker(device=configs['device'], loader=loader, 
                        model=model, criterion=criterion, epsilons=configs['eps'], 
                        save_samples=configs['save_samples'], img_path=configs['save_sample_root'], 
                        iterative=True, process_perturbed=configs['process_perturbed'])
        result = attacker.find_adversarial()
        plot_result(configs['save_fig_root'], 'iter', result, configs['eps'], model.name)
    elif configs['method'] == 'least':
        attacker = Attacker(device=configs['device'], loader=loader, 
                        model=model, criterion=criterion, epsilons=configs['eps'], 
                        save_samples=configs['save_samples'], img_path=configs['save_sample_root'], 
                        iterative=True, least=True, process_perturbed=configs['process_perturbed'])
        result = attacker.find_adversarial()
        plot_result(configs['save_fig_root'], 'least', result, configs['eps'], model.name)
    elif configs['method'] == 'all':
        results = []
        top5_results = []
        attacker = Attacker(device=configs['device'], loader=loader, 
                        model=model, criterion=criterion, epsilons=configs['eps'], 
                        save_samples=configs['save_samples'], img_path=configs['save_sample_root'], process_perturbed=configs['process_perturbed'])
        top, top5 = attacker.find_adversarial()
        results.append(top)
        top5_results.append(top5)
        attacker = Attacker(device=configs['device'], loader=loader, 
                        model=model, criterion=criterion, epsilons=configs['eps'], 
                        save_samples=configs['save_samples'], img_path=configs['save_sample_root'], 
                        iterative=True, process_perturbed=configs['process_perturbed'])
        top, top5 = attacker.find_adversarial()
        results.append(top)
        top5_results.append(top5)
        attacker = Attacker(device=configs['device'], loader=loader, 
                        model=model, criterion=criterion, epsilons=configs['eps'], 
                        save_samples=configs['save_samples'], img_path=configs['save_sample_root'], 
                        iterative=True, least=True, process_perturbed=configs['process_perturbed'])
        top, top5 = attacker.find_adversarial()
        results.append(top)
        top5_results.append(top5)
        print(results)
        print(top5_results)
        plot_result(configs['save_fig_root'], 'all', (results, top5_results), configs['eps'], model.name)
    else:
        print('No such method being implemented')
        raise NotImplementedError



def parsing():

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/resnet18.yaml')
    
    return parser.parse_args()


if __name__ == '__main__':
    main(parsing())