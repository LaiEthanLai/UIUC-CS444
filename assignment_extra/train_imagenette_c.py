import torch.optim as optim
from torch.utils.data import DataLoader

import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from utils import *

def main(args):
    
    with open(args.config, 'r') as f:
        configs = yaml.load(f, yaml.FullLoader)

    # prepare required stuff
    model, transform = prepare_model(configs['model'], from_scratch=configs['scratch'])
    train_dataset = build_dataset(configs['root_train'], transform)
    val_dataset = build_dataset(configs['root_val'], transform)
    loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=configs['workers'])
    val_loader = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=configs['workers'])
    criterion = prepare_loss(configs['criterion'])
    device = configs['device']
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=configs['lr'])
    t0 = configs['epoch'] if not configs['scratch'] else configs['epoch'] // 3
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, t0, T_mult=1, eta_min=0)

    for epoch in range(configs['epoch']):
        with tqdm(loader) as t:
            correct = 0.0
            total = 0.0
            acc = 0.0
            model.train()
            for (img, label) in t:
                
                optimizer.zero_grad()
                output = model(img.to(device))
                loss = criterion(output, label.to(device))
                loss.backward()

                total += img.shape[0]
                correct += sum(output.detach().cpu().argmax(1) == label)

                optimizer.step()
                scheduler.step()

        acc = correct / total
        print(f'{epoch+1} / {configs["epoch"]}, loss: {loss.item()}, train_acc: {acc}')

        with tqdm(val_loader) as t_val:
            val_correct = 0.0
            val_total = 0.0
            val_acc = 0.0
            for (img, label) in t_val:
                model.eval()
                with torch.no_grad():
                    output = model(img.to(device))
                    loss = criterion(output, label.to(device))
                val_total += img.shape[0]
                val_correct += sum(output.detach().cpu().argmax(1) == label)

        val_acc = val_correct / val_total
        print(f'loss: {loss.item()}, val_acc: {val_acc}')



def parsing():

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/imagnette_c.yaml')
    
    return parser.parse_args()


if __name__ == '__main__':
    main(parsing())