import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MultiTaskDataset
from model import BTN
from loss import multi_task_loss, multi_task_metric
from config import cfg_dict
from eval import Collector
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import logging
import argparse
from tqdm import tqdm


def load_weights(model, filepath):
    model.load_state_dict(torch.load(filepath, map_location=next(model.parameters()).device))

def save_weights(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_vgg_weights(model, logger, device, filepath=None, layers=None):
    vgg_rep_model = torchvision.models.vgg19(pretrained=True)

    if filepath:
        ckpt=torch.load(filepath, map_location=torch.device(device))
        vgg_rep_model.classifier[6] = torch.nn.Linear(4096, ckpt['classifier.6.weight'].size(0))
        vgg_rep_model.load_state_dict(ckpt)
        logger.info('load vgg weights size %d'%ckpt['classifier.6.weight'].size(0))

    with torch.no_grad():
        model.D[0].weight.copy_(vgg_rep_model.classifier[0].weight)
        if model.D[0].bias is not None or vgg_rep_model.classifier[0].bias is not None:
            model.D[0].bias.copy_(vgg_rep_model.classifier[0].bias)

        if 'fc2' in layers:
            model.D[2].weight.copy_(vgg_rep_model.classifier[3].weight)
            if model.D[2].bias is not None or vgg_rep_model.classifier[3].bias is None:
                model.D[2].bias.copy_(vgg_rep_model.classifier[3].bias)

def get_state(model):
    state = dict(trainable=[], frozen=[])
    for k, v in model.named_parameters():
        if v.requires_grad:
            state['trainable'].append(k)
        else:
            state['frozen'].append(k)
    return state

def freeze_layers(model, frozen_layers, logger):
    state = get_state(model)
    logger.info('before')
    logger.info(state)

    for p in model.parameters():
        p.requires_grad_(True)

    for layer in frozen_layers:
        getattr(model, layer).requires_grad_(False)

    state = get_state(model)
    logger.info('after')
    logger.info(state)

def build_optimizer(model, lr, separate_group, separate_lr, train_scale, logger):
    logger.info('Build optimizer')

    separate_group = set(separate_group)
    model.alpha.requires_grad_(train_scale)

    group1 = [v for k,v in model.named_parameters() if not k in separate_group]
    group2 = [v for k,v in model.named_parameters() if k in separate_group]
    for v in group2:
        v.requires_grad_(True)

    group1keys=[k for k,v in model.named_parameters() if not k in separate_group]
    group2keys=[k for k,v in model.named_parameters() if k in separate_group]

    logger.info('Params %s with lr %f'%(group1keys, lr))
    logger.info('Params %s with lr %f'%(group2keys, separate_lr))

    optimizer=torch.optim.Adam([{'params':group1},{'params':group2,'lr':separate_lr}], lr=lr)

    state = get_state(model)
    logger.info(state)
    return optimizer


class Logger:
    def __init__(self, outdir):
        self._logger = logging.getLogger('train')
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

        os.makedirs(outdir, exist_ok=True)
        logfile = logging.FileHandler(os.path.join(outdir, "log.txt"))
        self._logger.addHandler(logfile)
        self._logger.info("==================== New Run ==================")
        
        self._writer=SummaryWriter(outdir+'/events/')
        
    def add_config(self, config):
        params=config._dump_config(config.outdir)
        for k,v in params.items():
            self._logger.info(k + ': %s', v)

        try:
            from shutil import copyfile
            copyfile(os.path.abspath(__file__), os.path.join(config.outdir, 'main.py'))
            copyfile(os.path.dirname(os.path.abspath(__file__))+'/'+BTN.__module__+'.py', os.path.join(config.outdir, 'model.py'))
            copyfile(os.path.dirname(os.path.abspath(__file__))+'/'+MultiTaskDataset.__module__+'.py', os.path.join(config.outdir, 'dataset.py'))

        except Exception:
            pass

    def info(self, message):
        self._logger.info(message)
        
    def add_scalar(self, tag, scalar, global_step):
        self._writer.add_scalar(tag, scalar, global_step)
        self._writer.flush()
        
    def add_scalars(self, main_tag, scalars, names, excludes, global_step):
        assert len(scalars)==len(names), (len(scalars),len(names))
        tag_scalar_dict={k:v for k,v in zip(names,scalars)}
        self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)
        for k in excludes:
            if k in tag_scalar_dict:
                tag_scalar_dict.pop(k)
        self._writer.add_scalar(main_tag+'avg', np.mean(list(tag_scalar_dict.values())), global_step)
        
    def add_dict(self, main_tag, tag_scalar_dict, global_step=None):
        self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)


def train_epoch(epoch, model, dataloader, optimizer, criterion, metric, multitask, device, logger, cfg):
    model.train()
    loss_history=[]
    accuracy_history=[]
    n_supports=[]
    for inputs, targets in tqdm(dataloader):

        optimizer.zero_grad()
        if multitask:
            inputs = [[x_i.to(device) for x_i in x] for x in inputs]
            targets = [[x_i.to(device) for x_i in x] for x in targets]
            outputs, samples = model.forward_multitask(inputs)
        else:
            inputs = [x.to(device) for x in inputs]
            targets = [x.to(device) for x in targets]
            outputs, samples = model(inputs)

        loss, loss_list = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        accuracy, n_sup = metric(outputs, targets, True)
        loss_history.append([x.item() for x in loss_list])
        accuracy_history.append(accuracy)
        n_supports.append(n_sup)
        
    logger.info('Train epoch %d loss: %.4f acc: %.4f'%(epoch, loss.item(), np.mean(accuracy)))
    
    total_loss = np.average(np.array(loss_history), axis=0, weights=n_supports)
    total_accuracy = np.average(np.array(accuracy_history), axis=0, weights=n_supports)
    
    logger.add_scalars('train_loss', total_loss, cfg.names, [], epoch)
    logger.add_scalars('train_acc', total_accuracy, cfg.names, [], epoch)
    return np.mean(total_loss), np.mean(total_accuracy)


def valid_epoch(epoch, model, dataloader, criterion, metric, multitask, device, logger, cfg):
    model.eval()
    loss_history=[]
    accuracy_history=[]
    n_supports=[]
    with torch.no_grad():
        for inputs, targets in dataloader:
            if multitask:
                inputs = [[x_i.to(device) for x_i in x] for x in inputs]
                targets = [[x_i.to(device) for x_i in x] for x in targets]
                outputs, samples = model.forward_multitask(inputs)
            else:
                inputs = [x.to(device) for x in inputs]
                targets = [x.to(device) for x in targets]
                outputs, samples = model(inputs)

            loss, loss_list = criterion(outputs, targets, logscale=False)
            accuracy, n_sup = metric(outputs, targets, return_supports=True)
            loss_history.append([x.item() for x in loss_list])
            accuracy_history.append(accuracy)
            n_supports.append(n_sup)
            
        logger.info('Val epoch %d loss: %.4f acc: %.4f'%(epoch, loss.item(), np.mean(accuracy)))
    
    total_loss = np.average(np.array(loss_history), axis=0, weights=n_supports)
    total_accuracy = np.average(np.array(accuracy_history), axis=0, weights=n_supports)

    logger.add_scalars('val_loss', total_loss, cfg.names, cfg.exclude_names, epoch)
    logger.add_scalars('val_acc', total_accuracy, cfg.names, cfg.exclude_names, epoch)
    return np.mean(total_loss), np.mean(total_accuracy)


def test_epoch(model, dataloader, multitask, config, device, logger, cfg):
    model.eval()

    collector = Collector(config.names)
    with torch.no_grad():
        for inputs, targets in dataloader:
            if multitask:
                inputs = [[x_i.to(device) for x_i in x] for x in inputs]
                targets = [[x_i.to(device) for x_i in x] for x in targets]
                outputs, samples = model.forward_multitask(inputs)
            else:
                inputs = [x.to(device) for x in inputs]
                targets = [x.to(device) for x in targets]
                outputs, samples = model(inputs)

            collector.add(outputs, targets)

    collector.calculate_stats()
    report=collector.report_accuracy(cfg.exclude_names)
    logger.info('Test: %s'%report)
    logger.info('Test_avg: %s'%report['avg'])
    logger.add_dict('test', report)
    return collector


def main(config):

    datasets = dict(
        train=MultiTaskDataset(config.data_root, task=config.task, mode=config.data_sample_mode, is_rel=True, swap_probs=getattr(config, 'swap_probs', None)),
        val=MultiTaskDataset(config.val_root, task=config.task, mode=config.data_sample_mode, is_rel=True),
    )
    dataloaders = dict(

        train=DataLoader(datasets['train'], batch_size=config.batch_size_train, num_workers=4, shuffle=True),
        val=DataLoader(datasets['val'], batch_size=config.batch_size_test, num_workers=4),
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BTN(config)
    model.to(device)

    criterion = multi_task_loss
    metric = multi_task_metric

    multitask = config.multitask

    logger = Logger(config.outdir)
    logger.add_config(config)

    train_losses = []
    train_accuracies=[]
    val_losses = []
    val_accuracies=[]
    if config.scheduling_warmup['nepochs']:
        load_vgg_weights(model, logger, device, config.scheduling_warmup['pretrained_weights'], config.scheduling_warmup['transfer_layers'])
        freeze_layers(model, config.scheduling_warmup['frozen_layers'], logger)
        optimizer = build_optimizer(
            model, lr=config.scheduling_warmup['learning_rate'],
            separate_group=config.scheduling_warmup.get('separate_group',[]),
            separate_lr=config.scheduling_warmup.get('separate_lr',0),
            train_scale=getattr(config, 'train_scale', False), logger=logger)

        for n in range(config.scheduling_warmup['nepochs']):
            loss, acc = train_epoch(n, model, dataloaders['train'], optimizer, criterion, metric, multitask, device, logger, config)
            train_losses.append(loss)
            train_accuracies.append(acc)
            if n%1==0:

                loss, acc = valid_epoch(n, model, dataloaders['val'], criterion, metric, multitask, device, logger, config)
                val_losses.append(loss)
                val_accuracies.append(acc)
                save_weights(model, config.outdir+'/model.pth')

    optimizer = getattr(torch.optim, config.optimizer_type)(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    print(optimizer)
    if config.pretrained_weights:
        load_weights(model, config.pretrained_weights)
    
    freeze_layers(model, config.frozen_layers, logger)
    model.alpha.requires_grad_(getattr(config, 'train_scale', False))


    for n in range(config.scheduling_warmup['nepochs'], config.nepochs + config.scheduling_warmup['nepochs']):
        loss, acc = train_epoch(n, model, dataloaders['train'], optimizer, criterion, metric, multitask, device, logger, config)
        train_losses.append(loss)
        train_accuracies.append(acc)
        
        if n%1==0:
            loss, acc = valid_epoch(n, model, dataloaders['val'], criterion, metric, multitask, device, logger, config)
            val_losses.append(loss)
            val_accuracies.append(acc)
            save_weights(model, config.outdir+'/model.pth')

    test_epoch(model, dataloaders['val'], multitask, config, device, logger, config)


    plt.figure(figsize=(8,8))
    fig = plt.subplot(211)
    plt.plot(train_losses, color='blue')
    plt.plot(val_losses, color='red')
    plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
    plt.xlabel('number of training epochs')
    plt.ylabel('negative log likelihood loss')

    fig = plt.subplot(212)
    plt.plot(train_accuracies, color='blue')
    plt.plot(val_accuracies, color='red')
    plt.legend(['Train Acc', 'Val Acc'], loc='upper right')
    plt.xlabel('number of training epochs')
    plt.ylabel('accuracy')

    plt.savefig(config.outdir+'/loss.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args=parser.parse_args()

    config = cfg_dict[args.config]()
    main(config)
