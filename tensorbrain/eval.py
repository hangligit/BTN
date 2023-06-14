from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MultiTaskDataset
from model import BTN
from config import cfg_dict
import argparse
import pickle as pkl
import os
import sys
import logging
logger = logging.getLogger('eval')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def load_weights(model, filepath):
    logger.info(model.load_state_dict(torch.load(filepath, map_location=next(model.parameters()).device),strict=True))


class Collector:
    unary_label_keys={
        'full_S_accuracy':set(['Scls_accuracy','Scat_accuracy','Sliv_accuracy','Sage_accuracy','Scol_accuracy','Sact_accuracy','Sdan_accuracy']),
        'S_accuracy':set(['Scls_accuracy','Scat_accuracy','Sliv_accuracy','Scol_accuracy','Sact_accuracy']),
    }

    def __init__(self, names):
        self.results_history = []
        self.names = names

    def report_accuracy(self, exclude=None):
        out = dict()
        for k, v in self.results_history.items():
            if k.endswith('_accuracy'):
                out[k] = v

        avg = [v for k, v in out.items() if ((not exclude) or (k.split('_')[0] not in exclude))]
        out['avg'] = np.array(avg).mean()

        for k, v in self.results_history.items():
            if k.endswith('_P_hits') or k.endswith('_S_hits') or k.endswith('_O_hits'):
                out[k] = v

        for item,members in self.unary_label_keys.items():
            out[item] = np.mean([v for k,v in out.items() if '_'.join(k.split('_')[1:]) in members])
        return out

    def calculate_stats(self):
        self._view_as_dict()
        assert len(self.results_history) == len(self.names)
        metrics_history_dict = dict()
        for n, lhmn in zip(self.names, self.results_history):
            for k, v in lhmn.items():
                metrics_history_dict[n + '_' + k] = v
                if k == 'ranks':
                    metrics_history_dict[n + '_' + 'accuracy'] = self.calculate_accuracy(v)
                    metrics_history_dict[n + '_' + 'hits'] = self.calculate_hits_at_k(v)
        self.results_history = {k: self.to_cpu(v) for k, v in metrics_history_dict.items()}

    def add(self, inputs, targets, samples=None):
        if isinstance(targets[0], list):
            targets = self.flatten_list(targets)

        results = []  # L,N,M
        for input, target in zip(inputs, targets):
            if isinstance(input, list):
                for input_i, target_i in zip(input, target.T):
                    results.append(self._extract_metadata(input_i, target_i))
            else:
                results.append(self._extract_metadata(input, target))

        if samples:
            assert len(samples) % 2 == 0
            for i in range(0, len(samples), 2):
                if samples[i] is not None:
                    results.append(dict(instance_samples=samples[i], instance_probs=samples[i + 1]))

        self.results_history.append(results)  # History,(Location,Metric,Num)

    def _extract_metadata(self, inputs, targets, k=100):
        assert inputs.ndim == 2 and targets.ndim == 1, (inputs.ndim, targets.shape)
        label_ranks = self.get_label_rank(inputs, targets)
        label_probs = self.get_label_probability(inputs, targets)
        topk_samples, topk_probs = self.get_topk_probabilities(inputs, k)
        # if you need it back, do something like topk_probs[topk_samples], then you should recover predictions in original order
        return dict(ranks=label_ranks.cpu(), probs=label_probs.cpu(), topk_samples=topk_samples.cpu(), topk_probs=topk_probs.cpu(), gt=targets.cpu())

    def _view_as_dict(self):
        # H,L,M(rank,prob,probs),N -> L,H,M,N -> L,M,HN
        lhmn = zip(*self.results_history)
        lhmnnew = []
        for hmn in lhmn:
            m = defaultdict(list)
            for mn in hmn:
                for k, v in mn.items():
                    m[k].append(v)
            for k, v in m.items():
                m[k] = torch.cat(v, 0)
            lhmnnew.append(m)

        self.results_history = lhmnnew

    def to_cpu(self, x):
        if torch.is_tensor(x):
            return x.cpu().data.numpy()
        return x

    def accuracy(self):
        lll = np.average(np.array(self.accuracy_history), axis=0,
                         weights=self.n_supports_history)  # acchist N,M supphist N
        return {k: v for k, v in zip(self.names, lll)}

    def flatten_list(self, x):
        return [item for sublist in x for item in sublist]

    @staticmethod
    def get_label_rank(inputs, targets):
        assert inputs.ndim == 2 and targets.ndim == 1
        return torch.where(inputs.sort(dim=-1, descending=True).indices == targets.unsqueeze(1))[1]

    @staticmethod
    def get_label_probability(inputs, targets):
        assert inputs.ndim == 2 and targets.ndim == 1
        return inputs[torch.arange(targets.size(0), device=targets.device), targets]

    @staticmethod
    def get_topk_probabilities(inputs, topk):
        sort = inputs.sort(dim=-1, descending=True)
        return sort.indices[:, :topk], sort.values[:, :topk]

    @staticmethod
    def calculate_hits_at_k(ranks, hits=[1, 3, 5, 10]):
        assert ranks.ndim == 1
        return [len(torch.where(ranks < hit)[0])/len(ranks) for hit in hits]

    @staticmethod
    def calculate_accuracy(ranks):
        assert ranks.ndim == 1
        return (ranks == 0).float().mean().item()


def evaluation(model, dataloader, multitask, config, device, logger):
    model.eval()

    collector = Collector(config.test_names)

    if config.test_task=='perception_eval':
        config.test_task='perception'

    with torch.no_grad():
        for inputs, targets in dataloader:
            if multitask:
                inputs = [[x_i.to(device) for x_i in x] for x in inputs]
                targets = [[x_i.to(device) for x_i in x] for x in targets]
                outputs, samples = model.predict_multitask(inputs, config.test_task, config.test_sampling_mode)
            else:
                inputs = [x.to(device) for x in inputs]
                targets = [x.to(device) for x in targets]
                outputs, samples = model.predict(inputs, config.test_task, config.test_sampling_mode)

            collector.add(outputs, targets)

    collector.calculate_stats()
    for k,v in collector.report_accuracy().items():
        logger.info(k + ': %s', v)

    return collector

def main(config, outdir, save_name=None):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = MultiTaskDataset(config.test_root, task=config.test_task, mode=config.test_data_mode, is_rel=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size_test)

    model = BTN(config)
    model.to(device)
    load_weights(model, config.test_weights)

    multitask = config.multitask_eval

    collector=evaluation(model, dataloader, multitask, config, device, logger)
    if save_name:
        pkl.dump(collector.results_history, open(outdir+'/'+save_name,'wb'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--save', type=str, default=None)
    args=parser.parse_args()

    config = cfg_dict[args.config]()

    outdir=os.path.dirname(config.test_weights)

    config._load_config(outdir)

    logfile = logging.FileHandler(os.path.join(outdir, "eval.txt"))
    logger.addHandler(logfile)
    logger.info("==================== New Run ==================")

    for k,v in config._show_config().items():
        logger.info(k + ': %s', v)

    main(config, outdir, args.save)
