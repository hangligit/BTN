import numpy as np
import os
import sys
import logging
from collections import defaultdict
import argparse
import pickle as pkl


# logging
logger = logging.getLogger('eval')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

logfile = logging.FileHandler("log.txt")
logger.addHandler(logfile)


class Collector:
    def __init__(self, results_history):
        self.results_history = results_history

    def report_accuracy(self):
        keys=list(next(iter(self.results_history.values())).keys())
        ranks=defaultdict(list)
        for item in self.results_history.values():
            for k in keys:
                ranks[k].append(item[k])
                
        metrics=dict()
        for k,v in ranks.items():
            metrics[k+'_'+'accuracy'] = self.calculate_accuracy(np.array(v))
            metrics[k+'_'+'hits'] = self.calculate_hits_at_k(np.array(v))
            
        out = dict()
        for k, v in metrics.items():
            if k.endswith('_accuracy'):
                out[k] = v
        avg = [v for k, v in out.items()]
        out['avg'] = np.array(avg).mean()
        metrics.update(out)
        return metrics

    @staticmethod
    def calculate_hits_at_k(ranks, hits=[1, 3, 5]):
        assert ranks.ndim == 1
        return [len(np.where(ranks < hit)[0])/len(ranks) for hit in hits]

    @staticmethod
    def calculate_accuracy(ranks):
        assert ranks.ndim == 1
        return (ranks == 0).mean()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('result_file', type=str)
    args=parser.parse_args()
    
    result_history=pkl.load(open(args.result_file,'rb'))
    out=Collector(result_history).report_accuracy()

    logger.info(str(out))
