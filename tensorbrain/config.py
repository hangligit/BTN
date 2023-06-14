from pathlib import Path
import json
import numpy as np
import torch
import sys
import inspect


HOME = str(Path.home())
NAMES = dict(
    perception=['T', 'S', 'Scls', 'Scat', 'Sliv', 'Sage', 'Scol', 'Sact', 'Sins', 'O', 'Ocls', 'Ocat', 'Oliv', 'Oage',
                'Ocol', 'Oact', 'Oins', 'P'],
    perception_simple=['T', 'S', 'Scls', 'Scat', 'Sliv', 'Sage', 'Scol', 'Sact', 'Sins'],
    tkg=['S', 'Scls', 'Scat', 'Sliv', 'Sage', 'Scol', 'Sact', 'Sins', 'Sdan', 'O', 'Ocls', 'Ocat', 'Oliv', 'Oage',
         'Ocol', 'Oact', 'Oins', 'Odan', 'P'],
    tkg_simple=['S', 'Scls', 'Scat', 'Sliv', 'Sage', 'Scol', 'Sact', 'Sins', 'Sdan'],
    skg=['S', 'Scls', 'Scat', 'Sliv', 'Sage', 'Scol', 'Sact', 'Sins', 'Sdan', 'O', 'Ocls', 'Ocat', 'Oliv', 'Oage',
         'Ocol', 'Oact', 'Oins', 'Odan', 'P'],
    skg_simple=['S', 'Scls', 'Scat', 'Sliv', 'Sage', 'Scol', 'Sact', 'Sins', 'Sdan'],
)


def get_names(tasks):
    ret = []
    for k in tasks:
        prefix = k[:3] + '_'
        ret += [prefix + t for t in NAMES[k]]
    return ret

def get_exclude_names(tasks):
    ret = []
    excludes={'T','S','O'}
    for k in tasks:
        prefix = k[:3] + '_'
        ret += [prefix + t for t in NAMES[k] if t in excludes]
    return ret


class BaseConfig:
    HOME = str(Path.home())

    outdir = 'models/developmodelbt2n31'
    # data
    data_root = '../../ddddbug'
    val_root = '../../ddddbug'
    batch_size_train = 128
    batch_size_test = 128
    data_sample_mode = 'entity'

    # model
    num_entities = 27000  # 27000
    num_predicates = 72
    num_episodes = 9000
    num_concepts = 200
    semantic_heads_num = [100, 8, 2, 2, 14, 10, num_entities, 2]
    task = 'perception'
    sampling_training = 'integral'
    sampling_eval = 'integral'
    new_T=True

    dropout_m = 0.5
    dropout_v = 0.5
    dropout1 = 0

    # scheduling
    scheduling_warmup = dict(
        pretrained_weights=HOME+'/data/model_best_68_106.pth',
        transfer_layers=['fc1', 'fc2'],
        frozen_layers=['D'],
        nepochs=10,
        learning_rate=0.0001,
    )

    nepochs = 20
    optimizer_type='SGD'
    learning_rate = 0.0001
    weight_decay = 0
    pretrained_weights = None
    frozen_layers = []

    # test
    test_root = '../../ddddbug'
    test_weights='models/developmodelbt2n31/model.pth'
    test_task = 'tkg'
    test_sampling_mode = 'teacher'
    test_data_mode = 'entity'

    @property
    def semantic_heads(self):
        heads = np.cumsum([0] + self.semantic_heads_num)
        return [torch.arange(a, b, dtype=torch.int).long() for a, b in zip(heads[:-1], heads[1:])]

    @property
    def names(self):
        return get_names([self.task] if isinstance(self.task, str) else self.task)
    @property
    def exclude_names(self):
        return get_exclude_names([self.task] if isinstance(self.task, str) else self.task)
    @property
    def test_names(self):
        return get_names([self.test_task] if isinstance(self.test_task, str) else self.test_task)

    @property
    def multitask(self):
        return isinstance(self.task, list)

    @property
    def multitask_eval(self):
        return isinstance(self.test_task, list)

    @classmethod
    def _show_config(cls):
        params={k:v for (k,v) in inspect.getmembers(cls) if not k.startswith('_') and not isinstance(v,property)}
        return params

    @classmethod
    def _dump_config(cls, cfgdir):
        params={k:v for (k,v) in inspect.getmembers(cls) if not k.startswith('_') and not isinstance(v,property)}
        json.dump(params, open(cfgdir+'/'+'cfg.json', 'w'))
        return params

    @classmethod
    def _load_config(cls, cfgdir):
        params=json.load(open(cfgdir+'/'+'cfg.json', 'r'))
        for k,v in params.items():
            if not k.startswith('test_'):
                setattr(cls, k, v)
            else:
                print('skip: %s'%k)


#==============Train Config============#
class FinalTrain(BaseConfig):
    outdir='checkpoints_finaltrain/ours'
    # data
    data_root = '/home/ubuntu/data/vrdex_reprs_train'
    val_root = '/home/ubuntu/data/vrde_reprs_test'
    task = ['perception','tkg','skg']
    sampling_training = ['integral','teacher','teacher']
    sampling_eval = ['integral','teacher','teacher']
    data_sample_mode = ['entity','extend','extend']
    swap_probs = [0.01,0.002,0.002,0.002,0.002,0.002,0.98]

    new_T=False
    new_W=False
    norm=False
    flinear=2
    wlinear=1

    dropout_m=0
    dropout_v = 0.5
    dropout1 = 0
    
    train_scale=True

    scheduling_warmup = dict(
        pretrained_weights=HOME+'/data/model_best_68_106.pth',
        transfer_layers=['fc1', 'fc2'],
        frozen_layers=['D'],
        nepochs=60,
        learning_rate=0.0001,
        separate_group=['D.2.weight','D.2.bias'],
        separate_lr=1e-5,
    )
    nepochs=0

    test_root = '/home/ubuntu/data/vrde_reprs_test'
    test_weights='models/pes/model.pth'
    test_task = ['perception','tkg','skg']
    test_sampling_mode = ['integral','teacher','teacher']
    test_data_mode = ['entity','entity','entity']


#==============Test Config============#
class OursTest(BaseConfig):
    test_weights='checkpoints_finaltrain/ours/model.pth'
    test_task='perception'
    test_sampling_mode='integral'
    test_data_mode='entity'
# table 2
class T2R1(OursTest):
    test_root = '/home/ubuntu/data/vrde_reprs_test'
class T2R2(T2R1):
    test_sampling_mode = 'max'

class T2R4(OursTest):
    test_root='/home/ubuntu/data/vrdex_reprs_test'
class T2R5(T2R4):
    test_sampling_mode='max'

class T2R2t(T2R1):
    test_sampling_mode='teacher'
    test_data_mode='category'

# table 6
class T6R1(OursTest):
    test_root = '/home/ubuntu/data/vrde_reprs_train'
    test_task='tkg'
    test_sampling_mode='teacher'
    test_data_mode='entity'
class T6R2(BaseConfig):
    test_weights=''
    test_task='tkg'
    test_sampling_mode='teacher'
    test_data_mode='entity'

# table 8
class T8R1(OursTest):
    test_root = '/home/ubuntu/data/vrde_reprs_train'
    test_task='skg'
    test_sampling_mode='teacher'
    test_data_mode='category'
class T8R3(T8R1):
    test_data_mode='entity'

# table 9
class T9R3(OursTest):
    test_root = '/home/ubuntu/data/vrdex_reprs_test'
    test_names=['per_'+x for x in ['T', 'S', 'Scls', 'Scat', 'Sliv', 'Sage', 'Scol', 'Sact', 'Sins', 'Sdan', 'O', 'Ocls', 'Ocat', 'Oliv', 'Oage', 'Ocol', 'Oact', 'Oins', 'Odan', 'P']]
    test_task='perception_eval'

# SKG, TKG
class TSKG(BaseConfig):
    test_weights='checkpoints_finaltrain/ours/model.pth'
    test_root = '/home/ubuntu/data/vrde_reprs_train'
    test_task=['tkg','skg']
    test_sampling_mode=['teacher','teacher']
    test_data_mode=['entity','entity']


cfg_dict = {k:v for (k,v) in inspect.getmembers(sys.modules[__name__], inspect.isclass)}
