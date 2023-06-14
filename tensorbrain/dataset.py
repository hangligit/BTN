from torch.utils.data import Dataset
import pickle as pkl
import numpy as np
import os


def build_label_heads(goal='normalize'):
    cls = 100
    cat = 8
    liv = 2
    age = 2
    col = 14
    act = 10
    dan = 2
    ent = 27000
    if goal == 'normalize':
        return [0, cls, cat, liv, age, col, act, dan]
    else:
        return [0, cls, cat, liv, age, col, act, ent]


index_baselines = np.cumsum(build_label_heads('normalize'))
index_offsets = np.cumsum(build_label_heads('aggregate'))


class MultiTaskDataset(Dataset):
    def __init__(self, root, range=(0., 1.), task='perception', mode='entity', is_rel=True, swap_probs=None):
        super().__init__()
        '''
        swap_probs: sampling probability for each attribute, in the order of (bclass,pclass,gclass,age,color,act,entity)
        '''
        if isinstance(root, str):
            self.filepaths = self._list_dir(root)
        else:
            self.filepaths = []
            for root_i in root:
                self.filepaths.extend(self._list_dir(root_i))

        self.filepaths = self.filepaths[int(len(self.filepaths) * range[0]):int(len(self.filepaths) * range[1])]

        self.task = task
        self.mode = mode
        self.is_rel = is_rel
        self.multitask = isinstance(self.task, list)
        self.swap_probs=swap_probs
        if swap_probs is not None:
            assert len(self.swap_probs)==7

    def _list_dir(self, root):
        return sorted([os.path.join(root, x) for x in os.listdir(root)
                       if os.path.isfile(os.path.join(root, x))
                       and not x.startswith('.')])

    def _prepare_unary_label(self, labels):
        # return n_index target and n_unary target

        labels_normalized = np.array(labels) - index_baselines
        labels_normalized[[-2, -1]] = labels_normalized[[-1, -2]]

        labels_add_offsets = labels_normalized + index_offsets

        labels_normalized_valid = labels_normalized[:-1]
        labels_add_offsets_valid = labels_add_offsets[:-1]

        n_samples = dict(
            entity=labels_add_offsets_valid[-1],
            category=labels_add_offsets_valid[0],
            extend=np.random.choice(labels_add_offsets_valid, p=self.swap_probs)
        )
        n_unaries = dict(
            perception=labels_normalized_valid,
            tkg=labels_normalized,
            skg=labels_normalized,
            perception_eval=labels_normalized,
        )
        return n_samples, n_unaries

    def __getitem__(self, item):
        data = pkl.load(open(self.filepaths[item], 'rb'))
        img_rep, s_rep, o_rep, p_rep, t, s, o, p, s_enhan, o_enhan = data

        s_handler = self._prepare_unary_label(s_enhan)
        o_handler = self._prepare_unary_label(o_enhan)

        if not self.multitask:
            s_sample, s_unaries = s_handler[0][self.mode], s_handler[1][self.task]
            o_sample, o_unaries = o_handler[0][self.mode], o_handler[1][self.task]

            inputs = [img_rep, s_rep, o_rep, p_rep, t, s_sample, o_sample]
            if self.is_rel:
                targets = [t, s_sample, s_unaries, o_sample, o_unaries, p]
                if self.task == 'tkg' or self.task == 'skg':
                    targets = targets[1:]
            else:
                targets = [t, s_sample, s_unaries]
                if self.task == 'tkg' or self.task == 'skg':
                    targets = targets[1:]
        else:
            inputs = []
            targets = []
            for task_i, mode_i in zip(self.task, self.mode):
                s_sample, s_unaries = s_handler[0][mode_i], s_handler[1][task_i]
                o_sample, o_unaries = o_handler[0][mode_i], o_handler[1][task_i]

                inputs.append([img_rep, s_rep, o_rep, p_rep, t, s_sample, o_sample])
                if self.is_rel:
                    tgt = [t, s_sample, s_unaries, o_sample, o_unaries, p]
                    if task_i == 'tkg' or task_i == 'skg':
                        tgt = tgt[1:]
                    targets.append(tgt)
                else:
                    tgt = [s_sample, s_unaries]
                    if task_i == 'tkg' or task_i == 'skg':
                        tgt = tgt[1:]
                    targets.append(tgt)
        return inputs, targets

    def __len__(self):
        return len(self.filepaths)
