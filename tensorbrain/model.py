import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def multinomial_sampling(probs):
    sampler = Categorical(probs=probs)
    samples = sampler.sample()
    probs = sampler.probs[torch.arange(probs.size(0), device=probs.device), samples]
    return samples, probs


def onehot_encoding_torch(label, num_classes):
    return (torch.arange(num_classes, device=label.device).reshape(1, num_classes) == label.view(-1, 1)).float()


def onehot_encoding_np(label, num_classes):
    return (label == np.arange(num_classes)).astype(np.float32)


class GroupSoftmax(nn.Module):
    def __init__(self, slices, activation='Softmax'):
        super().__init__()
        self.slices = slices
        self.function = getattr(torch.nn, activation)(dim=-1)

    def forward(self, x):
        return [self.function(x[:, i]) for i in self.slices]


class EpisodicEmbedding(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(1, ndim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, nonlinearity='relu')

    def forward(self, size):
        return self.weights.expand(size, -1)


class IndexLayer(nn.Module):
    def __init__(self, nindices, ndims):
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(nindices, ndims))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, nonlinearity='relu')

    def forward(self, x):
        return F.linear(x, self.weights)

    def sampling(self, tf_index):
        return self.weights[tf_index], (None, None)

    def sampling_and_backward(self, n, tf_index, mode):
        if self.training:
            n = torch.exp(n)

        i_samples = i_probs = None

        if mode == 'integral':
            a = torch.mm(n, self.weights)
        elif mode == 'teacher':
            a = self.weights[tf_index]
        elif mode == 'max':
            a = self.weights[n.argmax(1)]
        elif mode == 'sample':
            i_samples, i_probs = multinomial_sampling(n)
            a = self.weights[i_samples]
        else:
            raise NotImplementedError

        if (not self.training) and (i_samples is None and i_probs is None):
            i_samples, i_probs = multinomial_sampling(n)

        return a, (i_samples, i_probs)


class BTN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_episodes = config.num_episodes
        self.num_concepts = config.num_concepts
        self.num_entities = config.num_entities
        self.num_predicates = config.num_predicates

        self.D = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
        )

        self.A_T = IndexLayer(self.num_episodes, 4096)
        self.A_C = IndexLayer(self.num_concepts + self.num_entities, 4096)
        self.A_P = nn.Linear(4096, self.num_predicates, bias=False)
        self.a_bar = EpisodicEmbedding(4096)

        self.W = nn.Sequential(
            nn.Linear(500, 4096),
            nn.ReLU()
        )
        config.wlinear=getattr(config,'wlinear',0)
        if config.wlinear:
            self.W=nn.Sequential(nn.Linear(500,4096))

        self.V = nn.Sequential(
            nn.Linear(4096, 500),
            nn.ReLU()
        )
        self.B = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU()
        )

        self.dropout_v = nn.Dropout(config.dropout_v)
        self.dropout_m = nn.Dropout(config.dropout_m)
        self.dropout1 = nn.Dropout(config.dropout1)
        self.dropout2 = nn.Dropout(0.5)

        self.ops = dict(
            perception=self.forward_perception,
            tkg=self.forward_tkg,
            skg=self.forward_skg,
        )

        self.new_W=getattr(config,'new_W',False)
        if self.new_W:
            self.W1 = nn.Sequential(
                nn.Linear(500, 4096),
            )

        self.new_T=getattr(config, 'new_T',False)

        self.norm=lambda x:x
        if getattr(config, 'norm', False):
            self.norm=nn.LayerNorm(4096, elementwise_affine=False)

        config.flinear=getattr(config,'flinear',0)
        if config.flinear==0:
            self.f_nonlinear=F.relu
        elif config.flinear==1:
            self.f_nonlinear=lambda x:x
        elif config.flinear==2:
            self.f_nonlinear=nn.LeakyReLU(0.02)
        elif config.flinear==3:
            self.f_nonlinear=nn.LeakyReLU(0.1)
        elif config.flinear==4:
            self.f_nonlinear=nn.LeakyReLU(0.2)
        else:
            raise

        self.alpha=nn.Parameter(torch.tensor(1.))

        self.task = config.task
        self.sampling_training = config.sampling_training
        self.sampling_eval = config.sampling_eval

        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.group_softmax = GroupSoftmax(config.semantic_heads, 'Softmax')
        self.log_group_softmax = GroupSoftmax(config.semantic_heads, 'LogSoftmax')

    def forward(self, x):
        return self.ops[self.task](x, self.sampling_training if self.training else self.sampling_eval)

    def forward_multitask(self, x):
        outs = []
        samples = []
        for i, task in enumerate(self.task):
            out, sample = self.ops[task](x[i], self.sampling_training[i] if self.training else self.sampling_eval[i])
            outs.extend(out)
            samples.extend(sample)
        return outs, samples

    def predict(self, x, task, mode):
        self.eval()
        return self.ops[task](x, mode)

    def predict_multitask(self, x, tasks, modes):
        self.eval()
        outs = []
        samples = []
        for i, (task, mode) in enumerate(zip(tasks, modes)):
            out, sample = self.ops[task](x[i], mode)
            outs.extend(out)
            samples.extend(sample)
        return outs, samples

    def forward_perception(self, x, mode):
        img_rep, s_rep, o_rep, p_rep, t_index, s_index, o_index = x
        f_t = self.D(img_rep.flatten(1))
        f_s = self.D(s_rep.flatten(1))
        f_o = self.D(o_rep.flatten(1))
        f_p = self.D(p_rep.flatten(1))

        activation = self.log_softmax if self.training else self.softmax
        activation_grp = self.log_group_softmax if self.training else self.group_softmax

        # q: representation vector, a: embedding vector from memory, n: probability vector
        # episode
        q_tilde_t = f_t
        n_t = activation(self.A_T(self.dropout1(self.f_nonlinear(q_tilde_t))))
        a_t, (t_samples, t_probs) = self.A_T.sampling_and_backward(n_t, t_index, mode)
        q_t = a_t + q_tilde_t
        if self.new_T:
            q_t = q_tilde_t
        h_skip = self.B(self.V(q_t))

        # subject
        h_w = self.W(h_skip)
        q_tilde_s = f_s + h_w
        n_tilde_s = activation(self.A_C(self.dropout1(self.f_nonlinear(q_tilde_s))))
        a_s, (s_samples, s_probs) = self.A_C.sampling_and_backward(n_tilde_s, s_index, mode)
        q_s = self.dropout_m(a_s)*self.alpha + self.dropout_v(q_tilde_s)
        n_s = activation_grp(self.A_C(self.dropout2(self.f_nonlinear(self.norm(q_s)))))
        h_skip = self.B(self.V(q_s) + h_skip)

        # object
        h_w = self.W(h_skip)
        q_tilde_o = f_o + h_w
        n_tilde_o = activation(self.A_C(self.dropout1(self.f_nonlinear(q_tilde_o))))
        a_o, (o_samples, o_probs) = self.A_C.sampling_and_backward(n_tilde_o, o_index, mode)
        q_o = self.dropout_m(a_o)*self.alpha + self.dropout_v(q_tilde_o)
        n_o = activation_grp(self.A_C(self.dropout2(self.f_nonlinear(self.norm(q_o)))))
        h_skip = self.B(self.V(q_o) + h_skip)

        # predicate
        h_w = self.W(h_skip)
        q_tilde_p = f_p + h_w
        n_p = activation(self.A_P(self.dropout2(self.f_nonlinear(q_tilde_p))))

        return (n_t, n_tilde_s, n_s, n_tilde_o, n_o, n_p), (s_samples, s_probs, o_samples, o_probs)

    def forward_tkg(self, x, mode):
        t_index, s_index, o_index = x[-3:]

        activation = self.log_softmax if self.training else self.softmax
        activation_grp = self.log_group_softmax if self.training else self.group_softmax

        # episode
        a_t, (t_samples, t_probs) = self.A_T.sampling(t_index)
        q_t = a_t
        h_skip = self.B(self.V(q_t))

        # subject
        h_w = self.W(h_skip)
        if self.new_W: h_w=h_w+self.W1(h_skip)
        q_tilde_s = h_w
        n_tilde_s = activation(self.A_C(self.dropout1(self.f_nonlinear(q_tilde_s))))
        a_s, (s_samples, s_probs) = self.A_C.sampling_and_backward(n_tilde_s, s_index, mode)
        q_s = self.dropout_m(a_s)*self.alpha + self.dropout_v(q_tilde_s)
        n_s = activation_grp(self.A_C(self.dropout2(self.f_nonlinear(self.norm(q_s)))))
        h_skip = self.B(self.V(q_s) + h_skip)

        # object
        h_w = self.W(h_skip)
        if self.new_W: h_w=h_w+self.W1(h_skip)
        q_tilde_o = h_w
        n_tilde_o = activation(self.A_C(self.dropout1(self.f_nonlinear(q_tilde_o))))
        a_o, (o_samples, o_probs) = self.A_C.sampling_and_backward(n_tilde_o, o_index, mode)
        q_o = self.dropout_m(a_o)*self.alpha + self.dropout_v(q_tilde_o)
        n_o = activation_grp(self.A_C(self.dropout2(self.f_nonlinear(self.norm(q_o)))))
        h_skip = self.B(self.V(q_o) + h_skip)

        # predicate
        h_w = self.W(h_skip)
        if self.new_W: h_w=h_w+self.W1(h_skip)
        q_tilde_p = h_w
        n_p = activation(self.A_P(self.dropout2(self.f_nonlinear(q_tilde_p))))

        return (n_tilde_s, n_s, n_tilde_o, n_o, n_p), (s_samples, s_probs, o_samples, o_probs)

    def forward_skg(self, x, mode):
        t_index, s_index, o_index = x[-3:]

        activation = self.log_softmax if self.training else self.softmax
        activation_grp = self.log_group_softmax if self.training else self.group_softmax

        # episode
        a_t = self.a_bar(t_index.size(0))
        q_t = a_t
        h_skip = self.B(self.V(q_t))

        # subject
        h_w = self.W(h_skip)
        if self.new_W: h_w=h_w+self.W1(h_skip)
        q_tilde_s = h_w
        n_tilde_s = activation(self.A_C(self.dropout1(self.f_nonlinear(q_tilde_s))))
        a_s, (s_samples, s_probs) = self.A_C.sampling(s_index)
        q_s = self.dropout_m(a_s)*self.alpha + self.dropout_v(q_tilde_s)
        n_s = activation_grp(self.A_C(self.dropout2(self.f_nonlinear(self.norm(q_s)))))
        h_skip = self.B(self.V(q_s) + h_skip)

        # object
        h_w = self.W(h_skip)
        if self.new_W: h_w=h_w+self.W1(h_skip)
        q_tilde_o = h_w
        n_tilde_o = activation(self.A_C(self.dropout1(self.f_nonlinear(q_tilde_o))))
        a_o, (o_samples, o_probs) = self.A_C.sampling_and_backward(n_tilde_o, o_index, mode)
        q_o = self.dropout_m(a_o)*self.alpha + self.dropout_v(q_tilde_o)
        n_o = activation_grp(self.A_C(self.dropout2(self.f_nonlinear(self.norm(q_o)))))
        h_skip = self.B(self.V(q_o) + h_skip)

        # predicate
        h_w = self.W(h_skip)
        if self.new_W: h_w=h_w+self.W1(h_skip)
        q_tilde_p = h_w
        n_p = activation(self.A_P(self.dropout2(self.f_nonlinear(q_tilde_p))))

        return (n_tilde_s, n_s, n_tilde_o, n_o, n_p), (s_samples, s_probs, o_samples, o_probs)
