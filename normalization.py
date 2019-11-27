import torch
from torch.nn import Module, init
from torch.nn import Parameter
from torch.nn import functional as F

# TODO: check contiguous in THNN
# TODO: use separate backend functions?
class _ConditionalInstanceNorm(Module):
    _version = 1
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, num_labels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ConditionalInstanceNorm, self).__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_labels, num_features))
            self.bias = Parameter(torch.Tensor(num_labels, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, label):
        # self._check_input_dim(input)
        if label >= self.num_labels:
            raise ValueError('Expected label to be < than {} but got {}'.format(self.num_labels, label))
        w = self.weight
        b = self.bias
        if self.affine:
            w = self.weight[label, :]
            b = self.bias[label, :]
            
        return F.instance_norm(
            input, self.running_mean, self.running_var, w, b,
            self.training or not self.track_running_stats,
            self.momentum, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 1) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_ConditionalInstanceNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
