import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchRenorm(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, r_max=3.0, d_max=5.0):
        super(BatchRenorm, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.r_max = r_max
        self.d_max = d_max

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('steps', torch.tensor(0.0))

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, training=True):
        if training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # Compute renormalization parameters r and d
            r = torch.clamp(torch.sqrt(batch_var + self.eps) / torch.sqrt(self.running_var + self.eps), 
                            1/self.r_max, self.r_max).detach()
            d = torch.clamp((batch_mean - self.running_mean) / torch.sqrt(self.running_var + self.eps), 
                            -self.d_max, self.d_max).detach()

            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            x_hat = r * x_hat + d
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        return self.gamma * x_hat + self.beta
