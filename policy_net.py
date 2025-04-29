import torch
from torch import nn

class Actor(nn.Module):

    lin_block = lambda self,i,o: nn.Sequential(nn.Linear(i,o), nn.ReLU())
    
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = state_dim
        self.actor = nn.GRU(self.state_dim, self.action_dim, 3)
        self.mu = nn.Sequential(nn.Linear(self.state_dim,self.action_dim), nn.Tanh())
        self.log_std = nn.Sequential(nn.Linear(self.state_dim,self.action_dim), nn.Tanh())

    def forward(self, x, hx, decay=None):
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        if hx is None:
            out,hidden_state = self.actor(x)
        else:
            out,hidden_state = self.actor(x, hx)
        mu = self.mu(out)
        log_std = self.log_std(out)
        if decay is not None:
            mu = mu * torch.tensor(decay).float()
            # log_std = log_std * torch.tensor([min(1.5,max(-1.5,i)) for i in torch.log(torch.tensor(decay).float()*0.05)]).float()
            log_std = log_std + torch.log(0.1*abs(mu))
        return mu, torch.exp(log_std), hidden_state

    def sample(self, state, hs, decay=None):
        if len(state.shape)==1:
            state = state.unsqueeze(0)
        mu, std, hidden = self.forward(state, hs)
        action = torch.normal(mu, std)
        return action, hidden
    

class Critic(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.critic = nn.Sequential(nn.Linear(self.state_dim, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 200),
                                    nn.ReLU(),
                                    nn.Linear(200,1),
                                    nn.ReLU())
    
    def forward(self, x):
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        return self.critic(x)