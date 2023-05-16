import torch
import torch.nn as nn
import torch.nn.functional as F
from offpolicy.utils.util import adj_init,init, to_torch

class AdjPolicy(nn.Module):
    """
    Computes total Q values given agent q values and global states.
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
    """
    def __init__(self, args, obs_dim ,state_dim, device, use_ReLU):
        super(AdjPolicy, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.adj_hidden_dim = args.adj_hidden_dim
        self.adj_output_dim = args.adj_output_dim
        self.num_variable = args.num_agents
        self.num_factor = args.num_factor
        self._use_orthogonal = args.use_orthogonal
        # dimension of the hidden layer of the mixing net
        self.hidden_layer_dim = args.mixer_hidden_dim
        # dimension of the hidden layer of each hypernet
        self.hypernet_hidden_dim = args.hypernet_hidden_dim

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        # 2 layer hypernets: output dimensions are same as above case
        self.hyper_w1 = nn.Sequential(
          init_(nn.Linear(state_dim, self.adj_output_dim * self.adj_output_dim//2)),
          nn.ReLU(),
          init_(nn.Linear(self.adj_output_dim * self.adj_output_dim//2, self.adj_output_dim * self.adj_output_dim//2))
        )
        self.hyper_b1 = init_(nn.Linear(state_dim, self.num_variable * self.adj_output_dim//2))  
        self.hyper_w2 = nn.Sequential(
                init_(nn.Linear(state_dim, self.adj_output_dim//2 * self.num_factor * 2)),
                nn.ReLU(),
                init_(nn.Linear(self.adj_output_dim//2 *self.num_factor * 2, self.adj_output_dim//2 * self.num_factor))
            )
        self.hyper_b2 = nn.Sequential(
            init_(nn.Linear(state_dim, self.num_variable * self.adj_output_dim//2)),
            nn.ReLU(),
            init_(nn.Linear(self.num_variable * self.adj_output_dim//2, self.num_variable * self.num_factor))
        )

    def forward(self, agent_emb, states):
        """
         Computes Q_tot using the individual agent q values and global state.
         :param agent_q_inps: (torch.Tensor) individual agent q values
         :param states: (torch.Tensor) state input to the hypernetworks.
         :return Q_tot: (torch.Tensor) computed Q_tot values
         """
        #agent_emb = to_torch(agent_emb).to(**self.tpdv)
        #states = to_torch(states).to(**self.tpdv)
        batch_size = agent_emb.size(0)
        

        w1 = self.hyper_w1(states)
        b1 = self.hyper_b1(states)
        w1 = w1.view(batch_size, self.adj_output_dim, self.adj_output_dim//2)
        b1 = b1.view(batch_size, self.num_variable, self.adj_output_dim//2)
        hidden_layer = F.elu(torch.matmul(agent_emb, w1) + b1)

        w2 = self.hyper_w2(states)
        b2 = self.hyper_b2(states)
        w2 = w2.view(batch_size, self.adj_output_dim//2, self.num_factor)
        b2 = b2.view(batch_size, self.num_variable, self.num_factor)
        out = torch.matmul(hidden_layer, w2) + b2
        
        max_out = torch.max(torch.abs(out),dim=1,keepdim=True)[0].detach() + 1e-5
        norm_out = torch.min(torch.ones_like(max_out ,dtype=torch.float32), 5 / max_out)
        out = torch.sigmoid(out * norm_out)
        
        return out
