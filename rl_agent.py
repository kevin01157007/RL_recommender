# rl_agent.py
import random, torch, torch.nn as nn, torch.optim as optim
from collections import deque
from q_network import QNetwork

class RLAgent:
    def __init__(self,
                 num_nodes,
                 emb_dim,
                 lr=1e-3,
                 gamma=0.99,
                 buffer_size=100,
                 batch_size=16,
                 target_update_freq=5,
                 n_step=15,
                 device="cpu"):

        self.q_net      = QNetwork(num_nodes, emb_dim).to(device)
        self.target_net = QNetwork(num_nodes, emb_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optim   = optim.Adam(self.q_net.parameters(), lr=lr)
        self.device  = device
        self.gamma   = gamma
        self.n_step  = n_step
        self.buffer  = deque(maxlen=buffer_size)
        self.batch   = batch_size
        self.tu_freq = target_update_freq
        self.step    = 0

    # ----------- replay buffer -----------
    def store(self, transition):
        self.buffer.append(transition)

    # ----------- ε-greedy ---------------
    def select(self, edge_index, u_idx, v_idx, z_s, eps):
        if random.random() < eps:
            return random.randint(0, v_idx.numel() - 1)

        with torch.no_grad():
            q_vals = self.q_net(edge_index, u_idx, v_idx, z_s)
            return q_vals.argmax().item()

    # ----------- SGD update --------------
    def update(self, edge_index):
        if len(self.buffer) < self.batch: return

        batch = random.sample(self.buffer, self.batch)
        s_batch, a_batch, r_batch, s_next_batch = zip(*batch)

        # unpack
        u_idx = torch.tensor([a[0] for a in a_batch]).to(self.device)
        v_idx = torch.tensor([a[1] for a in a_batch]).to(self.device)
        z_s   = torch.stack(s_batch).to(self.device)
        z_s_n = torch.stack(s_next_batch).to(self.device)
        r     = torch.tensor(r_batch, dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_pred = self.q_net(edge_index, u_idx, v_idx, z_s)

        # r + γ max_a' Q_target(s',a')
        with torch.no_grad():
            q_next = self.target_net(edge_index, u_idx, v_idx, z_s_n)
            target = r + self.gamma * q_next

        loss = nn.MSELoss()(q_pred, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.step += 1
        if self.step % self.tu_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
