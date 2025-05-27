import random, torch, torch.nn as nn, torch.optim as optim
from collections import deque
from q_network import QNetwork

class RLAgent:
    def __init__(
        self, num_nodes, emb_dim, lr=1e-3, gamma=0.99, buffer_size=100,
        batch_size=16, target_update_freq=5, n_step=15, device="cpu"
    ):
        self.q_net = QNetwork(num_nodes, emb_dim).to(device)
        self.target_net = QNetwork(num_nodes, emb_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optim = optim.Adam(self.q_net.parameters(), lr=lr)
        self.device = device
        self.gamma = gamma
        self.n_step = n_step
        self.buffer = deque(maxlen=buffer_size)
        self.n_step_buffer = []
        self.batch = batch_size
        self.tu_freq = target_update_freq
        self.step = 0

    def store(self, state, action, reward, next_state, graph_prev, graph_next, done):
        self.n_step_buffer.append((state, action, reward, next_state, graph_prev, graph_next, done))
        # 湊滿 n 步才寫進 buffer
        if len(self.n_step_buffer) >= self.n_step:
            s_k_n, a_k_n, _, _, G_k_n, _, _ = self.n_step_buffer[0]
            R = 0
            gamma = 1
            for i in range(self.n_step):
                R += self.n_step_buffer[i][2] * gamma
                gamma *= self.gamma
            s_k, _, _, _, _, G_k, done_flag = self.n_step_buffer[self.n_step - 1]
            self.buffer.append((s_k_n, a_k_n, R, s_k, G_k_n, G_k, done_flag))
            self.n_step_buffer.pop(0)
        # 若 episode 結束，把剩下的 transition 都寫進 buffer
        if done:
            while len(self.n_step_buffer) > 0:
                steps = len(self.n_step_buffer)
                s_k_n, a_k_n, _, _, G_k_n, _, _ = self.n_step_buffer[0]
                R = 0
                gamma = 1
                for i in range(steps):
                    R += self.n_step_buffer[i][2] * gamma
                    gamma *= self.gamma
                s_k, _, _, _, _, G_k, done_flag = self.n_step_buffer[-1]
                self.buffer.append((s_k_n, a_k_n, R, s_k, G_k_n, G_k, done_flag))
                self.n_step_buffer.pop(0)

    def select(self, edge_index, u_idx, v_idx, eps):
        if random.random() < eps:
            return random.randint(0, v_idx.numel() - 1)
        with torch.no_grad():
            q_vals = self.q_net(edge_index, u_idx, v_idx)
            return q_vals.argmax().item()

    def update(self, u_batch, i_batch):
        if len(self.buffer) < self.batch:
            return

        batch = random.sample(self.buffer, self.batch)
        s_batch, a_batch, r_batch, s_next_batch, _, _, done_batch = zip(*batch)

        u_idx_all = torch.tensor([a[0] for a in a_batch], device=self.device)
        v_idx_all = torch.tensor([a[1] for a in a_batch], device=self.device)
        r = torch.tensor(r_batch, dtype=torch.float32, device=self.device)
        done_mask = torch.tensor(done_batch, dtype=torch.float32, device=self.device)

        q_pred_list = []
        q_next_list = []

        for i in range(self.batch):
            # --- Q(s, a) ---
            edge_index = s_batch[i].to(self.device)
            q_pred = self.q_net(edge_index, u_idx_all[i].unsqueeze(0), v_idx_all[i].unsqueeze(0))
            q_pred_list.append(q_pred)

            # --- Q(s', a')，max over all a' ---
            edge_index_next = s_next_batch[i].to(self.device)
            with torch.no_grad():
                q_vals = self.target_net(edge_index_next, u_batch, i_batch)
                q_next  = q_vals.max()

            q_next_list.append(q_next)

        q_pred = torch.cat(q_pred_list, dim=0)
        q_next = torch.stack(q_next_list, dim=0)

        target = r + (1 - done_mask) * (self.gamma ** self.n_step) * q_next

        loss = nn.MSELoss()(q_pred, target.detach())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.step += 1
        if self.step % self.tu_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())