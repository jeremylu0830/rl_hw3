import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Deep Recurrent Q Learning
# Slide 17
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module4.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 2               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25        # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 2000         # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency
TRACE_LENGTH = 8 

# Global variables
EPSILON = STARTING_EPSILON
Q = None

class RecurrentReplayBuffer:
    def __init__(self, capacity, trace_length):
        self.capacity = capacity
        self.trace_length = trace_length
        self.buffer = [] # 儲存完整的 episodes
        self.current_episode = []

    def add(self, obs, action, reward, next_obs, done):
        # 將 transition 加入當前的 episode
        self.current_episode.append((obs, action, reward, next_obs, done))
        
        if done:
            # 如果 episode 結束了
            # 只有當 episode 長度足夠長時才儲存
            if len(self.current_episode) >= self.trace_length:
                self.buffer.append(self.current_episode)
            self.current_episode = []
            
            # 如果 buffer 滿了，移除最舊的 episode
            while len(self.buffer) > self.capacity:
                self.buffer.pop(0)

    def sample(self, batch_size):
        # 1. 隨機抽取 'batch_size' 個 episodes
        # (允許重複抽取)
        sampled_episodes_indices = np.random.randint(0, len(self.buffer), batch_size)
        
        # 2. 從每個 episode 中，隨機抽取一個有效的起始點
        start_indices = []
        sampled_episodes = []
        for idx in sampled_episodes_indices:
            episode = self.buffer[idx]
            # 確保有足夠的長度來抽取一個 trace
            max_start_idx = len(episode) - self.trace_length
            start_indices.append(np.random.randint(0, max_start_idx + 1))
            sampled_episodes.append(episode)

        # 3. 建立 batch
        obs_b, act_b, rew_b, next_obs_b, done_b = [], [], [], [], []

        for episode, start_idx in zip(sampled_episodes, start_indices):
            # 取得長度為 T 的序列
            trace = episode[start_idx : start_idx + self.trace_length]
            
            # 將 (obs, act, ...) 序列解開
            obs, act, rew, next_obs, done = zip(*trace)
            
            obs_b.append(np.array(obs))
            act_b.append(np.array(act))
            rew_b.append(np.array(rew))
            next_obs_b.append(np.array(next_obs))
            done_b.append(np.array(done))

        # 4. 堆疊並轉換為 PyTorch 張量
        # obs_b 形狀: (B, T, F)
        obs_b = t.f(np.stack(obs_b))
        # act_b 形狀: (B, T) -> (B, T, 1)
        act_b = t.f(np.stack(act_b)).long().unsqueeze(-1)
        # rew_b 形狀: (B, T) -> (B, T, 1)
        rew_b = t.f(np.stack(rew_b)).float().unsqueeze(-1)
        # next_obs_b 形狀: (B, T, F)
        next_obs_b = t.f(np.stack(next_obs_b))
        # done_b 形狀: (B, T) -> (B, T, 1)
        done_b = t.f(np.stack(done_b)).float().unsqueeze(-1)

        return obs_b, act_b, rew_b, next_obs_b, done_b

    def __len__(self):
        # 回傳目前儲存的 episode 數量
        return len(self.buffer)

# Deep recurrent Q network
class DRQN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        # 儲存隱藏層大小
        self.hidden_size = HIDDEN
        self.fc1 = nn.Linear(OBS_N, HIDDEN) #input layer
        self.lstm = nn.LSTM(input_size=HIDDEN, #hidden layer 
                            hidden_size=HIDDEN, 
                            num_layers=1, 
                            batch_first=True)
        self.fc2 = nn.Linear(HIDDEN, ACT_N)
        ## TODO: Create layers of DRQN (已完成)
    
    def forward(self, x, hidden):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.unsqueeze(1)
        lstm_out, next_hidden = self.lstm(x, hidden)
        x = lstm_out.squeeze(1)
        q_values = self.fc2(x) # 形狀: (batch_size, ACT_N)
        
        return q_values, next_hidden
        ## TODO: Forward pass (已完成)
        
    def init_hidden(self, batch_size):
        # 回傳一個 (h_n, c_n) 元組
        return (torch.zeros(1, batch_size, self.hidden_size).to(DEVICE),torch.zeros(1, batch_size, self.hidden_size).to(DEVICE))

# Create environment
# Create replay buffer
# Create network for Q(s, a)
# Create target network
# Create optimizer
def create_everything(seed):
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.PartiallyObservableCartPole(), 200)
    test_env = utils.envs.TimeLimit(utils.envs.PartiallyObservableCartPole(), 200)
    
    buf = RecurrentReplayBuffer(BUFSIZE, TRACE_LENGTH) 
    
    Q = DRQN().to(DEVICE)
    Qt = DRQN().to(DEVICE)
    OPT = torch.optim.Adam(Q.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT

## <--- [要求二]
## 刪除舊的、無狀態的 `policy` 函數。
## 我們將把它的邏輯直接移入 `train` 函數的主迴圈中。
# def policy(env, obs): ... (REMOVED)


# Update networks
def update_networks(epi, buf, Q, Qt, OPT):
    try:
        obs, act, rew, obs_next, done = buf.sample(MINIBATCH_SIZE)
    except ValueError:
        return 0.
    
    B = obs.shape[0]
    T = obs.shape[1]
    
    hidden = Q.init_hidden(B)
    hidden_t = Qt.init_hidden(B)
    
    q_values = []
    q_values_target_next = []
    
    for t in range(T):
        o_t = obs[:, t, :]
        o_next_t = obs_next[:, t, :]
        
        q_t, hidden = Q(o_t, hidden)
        q_values.append(q_t)
        
        # ✅ 修正: 在序列中間步驟 detach hidden
        # 這樣可以截斷 BPTT,避免梯度爆炸
        if t < T - 1:
            hidden = tuple(h.detach() for h in hidden)
        
        with torch.no_grad():
            q_next_t, hidden_t = Qt(o_next_t, hidden_t)
        q_values_target_next.append(q_next_t)
    
    q_s = torch.stack(q_values, dim=1)
    q_s_next = torch.stack(q_values_target_next, dim=1)
    q_s_a = q_s.gather(2, act.long())
    q_s_next_max = q_s_next.max(2, keepdim=True)[0]
    q_target = rew + GAMMA * q_s_next_max * (1 - done)
    
    loss = F.mse_loss(q_s_a, q_target.detach())
    
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    if epi % TARGET_NETWORK_UPDATE_FREQ == 0:
        Qt.load_state_dict(Q.state_dict())

    return loss.item()


# Play episodes
# Training function
def train(seed):
    global EPSILON, Q
    print("Seed=%d" % seed)

    env, test_env, buf, Q, Qt, OPT = create_everything(seed)
    EPSILON = STARTING_EPSILON
    testRs = [] 
    last25testRs = []
    
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:
        obs = env.reset()
        done = False
        hidden = Q.init_hidden(batch_size=1) 
        
        while not done:
            obs_tensor = t.f(obs).view(-1, OBS_N)

            with torch.no_grad():
                q_values, next_hidden = Q(obs_tensor, hidden)
            
            if np.random.rand() < EPSILON:
                action = np.random.randint(ACT_N)
            else:
                action = q_values.argmax().item()
            
            EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
            next_obs, reward, done, _ = env.step(action)
            buf.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
            # ✅ 修正: detach hidden state
            hidden = tuple(h.detach() for h in next_hidden)
        
        if epi >= TRAIN_AFTER_EPISODES:
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Q, Qt, OPT)

        # 測試部分
        Rews = []
        for epj in range(TEST_EPISODES):
            obs = test_env.reset()
            done = False
            ep_R = []
            hidden = Q.init_hidden(batch_size=1)
            
            while not done:
                obs_tensor = t.f(obs).view(-1, OBS_N)
                
                with torch.no_grad():
                    q_values, next_hidden = Q(obs_tensor, hidden)
                
                action = q_values.argmax().item()
                next_obs, reward, done, _ = test_env.step(action)
                ep_R += [reward]
                
                obs = next_obs
                # ✅ 修正: detach hidden state
                hidden = tuple(h.detach() for h in next_hidden)
            
            Rews += [sum(ep_R)]
            
        testRs += [sum(Rews)/TEST_EPISODES]
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()
    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'drqn')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig('drqn_gemini_performance.png', dpi=300, bbox_inches='tight')
    plt.show()