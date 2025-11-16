from turtle import update
import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# C51
# Based on Slide 11
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
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
EPISODES = 500          # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency
GRAD_CLIP_NORM = 10.0   # Gradient clipping threshold

# Suggested constants
ATOMS = 51              # Number of atoms for distributional network
ZRANGE = [0, 200]       # Range for Z projection 

# Global variables
EPSILON = STARTING_EPSILON
Z = None

# C51 globals
atoms = None
delta_z = None
Vmin = None
Vmax = None

# Create environment
# Create replay buffer
# Create distributional networks
# Create optimizer
def create_everything(seed):
    # 聲明我們將修改 C51 的全域變數
    global atoms, delta_z, Vmin, Vmax

    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)

    test_env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)

    buf = utils.buffers.ReplayBuffer(BUFSIZE)

    # C51 Atom (bins) setup
    Vmin, Vmax = ZRANGE
    # 創建 51 個 atoms (獎勵值的 "bins")，從 Vmin 到 Vmax
    atoms = torch.linspace(Vmin, Vmax, ATOMS).to(DEVICE)
    # 計算每個 atom 之間的間距
    delta_z = (Vmax - Vmin) / (ATOMS - 1)

    Z = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS) # 輸出是 (動作數 * atoms數)
    ).to(DEVICE)
    Zt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    
    # 初始時，將 Target 網路的權重設為與主網路相同
    Zt.load_state_dict(Z.state_dict())
    
    OPT = torch.optim.Adam(Z.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Z, Zt, OPT

# Create epsilon-greedy policy
def policy(env, obs):

    # 聲明 policy 會用到的全域變數
    global EPSILON, EPSILON_END, STEPS_MAX, Z, atoms
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        ## use Z to compute greedy action
        with torch.no_grad():
            # 1. 取得 Z-Network 的輸出 (logits)
            # 輸出 shape: [1, ACT_N * ATOMS]
            logits = Z(obs)
            
            # 2. Reshape 輸出為 [1, ACT_N, ATOMS] 並轉換為機率
            #    dim=2 表示在 ATOMS 維度上做 softmax
            probs = torch.softmax(logits.view(1, ACT_N, ATOMS), dim=2)
            
            # 3. 計算 Q-value (期望值)
            #    Q(s,a) = sum( prob(s,a,z_i) * z_i )
            #    (probs [1, ACT_N, ATOMS]) * (atoms [1, ATOMS])
            #    廣播 (broadcasting) 會自動處理
            q_values = (probs * atoms.unsqueeze(0)).sum(dim=2) # [1, ACT_N]
            
            # 4. 選擇 Q-value 最高的 action
            action = q_values.argmax(dim=1).item() # .item() 轉為 python 純量
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action

# Update networks
def update_networks(epi, tri, buf, Z, Zt, OPT):
    
    # 聲明 update 會用到的全域變數
    global atoms, delta_z, Vmin, Vmax

    # 1. 從 Replay Buffer 中取樣
    S, A, R, S_prime, D = buf.sample(MINIBATCH_SIZE, t)
    S = t.f(S); A = t.l(A); R = t.f(R); S_prime = t.f(S_prime); D = t.f(D)

    with torch.no_grad():
        # --- 建立 Target Distribution (C51 核心) ---
        
        # 2. 使用 Target Network (Zt) 取得下個狀態 (S_prime) 的分佈
        next_logits = Zt(S_prime).view(-1, ACT_N, ATOMS)
        next_probs = torch.softmax(next_logits, dim=2)
        
        # 3. 計算 Q-values (期望值) 以便找出最佳的 next_action
        next_q_values = (next_probs * atoms.unsqueeze(0)).sum(dim=2) # [BATCH_SIZE, ACT_N]
        
        # 4. 找出最佳的 next_action (a*)
        next_actions = next_q_values.argmax(dim=1) # [BATCH_SIZE]
        
        # 5. 取得 a* 對應的機率分佈 (p(s', a*))
        batch_indices = torch.arange(MINIBATCH_SIZE, device=DEVICE).long()
        next_dist_probs = next_probs[batch_indices, next_actions, :] # [BATCH_SIZE, ATOMS]
        
        # 6. 建立目標分佈 (m)，大小與 next_dist_probs 相同
        target_dist = torch.zeros_like(next_dist_probs)
        
        # 7. Bellman 投影 (Projection)
        #    將 atoms 投影到新的位置: Tz = R + (1-D) * gamma * z
        R_broadcast = R.unsqueeze(1)        # [BATCH_SIZE, 1]
        D_broadcast = D.unsqueeze(1)        # [BATCH_SIZE, 1]
        atoms_broadcast = atoms.unsqueeze(0)  # [1, ATOMS]
        
        Tz = R_broadcast + (1.0 - D_broadcast) * GAMMA * atoms_broadcast
        
        # 8. 將投影後的 atoms 限制在 [Vmin, Vmax] 範圍內
        Tz = torch.clamp(Tz, Vmin, Vmax)
        
        # 9. 計算投影後的 atoms 在原始 atom bins 中的位置
        b = (Tz - Vmin) / delta_z # [BATCH_SIZE, ATOMS]
        
        # 10. 找出鄰近的 bins (下界 l 和 上界 u)
        l = b.floor().long()
        u = b.ceil().long()
        
        # 11. 處理投影後剛好落在 atom 上的情況 (l == u)
        # 確保 l 和 u 不同，除非已在邊界
        l_equal_u = (l == u)
        l[l_equal_u & (u > 0)] -= 1
        u[l_equal_u & (l < (ATOMS - 1))] += 1
        
        # 12. 分配機率 (Distribution)
        #    將 p(s', a*) 的機率按比例分配給 l 和 u
        m_l = next_dist_probs * (u.float() - b)
        m_u = next_dist_probs * (b - l.float())
        
        # 13. 使用 scatter_add_ 將機率加總到 target_dist
        #     target_dist[i, l[i, j]] += m_l[i, j]
        #     target_dist[i, u[i, j]] += m_u[i, j]
        target_dist.scatter_add_(1, l, m_l)
        target_dist.scatter_add_(1, u, m_u)

    # --- 計算 Loss 並更新 Main Network (Z) ---
    
    # 14. 取得 Main Network (Z) 對 (S, A) 預測的分佈
    all_logits = Z(S).view(-1, ACT_N, ATOMS)
    
    # 15. 從 [BATCH_SIZE, ACT_N, ATOMS] 中選出實際採取的 A 對應的 logits
    A_broadcast = A.view(-1, 1, 1).expand(-1, 1, ATOMS)
    selected_logits = all_logits.gather(1, A_broadcast).squeeze(1) # [BATCH_SIZE, ATOMS]
    
    # 16. 計算 Cross-Entropy Loss
    #     Loss = - sum( target_dist * log_softmax(selected_logits) )
    #     target_dist 是 "標籤" (已是機率分佈，且 no_grad)
    #     selected_logits 是 "預測" (logits)
    log_probs = torch.log_softmax(selected_logits, dim=1)
    loss = - (target_dist * log_probs).sum(dim=1).mean()
    
    # 17. 執行反向傳播
    OPT.zero_grad()
    loss.backward()
    
    # 18. 梯度裁剪 (Gradient Clipping) - 防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(Z.parameters(), max_norm=GRAD_CLIP_NORM)
    
    OPT.step()

    # Update target network every TARGET_NETWORK_UPDATE_FREQ training steps
    if tri % TARGET_NETWORK_UPDATE_FREQ == 0:
        Zt.load_state_dict(Z.state_dict())

    return loss.item() # 回傳 loss 值以供監控


# Play episodes
# Training function
def train(seed):

    global EPSILON, Z
    print("Seed=%d" % seed)

    # Create environment, buffer, Z, Z target, optimizer
    env, test_env, buf, Z, Zt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, tri, buf, Z, Zt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 500
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std, 500), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'c51')
    plt.legend(loc='best')
    plt.show() #set KMP_DUPLICATE_LIB_OK=TRUE

# from turtle import update
# import gym
# import numpy as np
# import utils.envs, utils.seed, utils.buffers, utils.torch
# import torch
# import tqdm
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

# # C51
# # Based on Slide 11
# # cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

# # Constants
# SEEDS = [1, 2, 3, 4, 5]
# t = utils.torch.TorchHelper()
# DEVICE = t.device
# OBS_N = 4               # State space size
# ACT_N = 2               # Action space size
# STARTING_EPSILON = 1.0  # Starting epsilon
# STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
# EPSILON_END = 0.1       # At the end, keep epsilon at this value
# MINIBATCH_SIZE = 64     # How many examples to sample per train step
# GAMMA = 0.99            # Discount factor in episodic reward objective
# LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
# TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
# TRAIN_EPOCHS = 25        # Train for these many epochs every time
# BUFSIZE = 10000         # Replay buffer size
# EPISODES = 500          # Total number of episodes to learn over
# TEST_EPISODES = 10      # Test episodes
# HIDDEN = 512            # Hidden nodes
# TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# # Suggested constants
# ATOMS = 51              # Number of atoms for distributional network
# ZRANGE = [0, 200]       # Range for Z projection

# # Global variables
# EPSILON = STARTING_EPSILON
# Z = None

# # C51 globals
# atoms = None
# delta_z = None
# Vmin = None
# Vmax = None

# # Create environment
# # Create replay buffer
# # Create distributional networks
# # Create optimizer
# def create_everything(seed):
#     # 聲明我們將修改 C51 的全域變數
#     global atoms, delta_z, Vmin, Vmax

#     utils.seed.seed(seed)
#     env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)

#     test_env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)

#     buf = utils.buffers.ReplayBuffer(BUFSIZE)

#     # C51 Atom (bins) setup
#     Vmin, Vmax = ZRANGE
#     # 創建 51 個 atoms (獎勵值的 "bins")，從 Vmin 到 Vmax
#     atoms = torch.linspace(Vmin, Vmax, ATOMS).to(DEVICE)
#     # 計算每個 atom 之間的間距
#     delta_z = (Vmax - Vmin) / (ATOMS - 1)

#     Z = torch.nn.Sequential(
#         torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
#         torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
#         torch.nn.Linear(HIDDEN, ACT_N*ATOMS) # 輸出是 (動作數 * atoms數)
#     ).to(DEVICE)
#     Zt = torch.nn.Sequential(
#         torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
#         torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
#         torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
#     ).to(DEVICE)
    
#     # 初始時，將 Target 網路的權重設為與主網路相同
#     Zt.load_state_dict(Z.state_dict())
    
#     OPT = torch.optim.Adam(Z.parameters(), lr = LEARNING_RATE)
#     return env, test_env, buf, Z, Zt, OPT

# # Create epsilon-greedy policy
# def policy(env, obs):

#     # 聲明 policy 會用到的全域變數
#     global EPSILON, EPSILON_END, STEPS_MAX, Z, atoms
#     obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
#     # With probability EPSILON, choose a random action
#     # Rest of the time, choose argmax_a Q(s, a) 
#     if np.random.rand() < EPSILON:
#         action = np.random.randint(ACT_N)
#     else:
#         ## TODO: use Z to compute greedy action
#         with torch.no_grad():
#             # 1. 取得 Z-Network 的輸出 (logits)
#             # 輸出 shape: [1, ACT_N * ATOMS]
#             logits = Z(obs)
            
#             # 2. Reshape 輸出為 [1, ACT_N, ATOMS] 並轉換為機率
#             #    dim=2 表示在 ATOMS 維度上做 softmax
#             probs = torch.softmax(logits.view(1, ACT_N, ATOMS), dim=2)
            
#             # 3. 計算 Q-value (期望值)
#             #    Q(s,a) = sum( prob(s,a,z_i) * z_i )
#             #    (probs [1, ACT_N, ATOMS]) * (atoms [1, ATOMS])
#             #    廣播 (broadcasting) 會自動處理
#             q_values = (probs * atoms.unsqueeze(0)).sum(dim=2) # [1, ACT_N]
            
#             # 4. 選擇 Q-value 最高的 action
#             action = q_values.argmax(dim=1).item() # .item() 轉為 python 純量
    
#     # Epsilon update rule: Keep reducing a small amount over
#     # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
#     EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
#     return action

# # Update networks
# def update_networks(epi, buf, Z, Zt, OPT):
    
#     # 聲明 update 會用到的全域變數
#     global atoms, delta_z, Vmin, Vmax

#     ## TODO: Implement this function
    
#     # 1. 從 Replay Buffer 中取樣
#     S, A, R, S_prime, D = buf.sample(MINIBATCH_SIZE, t)
#     S = t.f(S); A = t.l(A); R = t.f(R); S_prime = t.f(S_prime); D = t.f(D)

#     with torch.no_grad():
#         # --- 建立 Target Distribution (C51 核心) ---
        
#         # 2. 使用 Target Network (Zt) 取得下個狀態 (S_prime) 的分佈
#         next_logits = Zt(S_prime).view(-1, ACT_N, ATOMS)
#         next_probs = torch.softmax(next_logits, dim=2)
        
#         # 3. 計算 Q-values (期望值) 以便找出最佳的 next_action
#         next_q_values = (next_probs * atoms.unsqueeze(0)).sum(dim=2) # [BATCH_SIZE, ACT_N]
        
#         # 4. 找出最佳的 next_action (a*)
#         next_actions = next_q_values.argmax(dim=1) # [BATCH_SIZE]
        
#         # 5. 取得 a* 對應的機率分佈 (p(s', a*))
#         batch_indices = torch.arange(MINIBATCH_SIZE, device=DEVICE).long()
#         next_dist_probs = next_probs[batch_indices, next_actions, :] # [BATCH_SIZE, ATOMS]
        
#         # 6. 建立目標分佈 (m)，大小與 next_dist_probs 相同
#         target_dist = torch.zeros_like(next_dist_probs)
        
#         # 7. Bellman 投影 (Projection)
#         #    將 atoms 投影到新的位置: Tz = R + (1-D) * gamma * z
#         R_broadcast = R.unsqueeze(1)        # [BATCH_SIZE, 1]
#         D_broadcast = D.unsqueeze(1)        # [BATCH_SIZE, 1]
#         atoms_broadcast = atoms.unsqueeze(0)  # [1, ATOMS]
        
#         Tz = R_broadcast + (1.0 - D_broadcast) * GAMMA * atoms_broadcast
        
#         # 8. 將投影後的 atoms 限制在 [Vmin, Vmax] 範圍內
#         Tz = torch.clamp(Tz, Vmin, Vmax)
        
#         # 9. 計算投影後的 atoms 在原始 atom bins 中的位置
#         b = (Tz - Vmin) / delta_z # [BATCH_SIZE, ATOMS]
        
#         # 10. 找出鄰近的 bins (下界 l 和 上界 u)
#         l = b.floor().long()
#         u = b.ceil().long()
        
#         # 11. 處理投影後剛好落在 atom 上的情況 (l == u)
#         # 這是 C51 實作中的一個重要技巧，確保機率能正確分配
#         l[(u > 0) & (l == u)] -= 1
#         u[(l < (ATOMS - 1)) & (l == u)] += 1
        
#         # 12. 分配機率 (Distribution)
#         #    將 p(s', a*) 的機率按比例分配給 l 和 u
#         m_l = next_dist_probs * (u.float() - b)
#         m_u = next_dist_probs * (b - l.float())
        
#         # 13. 使用 scatter_add_ 將機率加總到 target_dist
#         #     target_dist[i, l[i, j]] += m_l[i, j]
#         #     target_dist[i, u[i, j]] += m_u[i, j]
#         target_dist.scatter_add_(1, l, m_l)
#         target_dist.scatter_add_(1, u, m_u)

#     # --- 計算 Loss 並更新 Main Network (Z) ---
    
#     # 14. 取得 Main Network (Z) 對 (S, A) 預測的分佈
#     all_logits = Z(S).view(-1, ACT_N, ATOMS)
    
#     # 15. 從 [BATCH_SIZE, ACT_N, ATOMS] 中選出實際採取的 A 對應的 logits
#     A_broadcast = A.view(-1, 1, 1).expand(-1, 1, ATOMS)
#     selected_logits = all_logits.gather(1, A_broadcast).squeeze(1) # [BATCH_SIZE, ATOMS]
    
#     # 16. 計算 Cross-Entropy Loss
#     #     Loss = - sum( target_dist * log_softmax(selected_logits) )
#     #     target_dist 是 "標籤" (已是機率分佈，且 no_grad)
#     #     selected_logits 是 "預測" (logits)
#     log_probs = torch.log_softmax(selected_logits, dim=1)
#     loss = - (target_dist * log_probs).sum(dim=1).mean()
    
#     # 17. 執行反向傳播
#     OPT.zero_grad()
#     loss.backward()
#     OPT.step()

#     # Update target network
#     if epi%TARGET_NETWORK_UPDATE_FREQ==0:
#         Zt.load_state_dict(Z.state_dict())

#     return loss.item() # 回傳 loss 值以供監控


# # Play episodes
# # Training function
# def train(seed):

#     global EPSILON, Z
#     print("Seed=%d" % seed)

#     # Create environment, buffer, Z, Z target, optimizer
#     env, test_env, buf, Z, Zt, OPT = create_everything(seed)

#     # epsilon greedy exploration
#     EPSILON = STARTING_EPSILON

#     testRs = [] 
#     last25testRs = []
#     print("Training:")
#     pbar = tqdm.trange(EPISODES)
#     for epi in pbar:

#         # Play an episode and log episodic reward
#         S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
#         # Train after collecting sufficient experience
#         if epi >= TRAIN_AFTER_EPISODES:

#             # Train for TRAIN_EPOCHS
#             for tri in range(TRAIN_EPOCHS): 
#                 update_networks(epi, buf, Z, Zt, OPT)

#         # Evaluate for TEST_EPISODES number of episodes
#         Rews = []
#         for epj in range(TEST_EPISODES):
#             S, A, R = utils.envs.play_episode(test_env, policy, render = False)
#             Rews += [sum(R)]
#         testRs += [sum(Rews)/TEST_EPISODES]

#         # Update progress bar
#         last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
#         pbar.set_description("R25(%g)" % (last25testRs[-1]))

#     pbar.close()
#     print("Training finished!")
#     env.close()

#     return last25testRs

# # Plot mean curve and (mean-std, mean+std) curve with some transparency
# # Clip the curves to be between 0, 200
# def plot_arrays(vars, color, label):
#     mean = np.mean(vars, axis=0)
#     std = np.std(vars, axis=0)
#     plt.plot(range(len(mean)), mean, color=color, label=label)
#     plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,500), color=color, alpha=0.3)

# if __name__ == "__main__":

#     # Train for different seeds
#     curves = []
#     for seed in SEEDS:
#         curves += [train(seed)]

#     # Plot the curve for the given seeds
#     plot_arrays(curves, 'b', 'c51')
#     plt.legend(loc='best')
#     plt.show() #set KMP_DUPLICATE_LIB_OK=TRUE