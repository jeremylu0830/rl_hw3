import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
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

# Global variables
EPSILON = STARTING_EPSILON
Q = None
hidden_state = None

# Deep recurrent Q network
class DRQN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        # Input layer: from observation space to hidden
        self.fc1 = torch.nn.Linear(OBS_N, HIDDEN)
        
        # LSTM layer
        self.lstm = torch.nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
        
        # Output layer: from hidden to action space
        self.fc2 = torch.nn.Linear(HIDDEN, ACT_N)
    
    def forward(self, x, hidden=None):
        """
        x: input tensor of shape (batch, seq_len, OBS_N) or (batch, OBS_N)
        hidden: tuple of (h, c) for LSTM hidden state
        """
        # Handle both batched sequences and single observations
        if len(x.shape) == 2:
            # Single timestep: (batch, OBS_N) -> (batch, 1, OBS_N)
            x = x.unsqueeze(1)
        
        # First fully connected layer with ReLU
        x = torch.relu(self.fc1(x))
        
        # LSTM layer
        if hidden is None:
            x, hidden_out = self.lstm(x)
        else:
            x, hidden_out = self.lstm(x, hidden)
        
        # Output layer to get Q-values
        q_values = self.fc2(x)
        
        return q_values, hidden_out
    
    def init_hidden(self, batch_size=1):
        """Initialize hidden state for LSTM"""
        h = torch.zeros(1, batch_size, HIDDEN).to(DEVICE)
        c = torch.zeros(1, batch_size, HIDDEN).to(DEVICE)
        return (h, c)

# Create environment
# Create replay buffer
# Create network for Q(s, a)
# Create target network
# Create optimizer
def create_everything(seed):
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.PartiallyObservableCartPole(), 200)

    test_env = utils.envs.TimeLimit(utils.envs.PartiallyObservableCartPole(), 200)

    buf = utils.buffers.ReplayBuffer(BUFSIZE, recurrent=True)
    Q = DRQN().to(DEVICE)
    Qt = DRQN().to(DEVICE)
    Qt.load_state_dict(Q.state_dict())  # Initialize target network
    OPT = torch.optim.Adam(Q.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT

# Create epsilon-greedy policy
def policy(env, obs):
    global EPSILON, EPSILON_END, STEPS_MAX, Q, hidden_state
    
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor

    # Initialize hidden state if needed
    if hidden_state is None:
        hidden_state = Q.init_hidden(batch_size=1)
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        # Greedy policy: choose action with highest Q-value
        with torch.no_grad():
            q_values, hidden_state = Q(obs, hidden_state)
            action = q_values.squeeze().argmax().item()
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action


# Update networks
def update_networks(epi, buf, Q, Qt, OPT):
    """
    Sample a minibatch from replay buffer and update Q network
    """
    # Sample episodes from replay buffer
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    
    # S, S2 shape: (batch_size, seq_len, OBS_N)
    # A, R, D shape: (batch_size, seq_len)
    
    batch_size = S.shape[0]
    seq_len = S.shape[1]
    
    # Compute current Q values
    # Initialize hidden states for the batch
    hidden = Q.init_hidden(batch_size)
    q_values, _ = Q(S, hidden)  # Shape: (batch_size, seq_len, ACT_N)
    
    # Gather Q values for taken actions
    # Need to expand A to match q_values shape
    A_expanded = A.unsqueeze(-1)  # (batch_size, seq_len, 1)
    current_q = q_values.gather(2, A_expanded).squeeze(-1)  # (batch_size, seq_len)
    
    # Compute target Q values
    with torch.no_grad():
        hidden_target = Qt.init_hidden(batch_size)
        next_q_values, _ = Qt(S2, hidden_target)  # Shape: (batch_size, seq_len, ACT_N)
        max_next_q = next_q_values.max(dim=2)[0]  # (batch_size, seq_len)
        
        # Target: R + gamma * max_a' Q_target(s', a') * (1 - done)
        target_q = R + GAMMA * max_next_q * (1 - D)
    
    # Compute loss (MSE)
    loss = torch.nn.functional.mse_loss(current_q, target_q)
    
    # Optimize
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    # Update target network periodically
    if epi % TARGET_NETWORK_UPDATE_FREQ == 0:
        Qt.load_state_dict(Q.state_dict())

    return loss.item()


# Play episodes
# Training function
def train(seed):

    global EPSILON, Q, hidden_state
    print("Seed=%d" % seed)

    # Create environment, buffer, Q, Q target, optimizer
    env, test_env, buf, Q, Qt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Reset hidden state at the beginning of each episode
        hidden_state = None
        
        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Q, Qt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        if epi % 10 == 0: 
            Rews = []
            for epj in range(TEST_EPISODES):
                hidden_state = None 
                S, A, R = utils.envs.play_episode(test_env, policy, render = False)
                Rews += [sum(R)]
            testRs += [sum(Rews)/TEST_EPISODES]
            
            # 確保 testRs 不為空再計算平均
            if testRs: 
                last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]

        if last25testRs: 
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
    plot_arrays(curves, 'b', 'DRQN')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (25-episode rolling average)')
    plt.title('DRQN Performance on Partially Observable CartPole')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig('drqn_claude_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
