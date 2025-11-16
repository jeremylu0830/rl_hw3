import collections
import numpy as np
import random
import torch

# Replay buffer
class ReplayBuffer:
    
    # create replay buffer of size N
    def __init__(self, N, recurrent=False):
        self.buf = collections.deque(maxlen=N)
        self.recurrent = recurrent
        self.temp_episode = {'s': [], 'a': [], 'r': [], 's2': [], 'd': []}
    
    # add: add a transition (s, a, r, s2, d) or an episode
    def add(self, s, a, r, s2, d):
        if self.recurrent:
            # For recurrent networks, we'll collect transitions first
            # and store the complete episode when episode ends
            self.temp_episode['s'].append(s)
            self.temp_episode['a'].append(a)
            self.temp_episode['r'].append(r)
            self.temp_episode['s2'].append(s2)
            self.temp_episode['d'].append(d)
            
            # If episode is done, store it
            if d:  # d=1 means episode ended
                self.buf.append((
                    np.array(self.temp_episode['s']),
                    np.array(self.temp_episode['a']),
                    np.array(self.temp_episode['r']),
                    np.array(self.temp_episode['s2']),
                    np.array(self.temp_episode['d'])
                ))
                # Reset temporary episode storage
                self.temp_episode = {'s': [], 'a': [], 'r': [], 's2': [], 'd': []}
        else:
            # For standard DQN, store individual transitions
            self.buf.append((s, a, r, s2, d))
    
    # sample: return minibatch of size n
    def sample(self, n, t):
        if self.recurrent:
            return self._sample_episodes(n, t)
        else:
            return self._sample_transitions(n, t)
    
    def _sample_transitions(self, n, t):
        """Sample individual transitions (for standard DQN)"""
        minibatch = random.sample(self.buf, n)
        S, A, R, S2, D = [], [], [], [], []
        
        for mb in minibatch:
            s, a, r, s2, d = mb
            S += [s]; A += [a]; R += [r]; S2 += [s2]; D += [d]

        if type(A[0]) == int:
            return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)
        elif type(A[0]) == float:
            return t.f(S), t.f(A), t.f(R), t.f(S2), t.i(D)
        else:
            return t.f(S), torch.stack(A), t.f(R), t.f(S2), t.i(D)
    
    def _sample_episodes(self, n, t):
        """Sample episodes (for recurrent networks like DRQN)"""
        # Sample n episodes from buffer
        minibatch = random.sample(self.buf, min(n, len(self.buf)))
        
        # Find the maximum episode length for padding
        max_len = max(len(episode[0]) for episode in minibatch)
        
        batch_size = len(minibatch)
        
        # Get observation dimension from first episode
        obs_dim = minibatch[0][0].shape[1] if len(minibatch[0][0].shape) > 1 else len(minibatch[0][0][0])
        
        # Initialize arrays for padded sequences
        S = np.zeros((batch_size, max_len, obs_dim))  # (batch, seq_len, obs_dim)
        A = np.zeros((batch_size, max_len), dtype=np.int64)
        R = np.zeros((batch_size, max_len))
        S2 = np.zeros((batch_size, max_len, obs_dim))
        D = np.zeros((batch_size, max_len))
        
        # Fill in the sequences
        for i, (s, a, r, s2, d) in enumerate(minibatch):
            ep_len = len(s)
            S[i, :ep_len] = s
            A[i, :ep_len] = a
            R[i, :ep_len] = r
            S2[i, :ep_len] = s2
            D[i, :ep_len] = d
        
        # Convert to torch tensors
        S = torch.FloatTensor(S).to(t.device)
        A = torch.LongTensor(A).to(t.device)
        R = torch.FloatTensor(R).to(t.device)
        S2 = torch.FloatTensor(S2).to(t.device)
        D = torch.FloatTensor(D).to(t.device)
        
        return S, A, R, S2, D