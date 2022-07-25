import torch
import torch.nn.functional as F
import numpy as np
import random
import time
import os
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_', 'done'))

class ReplayMemory():
    def __init__(self, capacity, device='cpu'):
        self.memory = deque([], maxlen=capacity)
        self.experience = namedtuple('experience', ('state', 'action', 'reward', 'state_', 'done'))
        self.device = device

    def push(self, state, action, reward, state_, done):
        e = self.experience(state.copy(), action, reward, state_.copy(), done)
        self.memory.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.state_ for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(next_states.shape)
        # print(dones.shape)

        return (states, actions, rewards, next_states, dones)

class Agent():
    def __init__(self, env, args, model, target_net, wandb, device='cpu'):
        self.env = env
        self.batch_size = args['batch_size']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.lr = args['lr']
        self.eps = args['eps']
        self.eps_decay = args['eps_decay']
        self.eps_min = args['eps_min']
        self.episodes = args['max_episodes']
        self.episode_steps = args['max_ep_step']
        self.save_ckpt = args['save_ckpt']
        self.device = device
        self.wandb = wandb

        self.net = model
        self.target_net = target_net
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(args['memory_size'], device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args['lr'])

    def fit_model(self):
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        Q_target_next = self.target_net(states_).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.gamma * Q_target_next * (1 - dones)
        Q_exp = self.net(states).gather(1, actions)

        loss = F.mse_loss(Q_exp, Q_targets)
        loss.backward()
        self.optimizer.step()

    def soft_target_update(self):
        for target_param, net_param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau*net_param.data + (1.0 - self.tau)*target_param.data)

    def train(self):
        for e in range(self.episodes):
            self.optimizer.zero_grad()

            state = self.env.reset()
            done = False
            reward_e = 0
            ep_steps = 0
            pt = [0, 0]
            
            while not done and ep_steps < self.episode_steps:
                if random.random() <= self.eps:
                    action = self.env.action_space.sample()
                else:
                    self.net.eval()
                    with torch.no_grad():
                        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                        action = np.argmax(self.net(s).cpu())
                    self.net.train()

                state_, reward, done, _ = self.env.step(action)
                
                if not done:
                    if ep_steps + 1 == self.episode_steps:
                        self.memory.push(state.copy(), action, -1, state_.copy(), 1)
                    else:
                        self.memory.push(state.copy(), action, reward, state_.copy(), done)
                else:
                    self.memory.push(state.copy(), action, reward, state_.copy(), done)
                reward_e += reward
                if reward < 0:
                    pt[0] += 1
                if reward > 0:
                    pt[1] += 1
                ep_steps += 1
                state = state_.copy()

            self.eps = max(self.eps_min, self.eps*self.eps_decay)

            if len(self.memory.memory) >= self.batch_size:
                self.fit_model()
                self.soft_target_update()

            # TODO: log to wandb
            # print(f"episode: {e}, reward: {reward_e}, steps: {ep_steps}, eps: {self.eps:.5f}, score: {pt}")

            self.wandb.log({
                "reward": reward_e,
                "eps": round(self.eps, 5),
                "cpu_score": pt[0],
                "agent_score": pt[1]
            }, step=e)
    
            if e > 0 and e % self.save_ckpt == 0:
                torch.save(self.net.state_dict(), os.path.join(self.wandb.run.dir, f'model_{e}.pt'))
        
        torch.save(self.net.state_dict(), os.path.join(self.wandb.run.dir, f'model_final.pt'))