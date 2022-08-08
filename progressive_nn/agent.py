import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import copy
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
        self.max_grad_norm = args['grad_clip']
        self.device = device
        self.wandb = wandb

        self.net = model
        self.net.train()
        self.target_net = target_net
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(args['memory_size'], device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args['lr'])

    def fit_model(self):
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        state_q = self.net(states)
        states_q = self.net(states_)
        states_target_q = self.target_net(states_)

        Q_exp = state_q.gather(1, actions)
        Q_target_next = states_target_q.gather(1, states_q.max(1)[1].unsqueeze(1))
        Q_targets = rewards + self.gamma * Q_target_next * (1 - dones)

        loss = F.mse_loss(Q_exp, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def soft_target_update(self):
        for target_param, net_param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau*net_param.data + (1.0 - self.tau)*target_param.data)

    def train(self):
        # t = copy.deepcopy(self.net.columns[0].adapter_segmentation.weight)
        for e in range(self.episodes):
            # print(t - self.net.columns[0].adapter_segmentation.weight)
            state = self.env.reset()
            done = False
            reward_e = 0
            ep_steps = 0
            
            while not done and ep_steps < self.episode_steps:
                if random.uniform(0,1) <= self.eps:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                        action = np.argmax(self.net(s).cpu())

                state_, reward, done, _ = self.env.step(action)
                
                self.memory.push(state.copy(), action, reward, state_.copy(), done)
                
                reward_e += reward
                ep_steps += 1
                state = state_.copy()

                # every 100 steps fit the model
                if ep_steps % 100 == 0 and len(self.memory.memory) >= self.batch_size:
                    self.fit_model()
                    self.soft_target_update()
            
            # done
            self.eps = max(self.eps_min, self.eps*self.eps_decay)

            print(f"episode: {e}, reward: {reward_e}, steps: {ep_steps}, eps: {self.eps:.5f}")

        #     self.wandb.log({
        #         "reward": reward_e,
        #         "eps": round(self.eps, 5),
        #         "cpu_score": pt[0],
        #         "agent_score": pt[1]
        #     }, step=e)
    
        #     if e > 0 and e % self.save_ckpt == 0:
        #         torch.save(self.net.state_dict(), os.path.join(self.wandb.run.dir, f'model_{e}.pt'))
        
        # torch.save(self.net.state_dict(), os.path.join(self.wandb.run.dir, f'model_final.pt'))