import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torchvision.transforms as t


def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


class ReplayMemory(object):
    def __init__(self, capacity, transition_):
        self.transition = transition_
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Save transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size_):
        return random.sample(self.memory, batch_size_)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Main:
    def __init__(self, batch_size_, gamma_, eps_start_, eps_end_, eps_decay_, target_update_, num_episodes_):
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        plt.ion()
        self.resize = t.Compose([t.ToPILImage(),
                                 t.Resize(40, interpolation=Image.CUBIC),
                                 t.ToTensor()])

        self.device = torch.device('cpu')
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.batch_size = batch_size_
        self.gamma = gamma_
        self.eps_start = eps_start_
        self.eps_end = eps_end_
        self.eps_decay = eps_decay_
        self.target_update = target_update_
        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape

        self.n_actions = self.env.action_space.n
        self.policy_net = DQN(screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000, self.transition)

        self.steps_done = 0
        self.episode_durations = list()
        self.num_episodes = num_episodes_

    def get_cart_location(self, screen_width_):
        world_width = self.env.x_threshold * 2
        scale = screen_width_ / world_width
        return int(self.env.state[0] * scale + screen_width_ / 2.0)

    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        _, screen_height_, screen_width_ = screen.shape
        screen = screen[:, int(screen_height_ * 0.4):int(screen_height_ * 0.8)]
        view_width = int(screen_width_ * 0.6)
        cart_location = self.get_cart_location(screen_width_)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width_ - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        screen = screen[:, :, slice_range]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return self.resize(screen).unsqueeze(0).to(self.device)

    def select_action(self, state_):
        sample = random.random()
        eps_thresh = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_thresh:
            with torch.no_grad():
                return self.policy_net(state_).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('training')
        plt.xlabel('episode')
        plt.ylabel('duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = f.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        for i_episode in range(self.num_episodes):
            self.env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen
            for m in count():
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                last_screen = current_screen
                current_screen = self.get_screen()

                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()
                if done:
                    self.episode_durations.append(m + 1)
                    self.plot_durations()
                    break
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    batch_size = 128
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10
    num_episodes = 5000

    trainer = Main(batch_size, gamma, eps_start, eps_end, eps_decay, target_update, 5000)
    trainer.train()
