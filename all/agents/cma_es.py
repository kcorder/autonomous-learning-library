import numpy as np
import torch
from torch.nn.functional import mse_loss
from ._agent import Agent


class CMA_ES(Agent):
    def __init__(
        self,
        policy,
        discount_factor=0.99,
        population_size=50,
        std=0.1,
        lr=0.001,
    ):
        self.policy = policy
        self.discount_factor = discount_factor
        self.population_size = population_size
        self.std = std
        self.lr = lr
        self._parameters = list([p for p in self.policy.model.parameters() if p.requires_grad])
        self._generate_population()
        self._load_next_policy()

    def act(self, state):
        self._record_reward(state)
        if state.done:
            if self._should_update():
                self._update_population()
            self._load_next_policy()
        return self.policy.no_grad(state).sample()

    def _generate_population(self):
        self._initial = list([torch.clone(p.data) for p in self._parameters])
        base_noise = [self.std * torch.randn([self.population_size // 2, *p.data.shape], device=p.data.device) for p in self._parameters]
        self._perturbations = [torch.cat((noise, -noise), dim=0) for noise in base_noise]
        self._i = -1
        self._returns = np.zeros(self.population_size)

    def _load_next_policy(self):
        self._i += 1
        with torch.no_grad():
            [p.copy_(initial + perturbation[self._i]) for p, initial, perturbation in zip(self._parameters, self._initial, self._perturbations)]

    def _record_reward(self, state):
        self._returns[self._i] += state.reward

    def _should_update(self):
        return self._i >= self.population_size - 1

    def _update_population(self):
        with torch.no_grad():
            returns = torch.tensor(self._returns, device=self.policy.device).float()
            if returns.std() > 0:
                normalized = (returns - returns.mean()) / returns.std()
                new_parameters = [initial + self.lr / (self.population_size * self.std) * torch.tensordot(normalized, perturbation, dims=1) for initial, perturbation in zip(self._initial, self._perturbations)]
                [p.copy_(new) for p, new in zip(self._parameters, new_parameters)]
        self._generate_population()

class CMA_ES_TestAgent(CMA_ES):
    def __init__(self, policy):
        self.policy = policy

    def act(self, state):
        return self.policy.eval(state).sample()
