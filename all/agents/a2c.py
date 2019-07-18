import torch
from all.environments import State
from all.memory import NStepAdvantageBuffer
from .abstract import Agent


class A2C(Agent):
    def __init__(
            self,
            features,
            v,
            policy,
            n_envs=None,
            n_steps=4,
            discount_factor=0.99
    ):
        if n_envs is None:
            raise RuntimeError("Must specify n_envs.")
        self.features = features
        self.v = v
        self.policy = policy
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer()
        self._features = []

    def act(self, states, rewards):
        self._buffer.store(states, torch.zeros(self.n_envs), rewards)
        self._train()
        features = self.features(states)
        self._features.append(features)
        return self.policy(features)

    def _train(self):
        if len(self._buffer) >= self._batch_size:
            states = State.from_list(self._features)
            _, _, advantages = self._buffer.sample(self._batch_size)
            self.v(states)
            self.v.reinforce(advantages)
            self.policy.reinforce(advantages)
            self.features.reinforce()
            self._features = []

    def _make_buffer(self):
        return NStepAdvantageBuffer(
            self.v,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor
        )
 