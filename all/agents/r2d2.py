import torch
from torch.nn.functional import mse_loss
import numpy as np
from all.core import State
from ._agent import Agent



class R2D2(Agent):
    # TODO:
    # [ ] decorrelate rollouts
    # [ ] target network
    # [ ] n-step
    # [ ] value function rescaling
    # [ ] no reward clipping
    # [ ] store hidden state
    def __init__(self,
                    q_rnn,
                    discount_factor=0.99,
                    exploration=0.02,
                    minibatch_size=32,
                    rollout_len=40,
                    update_frequency=10,
                    replay_buffer_size=100000,
                    writer=None,
                 ):
        # objects
        self.q_rnn = q_rnn
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.exploration = exploration
        self.rollout_len = 20
        self.update_frequency = update_frequency
        self.writer = writer
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0
        # buffer
        self.replay_buffer_size = replay_buffer_size
        self._warmup_temp_buffer = []
        self._train_temp_buffer = []
        self._buffer = []
        self.h = None
        self.c = None
        self.eval_h = None
        self.eval_c = None

    def act(self, state):
        self._train()
        if self.h is None:
            q, (self.h, self.c) = self.q_rnn.no_grad(state)
        else:
            q, (self.h, self.c) = self.q_rnn.no_grad(state, (self.h, self.c))
        action = self._choose_action(q)
        self._store(state, action)
        return action

    def _choose_action(self, q):
        best_actions = torch.argmax(q, dim=-1)
        random_actions = torch.randint(0, q.shape[-1], best_actions.shape, device=best_actions.device)
        choices = (torch.randn(best_actions.shape, device=best_actions.device) < self.exploration).int()
        return choices * random_actions + (1 - choices) * best_actions

    def eval(self, state):
        if self.h is None:
            q, (self.eval_h, self.eval_c) = self.q_rnn.eval(state)
        else:
            q, (self.eval_h, self.eval_c) = self.q_rnn.eval(state, (self.h, self.c))
        return self._choose_action(q)

    def _train(self):
        if self._should_train():
            warmup_states, train_states = self._sample()
            # forward pass
            h = None
            c = None
            _, (h, c) = self.q_rnn.no_grad(warmup_states)
            values, (h, c) = self.q_rnn(train_states, (h, c))
            q_values = values.gather(2, train_states['action'].unsqueeze(-1)).squeeze(2)
            # compute_targets
            targets = train_states['reward'][:-1] + self.discount_factor * q_values[1:].detach()
            # compute loss
            loss = mse_loss(q_values[0:-1], targets)
            # backward pass
            self.q_rnn.reinforce(loss)
            # info
            self.writer.add_loss('q_mean', q_values.mean().item())

    def _should_train(self):
        self._frames_seen += 1
        return (len(self._buffer) > self.minibatch_size and
                self._frames_seen % self.update_frequency == 0)

    def _store(self, state, action):
        state = state.update('action', action)

        if len(self._warmup_temp_buffer) < self.rollout_len:
            self._warmup_temp_buffer.append(state)
            if len(self._warmup_temp_buffer) == self.rollout_len:
                self._warmup_temp_buffer = State.array(self._warmup_temp_buffer)
            return
        if len(self._train_temp_buffer) < self.rollout_len:
            self._train_temp_buffer.append(state)
            return
        if len(self._train_temp_buffer) == self.rollout_len:
            _train_temp_buffer = State.array(self._train_temp_buffer)
            # TODO need to decouple states
            self._buffer.append((self._warmup_temp_buffer, _train_temp_buffer))
            self._warmup_temp_buffer = _train_temp_buffer
            self._train_temp_buffer = []
            if len(self._buffer) * self.rollout_len * state.shape[0] > self.replay_buffer_size:
                self._buffer.pop()

    def _sample(self):
        keys = np.random.choice(len(self._buffer), self.minibatch_size, replace=True)
        minibatch = [self._buffer[key] for key in keys]
        warmup_states = [x[0] for x in minibatch]
        train_states = [x[1] for x in minibatch]
        return warmup_states[0], train_states[0]
        # return State.from_list(warmup_states), State.from_list(train_states)
