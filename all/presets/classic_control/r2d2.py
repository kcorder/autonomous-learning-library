import torch
from torch import nn
from torch.nn.functional import relu
from torch.optim import Adam
from all.agents import R2D2
from all.approximation import Approximation
from all.logging import DummyWriter
from all.optim import LinearScheduler

class RecurrentQ(nn.Module):
    def __init__(self, env, hidden=256):
        super().__init__()
        self.encoder = nn.Linear(env.state_space.shape[0], hidden)
        self.lstm = nn.LSTMCell(hidden, hidden)
        self.decoder = nn.Linear(hidden, env.action_space.n)
        self.hidden = hidden

    def forward(self, state, hc=None):
        if hc is None:
            h = torch.zeros((state.shape[-1], self.hidden), device=state.device)
            c = torch.zeros((state.shape[-1], self.hidden), device=state.device)
        else:
            h, c = hc
        if len(state.shape) == 1:
            encoded = relu(state.as_output(self.encoder(state.as_input('observation'))))
            _h, _c = self.lstm(encoded, (h, c))
            q_values = state.as_output(self.decoder(_h.view(-1, self.hidden)))
            return state.apply_mask(state.as_output(q_values)), (state.apply_mask(_h), state.apply_mask(_c))

        encoded = relu(state.as_output(self.encoder(state.as_input('observation'))))
        latent = []
        for t in range(state.shape[0]):
            h, c = self.lstm(encoded[t], (h, c))
            h, c = h * state.mask[t].unsqueeze(-1), c * state.mask[t].unsqueeze(-1)
            latent.append(h)
        latent = torch.stack(latent)
        t, b = state.shape
        values = self.decoder(latent.view(-1, self.hidden)).view((t, b, -1))
        return state.apply_mask(values), (h, c)

def r2d2(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr=7e-4,
        # Training settings
        minibatch_size=32,
        update_frequency=5,
        # Exploration settings
        initial_exploration=1.,
        final_exploration=0.01,
        final_exploration_frame=1000,
        # parallel settings
        n_envs=32,
        # network settings
        hidden=256,
):
    def _r2d2(env, writer=DummyWriter()):
        model = RecurrentQ(env[0], hidden=hidden).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        q_rnn = Approximation(model, optimizer, writer=writer)
        return R2D2(
            q_rnn,
            exploration=LinearScheduler(
                initial_exploration,
                final_exploration,
                0,
                final_exploration_frame,
                name="epsilon",
                writer=writer
            ),
            discount_factor=discount_factor,
            minibatch_size=minibatch_size,
            rollout_len=20,
            update_frequency=update_frequency,
        )
    return _r2d2, n_envs


__all__ = ["r2d2"]
