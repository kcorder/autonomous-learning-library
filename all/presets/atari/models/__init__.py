import numpy as np
import torch
from torch.nn.functional import relu
from all import nn

def nature_dqn(env, frames=4):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n)
    )

def nature_ddqn(env, frames=4):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dueling(
            nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear0(512, 1)
            ),
            nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear0(512, env.action_space.n)
            ),
        )
    )

def nature_features(frames=4):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
    )

def nature_value_head():
    return nn.Linear(512, 1)

def nature_policy_head(env):
    return nn.Linear0(512, env.action_space.n)

def nature_c51(env, frames=4, atoms=51):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n * atoms)
    )

def nature_rainbow(env, frames=4, hidden=512, atoms=51, sigma=0.5):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.CategoricalDueling(
            nn.Sequential(
                nn.NoisyFactorizedLinear(3136, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            ),
            nn.Sequential(
                nn.NoisyFactorizedLinear(3136, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    env.action_space.n * atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            )
        )
    )

class RecurrentDQN(nn.Module):
    def __init__(self, encoder, lstm, decoder, hidden):
        super().__init__()
        self.encoder = encoder
        self.lstm = lstm
        self.decoder = decoder
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

def nature_recurrent_dqn(env, hidden=512):
    encoder = nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(4, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    lstm = nn.LSTMCell(3136, hidden)
    decoder = nn.Dueling(
        nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear0(hidden, 1)
        ),
        nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear0(hidden, env.action_space.n)
        ),
    )
    return RecurrentDQN(encoder, lstm, decoder, hidden)
