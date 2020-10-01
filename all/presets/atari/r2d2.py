from torch.nn.functional import relu
from torch.optim import Adam
from all.agents import R2D2
from all.approximation import Approximation, FixedTarget
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.optim import LinearScheduler
from .models import nature_recurrent_dqn


def r2d2(
        # Common settings
        device="cuda",
        discount_factor=0.997,
        last_frame=40e6,
        # Adam optimizer settings
        lr=1e-4,
        eps=1e-3,
        # Training settings
        minibatch_size=64,
        update_frequency=4,
        target_update_frequency=500,
        # Replay buffer settings
        replay_start_size=80000,
        replay_buffer_size=1000000,
        # Recurrent settings
        rollout_len=40,
        # Exploration settings
        initial_exploration=1.,
        final_exploration=0.01,
        final_exploration_frame=1000000,
        # parallel settings
        n_envs=32,
        # network settings
        model_constructor=nature_recurrent_dqn,
):
    def _r2d2(env, writer=DummyWriter()):
        action_repeat = 4
        last_timestep = last_frame / action_repeat
        last_update = (last_timestep - replay_start_size) / update_frequency
        final_exploration_step = final_exploration_frame / action_repeat

        model = model_constructor(env[0]).to(device)
        optimizer = Adam(model.parameters(), lr=lr, eps=eps)
        q_rnn = Approximation(model, optimizer, target=FixedTarget(target_update_frequency), writer=writer)
        return DeepmindAtariBody(
            R2D2(
                q_rnn,
                exploration=LinearScheduler(
                    initial_exploration,
                    final_exploration,
                    replay_start_size // n_envs,
                    (final_exploration_step - replay_start_size) // n_envs,
                    name="exploration",
                    writer=writer
                ),
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                rollout_len=rollout_len,
                update_frequency=update_frequency,
                writer=writer,
            )
        )
    return _r2d2, n_envs


__all__ = ["r2d2"]
