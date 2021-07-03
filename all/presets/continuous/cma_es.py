import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import CMA_ES, CMA_ES_TestAgent
from all.logging import DummyWriter
from all.policies import GaussianPolicy
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
from all.presets.continuous.models import fc_policy


default_hyperparameters = {
    # Optimization settings
    "population_size": 100,
    "lr": 2.,
    "std": 0.1,
    # Model construction
    "policy_model_constructor": fc_policy
}


class CMA_ES_ContinuousPreset(Preset):
    """
    Covariance Matrix Adaptation Evolutionary Strategies (CMA-ES) Classic Control preset.

    Args:
        env (all.environments.ContinuousEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        TODO
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.policy_model = hyperparameters['policy_model_constructor'](env, time_feature=False).to(device)
        self.action_space = env.action_space

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        policy = GaussianPolicy(self.policy_model, space=self.action_space)
        return CMA_ES(
            policy,
            population_size=self.hyperparameters["population_size"],
            lr=self.hyperparameters["lr"],
            std=self.hyperparameters["std"]
        )

    def test_agent(self):
        return CMA_ES_TestAgent(GaussianPolicy(copy.deepcopy(self.policy_model), space=self.action_space))


cma_es = PresetBuilder('cma_es', default_hyperparameters, CMA_ES_ContinuousPreset)
