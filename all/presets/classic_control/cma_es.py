import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import CMA_ES, CMA_ES_TestAgent
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
from all.presets.classic_control.models import fc_sin_policy


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Optimization settings
    "population_size": 100,
    "lr": 0.2,
    "std": 0.1,
    # Model construction
    "policy_model_constructor": fc_sin_policy
}


class CMA_ES_ClassicControlPreset(Preset):
    """
    Covariance Matrix Adaptation Evolutionary Strategies (CMA-ES) Classic Control preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        TODO
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.policy_model = hyperparameters['policy_model_constructor'](env).to(device)

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        policy = SoftmaxPolicy(self.policy_model, None)
        return CMA_ES(
            policy,
            discount_factor=self.hyperparameters["discount_factor"],
            population_size=self.hyperparameters["population_size"],
            lr=self.hyperparameters["lr"],
            std=self.hyperparameters["std"]
        )

    def test_agent(self):
        return CMA_ES_TestAgent(SoftmaxPolicy(copy.deepcopy(self.policy_model), None))


cma_es = PresetBuilder('cma_es', default_hyperparameters, CMA_ES_ClassicControlPreset)
