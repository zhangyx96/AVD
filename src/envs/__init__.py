from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from MPE.mpe import make_parallel_env
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# REGISTRY["MPE"] = make_parallel_env('simple_spread', 1, 1000, True)
REGISTRY["MPE"] = partial(env_fn, env=make_parallel_env)


#env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,config.discrete_action)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH","~/StarCraftII")
