from typing import Union

from .fql import FQLAgent, FQLCfg
from .ifql import IFQLAgent, IFQLCfg
from .dsrl import DSRLAgent, DSRLCfg
from .reform import ReFORMAgent, ReFORMCfg

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    dsrl=DSRLAgent,
    reform=ReFORMAgent,
)

agent_cfgs = dict(
    fql=FQLCfg,
    ifql=IFQLCfg,
    dsrl=DSRLCfg,
    reform=ReFORMCfg,
)

Agent = Union[FQLAgent, IFQLAgent, DSRLAgent, ReFORMAgent]
AgentCfg = Union[FQLCfg, IFQLCfg, DSRLCfg, ReFORMCfg]
