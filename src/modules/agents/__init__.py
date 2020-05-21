REGISTRY = {}

from .rnn_agent import RNNAgent
from .attention_layer import Attention_Layer
REGISTRY["rnn"] = RNNAgent
REGISTRY["attention_layer"] = Attention_Layer