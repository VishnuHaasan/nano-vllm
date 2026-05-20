from nanovllm.config import Config
from nanovllm.models.gemma2 import Gemma2ForCausalLM
from nanovllm.models.llama3 import Llama3ForCausalLM
from nanovllm.models.mistral import MistralForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM

MODEL_REGISTRY = {
    "LlamaForCausalLM": Llama3ForCausalLM,
    "MistralForCausalLM": MistralForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Gemma2ForCausalLM": Gemma2ForCausalLM
}

def get_model(config : Config):

    model_name = config.hf_config.architectures[0]
    return MODEL_REGISTRY[model_name](config.hf_config)
