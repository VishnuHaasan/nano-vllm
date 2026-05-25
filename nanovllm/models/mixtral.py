import torch
from torch import nn
from torch.nn import functional as F
from transformers import MixtralConfig
from nanovllm.layers.activation import ACTIVATION_MAPPING
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from nanovllm.models.mistral import MistralAttention

class MixtralExpertMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str
    ) -> None:
        
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False
        )
        self.w2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False
        )
        self.act_fn = ACTIVATION_MAPPING[hidden_act]()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.w2(x)
        return x

class MixtralMoE(nn.Module):

    def __init__(
        self,
        config: MixtralConfig
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.gate = ReplicatedLinear(config.hidden_size, self.num_experts)
        self.experts = nn.ModuleList([
            MixtralExpertMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
            for _ in range(self.num_experts)
        ])

    def forward(
        self,
        x
    ):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        top_w, top_idx = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        top_w = top_w.to(x.dtype)

        final = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (top_idx == e)
            if not mask.any():
                continue
            tok_idx, slot_idx = mask.nonzero(as_tuple=True)
            y = self.experts[e](x[tok_idx]) * top_w[tok_idx, slot_idx, None]
            final.index_add_(0, tok_idx, y)
        return final
    
class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig
    ) -> None:
        super().__init__()
        self.self_attn = MistralAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
        )
        self.block_sparse_moe = MixtralMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual
    
class MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MixtralDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    
class MixtralForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "w1": ("gate_up_proj", 0),
        "w3": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: MixtralConfig
    ) -> None:
        super().__init__()
        self.model = MixtralModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states) 