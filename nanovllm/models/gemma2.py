from torch import nn
import torch
from transformers import Gemma2Config
import torch.distributed as dist

from nanovllm.layers.activation import ACTIVATION_MAPPING
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import Gemma2RMSNorm
from nanovllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope

class Gemma2MLP(nn.Module):
    
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
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False
        )
        self.act_fn = ACTIVATION_MAPPING[hidden_act]()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
    
class Gemma2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_idx: int,
        max_position: int = 8192,
        head_dim: int | None = None,
        rope_theta: float = 10000,
        attn_logit_softcapping: float = 50,
        sliding_window: int = 4096,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        window = sliding_window if layer_idx % 2 != 0 else -1
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            softcap=attn_logit_softcapping,
            sliding_window=window
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

class Gemma2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma2Config,
        layer_idx: int
    ) -> None:
        super().__init__()
        self.self_attn = Gemma2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            attn_logit_softcapping=config.attn_logit_softcapping,
            sliding_window=config.sliding_window,
            layer_idx=layer_idx,
        )
        self.mlp = Gemma2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
    
class Gemma2ScaledEmbedding(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float
    ) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = super().forward(x)
        hidden_states = hidden_states * self.embed_scale
        return hidden_states
    
class Gemma2Model(nn.Module):

    def __init__(
        self,
        config: Gemma2Config
    ) -> None:
        super().__init__()
        self.embed_tokens = Gemma2ScaledEmbedding(config.vocab_size, config.hidden_size, config.hidden_size**0.5)
        self.layers = nn.ModuleList([Gemma2DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
class Gemma2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Gemma2Config
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma2Model(config)
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
        logits = self.lm_head(hidden_states) 
        if self.config.final_logit_softcapping:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits
