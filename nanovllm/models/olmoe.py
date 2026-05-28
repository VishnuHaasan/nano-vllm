from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from transformers import OlmoeConfig
import torch.distributed as dist
from collections.abc import Sequence

from nanovllm.layers.activation import ACTIVATION_MAPPING
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear, RowParallelLinear, divide
from nanovllm.layers.rotary_embedding import get_rope

def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

class OlmoeExpertMLP(nn.Module):

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
    
class OlmoeMoE(nn.Module):

    def __init__(
        self,
        config: OlmoeConfig
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = ReplicatedLinear(config.hidden_size, self.num_experts)
        self.experts = nn.ModuleList([
            OlmoeExpertMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
            for _ in range(self.num_experts)
        ])

    def forward(
        self,
        x
    ):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        top_w, top_idx = probs.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
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
    
class OlmoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % self.tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads
        )

        self.q_norm = RMSNorm(self.total_num_heads * self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.total_num_kv_heads * self.head_dim, eps=rms_norm_eps)

    def _apply_qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q_gather = [torch.empty_like(q) for _ in range(self.tp_size)]
            k_gather = [torch.empty_like(k) for _ in range(self.tp_size)]
            dist.all_gather(q_gather, q.contiguous())
            dist.all_gather(k_gather, k.contiguous())
            q = torch.cat(q_gather, dim=-1)
            k = torch.cat(k_gather, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank].contiguous()
            k = splitter(k)[self.tp_rank].contiguous()
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
    
class OlmoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OlmoeConfig
    ) -> None:
        super().__init__()
        self.self_attn = OlmoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
        )
        self.mlp = OlmoeMoE(
            config=config
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    
class OlmoeModel(nn.Module):

    def __init__(
        self,
        config: OlmoeConfig
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OlmoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    
class OlmoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: OlmoeConfig
    ) -> None:
        super().__init__()
        self.model = OlmoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids, positions)
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)