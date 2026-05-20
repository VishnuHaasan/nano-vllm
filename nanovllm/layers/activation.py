import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
    
class GeluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.gelu(x) * y
    
class GeluTanhAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.gelu(x, approximate="tanh")
    
ACTIVATION_MAPPING = {
    "silu": SiluAndMul,
    "gelu": GeluAndMul,
    "gelu_pytorch_tanh": GeluTanhAndMul
}
