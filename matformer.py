import torch
import torch.nn as nn

def print_grad(grad):
    print(f"Number of zeros in the gradient: {(grad == 0).sum().item()} out of {grad.numel()}")

class LinearMatryoshka(nn.Module):

    def __init__(self, in_size: int, out_size: int, matryoshka_depth: int, matryoshka_axis: int = 0, device="cuda"):
        super().__init__()
        self.grans = [i for i in range(matryoshka_depth)]
        matryoshka_len = in_size if matryoshka_axis == 0 else out_size
        self.matryoshka_sizes = [matryoshka_len // (2 ** g) for g in self.grans[1:]]
        self.submodels = nn.ModuleList()
        self.main_model = nn.Linear(in_size, out_size).to(device)
        self.matryoshka_axis = matryoshka_axis
        for i in self.matryoshka_sizes:
            if matryoshka_axis == 0:
                submodel = nn.Linear(i, out_size)
            elif matryoshka_axis == 1:
                submodel = nn.Linear(in_size, i)
            else:
                raise ValueError(f"Invalid matryoshka_axis {matryoshka_axis}. Must be 0 or 1.")

            del submodel._parameters['weight']
            del submodel._parameters['bias']
            if matryoshka_axis == 0:
                submodel.weight = self.main_model.weight[:, :i]
                submodel.bias = self.main_model.bias
            elif matryoshka_axis == 1:
                submodel.weight = self.main_model.weight[:i, :]
                submodel.bias = self.main_model.bias[:i]

            self.submodels.append(submodel)
        # self.main_model.weight.register_hook(print_grad)

    def forward(self, x: torch.Tensor, granularity: int) -> torch.Tensor:
        if granularity not in self.grans:
            raise ValueError(f"Invalid granularity {granularity}. Must be one of {self.grans}.")
        if granularity == 0:
            return self.main_model(x)
        else:
            submodel = self.submodels[granularity - 1]
            if self.matryoshka_axis == 0:
                return submodel(x[:, :self.matryoshka_sizes[granularity - 1]])
            else:
                return submodel(x)