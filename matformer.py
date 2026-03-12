import torch
import torch.nn as nn

class LinearMatryoshka(nn.Module):

    def __init__(self, in_size: int, out_size: int, num_granularities: int, matryoshka_axis: int = 0):
        super().__init__()
        self.grans = [i for i in range(num_granularities)]
        self.matryoshka_sizes = [in_size // (2 ** g) for g in self.grans]
        self.submodels = []
        self.main_model = nn.Linear(in_size, out_size)
        self.matryoshka_axis = matryoshka_axis
        for i in self.matryoshka_sizes:
            if matryoshka_axis == 0:
                submodel = nn.Linear(i, out_size, bias=False)
            else:
                submodel = nn.Linear(in_size, i, bias=False)

            del submodel._parameters['weight']
            if matryoshka_axis == 0:
                submodel.weight = self.main_model.weight[:i, :]
            else:
                submodel.weight = self.main_model.weight[:, :i]
            self.submodels.append(submodel)

    def forward(self, x: torch.Tensor, granularity: int) -> torch.Tensor:
        if granularity not in self.grans:
            raise ValueError(f"Invalid granularity {granularity}. Must be one of {self.grans}.")
        if granularity == 0:
            return self.main_model(x)
        else:
            submodel = self.submodels[granularity - 1]
            if self.matryoshka_axis == 0:
                return submodel(x[:, :self.in_sizes[granularity - 1]])
            else:
                return submodel(x)