from torch.nn import Module


class EMA(Module):
    def __init__(self, module: Module, decay: float) -> None:
        super(EMA, self).__init__()
        self._module = module
        self._decay = decay
        self._shadow = {}
        self._backup = {}

    def register(self) -> None:
        for name, param in self._module.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self._module.named_parameters():
            if param.requires_grad:
                new_average = (1. - self._decay) * param.data + self._decay * self._shadow[name]
                self._shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        for name, param in self._module.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data
                param.data = self._shadow[name]

    def restore(self) -> None:
        for name, param in self._module.named_parameters():
            if param.requires_grad:
                param.data = self._backup[name]
        self._backup = {}

    def save(self) -> dict:
        return {'shadow': self._shadow, 'backup': self._backup}

    def load(self, data: dict) -> None:
        self._shadow = data['shadow']
        self._backup = data['backup']
