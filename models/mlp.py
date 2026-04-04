import torch.nn as nn


class MLPClassifier(nn.Module):
    """Pure MLP image classifier: flatten image -> stacked linear blocks -> logits."""

    def __init__(
        self,
        input_dim,
        num_classes,
        num_hidden_layers=2,
        hidden_dim=512,
        hidden_dims=None,
        activation='relu',
        use_batchnorm=False,
        dropout=0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        if hidden_dims:
            dims = hidden_dims
        else:
            dims = [hidden_dim] * num_hidden_layers

        layers = [nn.Flatten()]
        in_dim = input_dim

        for h_dim in dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self._build_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _build_activation(name):
        name = name.lower()
        if name == 'relu':
            return nn.ReLU(inplace=True)
        if name == 'gelu':
            return nn.GELU()
        if name == 'tanh':
            return nn.Tanh()
        if name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        raise ValueError(f'Unsupported activation: {name}')

    def forward(self, x):
        return self.net(x)
