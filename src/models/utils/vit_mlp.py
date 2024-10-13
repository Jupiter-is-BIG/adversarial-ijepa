import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters:
    -----------
    in_features     : int   | Number of input features
    hidden_features : int   | Number of nodes in the hidden layer
    out_features    : int   | Nubmer of output features
    drop            : float | Dropout probabilty
    """

    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor | Shape `(batch_size, n_patches + 1, in_features)`

        Returns:
        --------
        torch.Tensor     | Shape `(batch_size, n_patches + 1, out_featues)`
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x