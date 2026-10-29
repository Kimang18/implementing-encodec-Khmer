from torch import nn


class SLSTM(nn.Module):
    def __init__(self, dimension, num_layers=2, bidirectional=True, skip=True):
        super().__init__()
        self.lstm = nn.LSTM(
            dimension,
            dimension,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        if bidirectional:
            self.proj = nn.Linear(dimension * 2, dimension)
        else:
            self.proj = nn.Identity()
        self.skip = skip

    def forward(self, x):
        # permute b c l -> b l c
        x = x.permute(0, 2, 1)
        y, _ = self.lstm(x)
        y = self.proj(y)

        if self.skip:
            y = y + x
        # put back
        y = y.permute(0, 2, 1)
        return y
