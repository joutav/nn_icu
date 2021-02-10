from torch import nn


class ModeloNeural(nn.Module):

    def __init__(self, input_size, hidden_size, second_hidden, out_size):
        super(ModeloNeural, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, second_hidden),
            nn.Dropout(),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(second_hidden, out_size)
            
        )

    def forward(self, x):
        hidden = self.features(x)
        output = self.classifier(hidden)

        return output

