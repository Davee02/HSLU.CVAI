import torch.nn as nn

class GestureClassifier(nn.Module):  
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.2):
        super(GestureClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)