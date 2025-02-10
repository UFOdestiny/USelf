import torch.nn as nn
from base.model import BaseModel

class HL(BaseModel):
    def __init__(self, **args):
        super(HL, self).__init__(**args)   
        self.fake = nn.Linear(12, 1)


    def forward(self, input, label=None):  # (b, t, n, f)
        x = input.permute(0, 2, 3, 1)
        x = self.fake(x)
        x = x.permute(0, 3, 1, 2)
        # x = input[:,[-1],:,:].expand(-1, self.horizon, -1, -1)


        return x