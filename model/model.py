import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, input_dim=2, dim_model=64, num_heads=1, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(in_features=input_dim, out_features=dim_model)

        self.pos_emb = self.positional_encoder()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=dim_model, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, hand):
        # x shape: (batch_size, 21, 2)

        hand = self.input_proj(hand)
        hand = hand + self.pos_emb
        hand = self.encoder(hand)
        hand = hand.mean(dim=1)
        return self.net(hand)

    def positional_encoder(self):
        encode_table = torch.rand((21,2))
        encode_table[:, 1] = encode_table[:, 0]
        return encode_table

class MLP(nn.Module):
    def __init__(self, input_dim=42, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=128, out_features=num_classes),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            #
            # nn.Linear(in_features=64, out_features=num_classes),
            nn.ReLU(),
        )
    def forward(self, hand):
        # x shape: (batch, 21, 2)
        hand = hand.view(hand.size(0), -1)  # flatten -> (batch, 42)
        return self.net(hand)