import torch
import torch.nn as nn

# Parameters
EMBED_MLP = 32
EMBED_GMF = 8
MLP_LAYERS = [64, 32, 16, 8]

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_mlp=EMBED_MLP, embed_gmf=EMBED_GMF, mlp_layers=MLP_LAYERS):
        super(NCF, self).__init__()
        self.user_embed_mlp = nn.Embedding(num_users, embed_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, embed_mlp)
        self.user_embed_gmf = nn.Embedding(num_users, embed_gmf)
        self.item_embed_gmf = nn.Embedding(num_items, embed_gmf)

        # MLP tower
        mlp_modules = []
        input_size = 2 * embed_mlp  # Concat size
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_modules[:-1])  # Remove last ReLU for output

        # Prediction layer: concat GMF and MLP out
        self.predict = nn.Linear(embed_gmf + mlp_layers[-1], 1)

    def forward(self, user, item):
        u_mlp = self.user_embed_mlp(user)
        i_mlp = self.item_embed_mlp(item)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)

        u_gmf = self.user_embed_gmf(user)
        i_gmf = self.item_embed_gmf(item)
        gmf = u_gmf * i_gmf

        combined = torch.cat([gmf, mlp_out], dim=-1)
        out = torch.sigmoid(self.predict(combined))
        return out.squeeze()

    def get_user_item_embeds(self, users, items):
        """Get concatenated user-item embeddings for coverage/similarity"""
        u_mlp = self.user_embed_mlp(users)
        i_mlp = self.item_embed_mlp(items)
        return torch.cat([u_mlp, i_mlp], dim=-1).detach().numpy()