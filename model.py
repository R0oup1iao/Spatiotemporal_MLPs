from einops import rearrange, repeat 
import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=input_dim,  out_features=hidden_dim, bias=True)
        self.fc2 = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)
        self.ii = nn.Linear(input_dim, hidden_dim)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + self.ii(input_data)                          # residual
        return hidden


class STMLP(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        # self.rev_in = RevIN(self.input_dim)
        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = MultiLayerPerceptron(
            input_dim=self.input_dim * self.input_len, hidden_dim=self.embed_dim)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Linear(
            in_features=self.hidden_dim, out_features=self.output_len, bias=True)

    def forward(self, seq_x: torch.Tensor, seq_x_mark: torch.Tensor, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            seq_x (torch.Tensor): history data with shape [B, L, N]
            seq_x_mark (torch.Tensor): weekday, and hour with shape [B, L, 2]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N]
        """
        # prepare data
        B, L, N = seq_x.shape
        input_data = seq_x.unsqueeze(-1)    # [B, L, N, 1]

        if self.if_time_in_day:
            t_i_d_data = seq_x_mark[:, -1, 1] # hour
            time_in_day_emb = self.time_in_day_emb[t_i_d_data]
            time_in_day_emb = repeat(time_in_day_emb, 'b c -> b n c', n=N)
        else:
            time_in_day_emb = None
            
        if self.if_day_in_week:
            d_i_w_data = seq_x_mark[:, -1, 0]
            day_in_week_emb = self.day_in_week_emb[d_i_w_data]
            day_in_week_emb = repeat(day_in_week_emb, 'b c -> b n c', n=N)
        else:
            day_in_week_emb = None
        
        if self.if_spatial:
            node_emb = repeat(self.node_emb, 'n c -> b n c', b=B)
        else:
            node_emb = None
            
        # time series embedding
        input_data = rearrange(input_data, 'b l n c -> b n (l c)')
        time_series_emb = self.time_series_emb_layer(input_data)
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb)
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb)
        # concate all embeddings
        if node_emb is not None and tem_emb:
            hidden = torch.cat([time_series_emb] + [node_emb] + tem_emb, dim=-1)
        elif tem_emb:
            hidden = torch.cat([time_series_emb] + tem_emb, dim=-1)
        elif node_emb is not None:
            hidden = torch.cat([time_series_emb] + [node_emb], dim=-1)
        else:
            hidden = time_series_emb
        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)
        prediction = rearrange(prediction, 'b n l -> b l n')
        return prediction

