from easydict import EasyDict
import torch

args = EasyDict()

args.train = EasyDict()
args.train = {
    "root_path":"raw_data",
    "batch_size":256,
    "lr": 1e-3,
    "seed": 2024,
    "loss": torch.nn.L1Loss()
}


args.model = EasyDict()
args.model = {
    "num_nodes": 32,
    "input_len": 12,
    "input_dim": 1,
    "embed_dim": 32,
    "hidden_dim": 32,
    "output_len": 12,
    "num_layer": 3,
    "if_node": True,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "node_dim": 32,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": 24 * 6,
    "day_of_week_size": 7
}