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
    "embed_dim": 64,
    "hidden_dim": 64,
    "output_len": 12,
    "num_layer": 3,
    "if_node": False,
    "if_T_i_D": False,
    "if_D_i_W": False,
    "node_dim": 64,
    "temp_dim_tid": 64,
    "temp_dim_diw": 64,
    "time_of_day_size": 24 * 6,
    "day_of_week_size": 7
}