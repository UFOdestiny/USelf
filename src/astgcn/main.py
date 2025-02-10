import os
import argparse
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

torch.set_num_threads(8)

from astgcn_model import ASTGCN
from astgcn_engine import ASTGCN_Engine
from utils.args import get_public_config, get_log_path, print_args, check_quantile
from utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from utils.graph_algo import normalize_adj_mx, calculate_cheb_poly
from utils.metrics import masked_mae
from utils.log import get_logger

from base.engine import BaseEngine
from base.quantile_engine import Quantile_Engine
from src.astgcn.astgcn_engine import ASTGCN_Engine_Quantile


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--nb_block', type=int, default=2)
    parser.add_argument('--nb_chev_filter', type=int, default=64)
    parser.add_argument('--nb_time_filter', type=int, default=64)
    parser.add_argument('--time_stride', type=int, default=1)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()
    args.model_name = "ASTGCN"

    log_dir = get_log_path(args)
    logger = get_logger(log_dir, __name__, )
    print_args(logger, args)  # logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    # logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    adj = np.zeros((node_num, node_num), dtype=np.float32)
    for n in range(node_num):
        idx = np.nonzero(adj_mx[n])[0]
        adj[n, idx] = 1

    L_tilde = normalize_adj_mx(adj, 'scalap')[0]
    cheb_poly = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in
                 calculate_cheb_poly(L_tilde, args.order)]

    dataloader, scaler = load_dataset(data_path, args, logger)
    args, engine_template = check_quantile(args, ASTGCN_Engine, ASTGCN_Engine_Quantile)
    # if args.quantile:
    #     args.output_dim=3


    model = ASTGCN(node_num=node_num,
                   input_dim=args.input_dim,
                   output_dim=args.output_dim,
                   horizon=args.horizon,
                   device=args.device,
                   cheb_poly=cheb_poly,
                   order=args.order,
                   nb_block=args.nb_block,
                   nb_chev_filter=args.nb_chev_filter,
                   nb_time_filter=args.nb_time_filter,
                   time_stride=args.time_stride
                   )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    engine = engine_template(device=device,
                           model=model,
                           dataloader=dataloader,
                           scaler=scaler,
                           sampler=None,
                           loss_fn=loss_fn,
                           lrate=args.lrate,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           clip_grad_value=args.clip_grad_value,
                           max_epochs=args.max_epochs,
                           patience=args.patience,
                           log_dir=log_dir,
                           logger=logger,
                           seed=args.seed,
                           alpha=args.quantile_alpha,
                           )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()