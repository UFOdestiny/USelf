import os
import argparse
import numpy as np
import platform
import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(8)

from hl_model import HL
from hl_engine import HL_Engine
from utils.args import get_public_config, get_log_path, print_args
from utils.dataloader import load_dataset, get_dataset_info
from utils.metrics import masked_mae
from utils.log import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    args = parser.parse_args()

    args.model_name = "HL"
    log_dir = get_log_path(args)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    print_args(logger,args) #logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, _, node_num = get_dataset_info(args.dataset)
    
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = HL(node_num=node_num,
               input_dim=args.input_dim,
               output_dim=args.output_dim
               )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    engine = HL_Engine(device=device,
                       model=model,
                       dataloader=dataloader,
                       scaler=scaler,
                       sampler=None,
                       loss_fn=loss_fn,
                       lrate=0,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       clip_grad_value=0,
                       max_epochs=args.max_epochs,
                       patience=args.patience,
                       log_dir=log_dir,
                       logger=logger,
                       seed=args.seed,
                       normalize=args.normalize
                       )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()