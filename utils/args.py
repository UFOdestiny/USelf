import argparse
import platform

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str, default='')

    parser.add_argument('--dataset', type=str, default='NYISO_HDM')  # Shenzhen NYC
    parser.add_argument('--years', type=str, default='2024')
    parser.add_argument('--model_name', type=str, default='')

    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=1)

    parser.add_argument('--feature', type=int, default=1)
    parser.add_argument('--input_dim', type=int, default=1)  # feature
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=400)
    parser.add_argument('--normalize', type=bool, default=True)

    parser.add_argument('--quantile', type=bool, default=True)
    parser.add_argument('--quantile_alpha', type=float, default=0.1)
    parser.add_argument('--hour_day_month', type=bool, default=True)

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--print_args', type=bool, default=False)

    
    return parser

def get_log_path(args):
    if platform.system().lower() == 'linux':
        log_dir = '/home/dy23a.fsu/st/result/{}/{}/'.format(args.model_name, args.dataset)
    else:
        log_dir = r'D:/OneDrive - Florida State University/mycode/st/result/{}/{}/'.format(
            args.model_name, args.dataset)

    return log_dir

def get_data_path():
    if platform.system().lower() == 'linux':
        path = '/blue/gtyson.fsu/dy23a.fsu/datasets/'
        # path = '/home/dy23a.fsu/neu24/LargeST-old/data/'
    else:
        path = 'D:/OneDrive - Florida State University/mycode/st/data/'

    return path

def print_args(logger, args):
    if args.print_args:
        for k, v in vars(args).items():
            logger.info('{}: {}'.format(k, v))

def check_quantile(args, normal_model, quantile_model):
    if args.quantile:
        assert args.horizon == 1
        assert args.output_dim == 1
        args.horizon = 3
        args.output_dim = 3
        return args, quantile_model
    return args, normal_model