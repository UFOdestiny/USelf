import argparse
import os
import torch
import numpy as np


class StandardScaler_:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StandardScaler:
    def __init__(self, mean=None, std=None, offset=None):
        """
        :param axis: 指定标准化的轴。
                     - axis=0：对每个特征进行标准化。
                     - axis=1：对每个样本的特征集合标准化。
                     - axis=2：对每个时间点的特征集合标准化。
        """
        if mean is not None:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)
            self.offset = torch.tensor(offset)
        else:
            self.mean = mean
            self.std = std
            self.offset = offset

    def fit(self, data):
        sequence,region,dim=data.shape
        temp_data = data.reshape(sequence*region,dim)
        self.mean = np.mean(temp_data, axis=0)
        self.std = np.std(temp_data, axis=0)

        # 确保标准差不为 0，避免除以 0 的情况
        self.std[self.std == 0] = 1.0

        # 计算标准化后的数据的最小值
        normalized_data = (temp_data - self.mean) / self.std
        self.offset = -np.min(normalized_data, axis=0)

        # self.offset+=1.5 # important

    def transform(self, data):
        if self.mean is None or self.std is None or self.offset is None:
            raise ValueError("StandardScaler is not fitted yet. Call 'fit' with training data first.")
        normalized_data = (data - self.mean) / self.std
        return normalized_data + self.offset

    def inverse_transform(self, data):
        if self.mean is None or self.std is None or self.offset is None:
            raise ValueError("StandardScaler is not fitted yet. Call 'fit' with training data first.")
        unshifted_data = data - self.offset.to(device="cuda")
        return (unshifted_data * self.std.to(device="cuda")) + self.mean.to(device="cuda")




class LogScaler:
    def __init__(self):
        self.base = 1

    def transform(self, data):
        return np.log(data + 1)

    def inverse_transform(self, data):
        return np.exp(data) - 1


class RatioScaler:
    def __init__(self, ratio=1000):
        self.ratio = ratio

    def transform(self, data):
        return data / self.ratio

    def inverse_transform(self, data):
        return data * self.ratio.to(device="cuda")


def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape
    print(df.shape)
    data = np.expand_dims(df.values, axis=-1)

    feature_list = [data]
    if add_time_of_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
    if add_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    data_path = "D:/OneDrive - Florida State University/datasets/electricity/NYISO/nyiso2024_hdm.npy"
    # data_path = "D:/OneDrive - Florida State University/datasets/electricity/CAISO/caiso2024_hdm.npy"
    # data_path = "D:/OneDrive - Florida State University/datasets/chicago/mobility.npy"
    # data_path = "D:/OneDrive - Florida State University/datasets/chicago/mobility_crime.npy"
    # data_path = "D:/OneDrive - Florida State University/datasets/nyc/mobility_crash_crime.npy"
    # data_path = "D:/OneDrive - Florida State University/datasets/nyc/crash_crime.npy"
    # data_path = "D:/OneDrive - Florida State University/datasets/shenzhen/shenzhen_1h/values_in.npy"
    # data_path="D:/OneDrive - Florida State University/datasets/values_in.npy"

    # data, idx = generate_data_and_idx(df, x_offsets, y_offsets, args.tod, args.dow)
    # nyc = "D:/OneDrive - Florida State University/datasets/values_in.npy"
    # sz = "D:/OneDrive - Florida State University/code/UAHGNN/data/shenzhen/shenzhen_1h/values_in.npy"
    # sz2 = "D:/OneDrive - Florida State University/heterogeneous/data/values_in2.npy"
    # sz_1month="D:/OneDrive - Florida State University/code/UAHGNN/data/shenzhen/shenzhen_1h/values_in.npy"

    data = np.load(data_path)
    data = data.transpose(2, 0, 1)

    # data = data[:720,:,:]
    # important
    # data += 1
    # important

    min_t = abs(min(x_offsets))
    max_t = abs(data.shape[0] - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)

    print('final data shape:', data.shape, 'idx shape:', idx.shape)
    num_samples = len(idx)
    num_train = round(num_samples * 0.8)
    num_val = round(num_samples * 0.1)

    # split idx

    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]
    idx_all = idx[:]

    # normalize
    # x_train = data[:idx_val[0] - args.seq_length_x, :, :]

    # hour day month
    data_hdm=data[:,:,1:]
    data=np.expand_dims(data[:,:,0], axis=-1)

    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    mean = scaler.mean
    std = scaler.std
    offset = scaler.offset

    # hour day month
    data=np.concatenate([data, data_hdm], axis=-1)

    print(mean)
    print(std)
    print(offset)

    print(data.max())
    print(data.min())
    print(data.mean())


    out_dir = args.dataset + '/' + args.years
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.savez_compressed(os.path.join(out_dir, 'his.npz'), data=data, mean=mean, std=std, offset=offset)
    np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test'), idx_test)
    np.save(os.path.join(out_dir, 'idx_all'), idx_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='nyiso_hdm', help='dataset name')
    parser.add_argument('--years', type=str, default='2024')
    parser.add_argument('--seq_length_x', type=int, default=12, help='sequence Length')
    parser.add_argument('--seq_length_y', type=int, default=1, help='sequence Length')
    parser.add_argument('--tod', type=int, default=1, help='time of day')
    parser.add_argument('--dow', type=int, default=1, help='day of week')

    args = parser.parse_args()
    generate_train_val_test(args)
