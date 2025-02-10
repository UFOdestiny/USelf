import os
import time

import numpy as np
import torch

from utils.metrics import compute_all_metrics, masked_mae, masked_kl, masked_crps, masked_mpiw_ens
from utils.metrics import masked_mape
from utils.metrics import masked_rmse


class BaseEngine:
    def __init__(self, device, model, dataloader, scaler, sampler, loss_fn, lrate, optimizer,
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed, alpha=0.1, normalize=True,
                 hour_day_month=False):
        super().__init__()
        self._normalize = normalize
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed

        self._logger.info('The number of parameters: {}'.format(self.model.param_num()))

        self._time_model = 'final_model_{}.pt'.format(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))

        self.alpha = alpha
        self.lower_bound = self.alpha / 2
        self.upper_bound = 1 - self.alpha / 2
        self.hour_day_month = hour_day_month

    def split_hour_day_month(self, X, Y):
        data = X[...,0].unsqueeze(-1)
        hdm = X[..., 1:]
        y=Y[...,0].unsqueeze(-1)
        return data, hdm, y

    def predict(self, x):
        return self.model(x)

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)

    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            res = []
            for t in tensors:
                if type(t) == tuple:
                    res.append([inv(j) for j in t])
                else:
                    res.append(inv(t))
            return res
        else:
            return inv(tensors)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # filename = 'final_model_s{}.pt'.format(self._seed)
        filename = self._time_model
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def load_model(self, save_path):
        # filename = 'final_model_s{}.pt'.format(self._seed)
        filename = self._time_model
        f=os.path.join(save_path, filename)
        if not os.path.exists(f):
            models=[i for i in os.listdir(save_path) if i[-3:]=='.pt']
            if len(models)==0:
                self._logger.info(f"Model {f} Not Exist. No More Models.")
                exit()

            models.sort(key=lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
            m=os.path.join(save_path, models[-1])
            self._logger.info(f"Model {f} Not Exist. Try the Newest Model {m}.")
            f=m

        self.model.load_state_dict(torch.load(f, weights_only=False))

    def load_exact_model(self, path):
        self.model.load_state_dict(torch.load(path,weights_only=False))

    def train_batch(self):

        self.model.train()

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_kl = []
        train_mpiw = []
        train_crps = []

        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            # print(X.shape, label.shape)
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                self._logger.info(f'check mask value {mask_value}')

            if type(pred)==tuple:
                pred, _ = pred
            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

            loss = self._loss_fn(pred, label, mask_value)

            mae = masked_mae(pred, label, mask_value).item()
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()
            crps = masked_crps(pred, label, mask_value).item()
            mpiw = masked_mpiw_ens(pred, label, mask_value).item()
            kl = masked_kl(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())

            train_mae.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)

            train_kl.append(kl)
            train_mpiw.append(mpiw)
            train_crps.append(crps)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mae), np.mean(train_mape), np.mean(train_rmse), np.mean(
            train_kl), np.mean(train_mpiw), np.mean(train_crps)

    def train(self):

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_kl, mtrain_mpiw, mtrain_crps = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_kl, mvalid_mpiw, mvalid_crps = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            # message = 'Epoch: {:03d}, Train Loss: {:.3f}, Train RMSE: {:.3f}, Train MAPE: {:.3f}, Valid Loss: {:.3f}, Valid RMSE: {:.3f}, Valid MAPE: {:.3f}, Train Time: {:.3f}s/epoch, Valid Time: {:.3f}s, LR: {:.4e}'
            message = ('Epoch: {:d}, T Loss: {:.3f},'
                       ' T MAE: {:.3f}, T RMSE: {:.3f}, T MAPE: {:.3f}, T KL: {:.3f}, T MPIW: {:.3f}, T CRPS: {:.3f},'
                       ' V MAE: {:.3f}, V RMSE: {:.3f}, V MAPE: {:.3f}, V KL: {:.3f}, V MPIW: {:.3f}, V CRPS: {:.3f},'
                       ' LR: {:.4e}')
            self._logger.info(message.format(epoch + 1, mtrain_loss,
                                             mtrain_mae, mtrain_rmse, mtrain_mape, mtrain_kl, mtrain_mpiw, mtrain_crps,
                                             mvalid_mae, mvalid_rmse, mvalid_mape, mvalid_kl, mvalid_mpiw, mvalid_crps,
                                             cur_lr))
            

            if mvalid_mae < min_loss:
                if mvalid_mae==0:
                    self._logger.info("Something went WRONG!")
                    break

                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.3f} to {:.3f}'.format(min_loss, mvalid_mae))
                min_loss = mvalid_mae
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        self.evaluate('test')

    def evaluate(self, mode, model_path=None):
        if mode == 'test' or mode == 'export':
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)

                if type(pred) == tuple:
                    pred, _ = pred

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = masked_mae(pred, label, mask_value).item()
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()
            crps = masked_crps(pred, label, mask_value).item()
            mpiw = masked_mpiw_ens(pred, label, mask_value).item()
            kl = masked_kl(pred, label, mask_value).item()
            return mae, mape, rmse, kl, mpiw, crps

        elif mode == 'test' or mode == 'export':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_kl = []
            test_mpiw = []
            test_crps = []

            self._logger.info(f'check mask value {mask_value}')
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)

                log = ('Test Horizon: {:d}, '
                       'MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}')

                self._logger.info(log.format(i + 1, res[0], res[1], res[2], res[3], res[4], res[5]))
                test_mae.append(res[0])
                test_rmse.append(res[1])
                test_mape.append(res[2])
                test_kl.append(res[3])
                test_mpiw.append(res[4])
                test_crps.append(res[5])

            log = 'Average Test MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape),
                                         np.mean(test_kl), np.mean(test_mpiw), np.mean(test_crps)))

            if mode == 'export':
                mae = torch.mean(test_mae[0].unsqueeze(0), axis=1)
                mape = torch.mean(test_mape[0].unsqueeze(0), axis=1)
                rmse = torch.mean(test_rmse[0].unsqueeze(0), axis=1)
                kl = torch.mean(test_kl[0].unsqueeze(0), axis=1)
                mpiw = torch.mean(test_mpiw[0].unsqueeze(0), axis=1)
                crps = torch.from_numpy(test_crps[0])
                crps = torch.mean(crps.unsqueeze(0), axis=1)

                metrics = np.vstack((mae, mape, rmse, kl, mpiw, crps))
                print(metrics.shape)
                np.save(f"{self._save_path}/metrics.npy", metrics)

                preds.squeeze_(dim=1)
                labels.squeeze_(dim=1)
                preds.unsqueeze_(dim=0)
                labels.unsqueeze_(dim=0)

                result = np.vstack((preds, labels))
                np.save(f"{self._save_path}/preds_labels.npy", result)