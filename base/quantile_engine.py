import os
import time
import numpy as np
import torch

from base.engine import BaseEngine
from utils.metrics import compute_all_metrics, masked_mae, masked_kl, masked_crps, masked_mpiw_ens, masked_mpiw, \
    masked_coverage, masked_wink, masked_nonconf
from utils.metrics import masked_mape
from utils.metrics import masked_rmse

class Quantile_Engine(BaseEngine):
    def __init__(self, **args):
        super(Quantile_Engine, self).__init__(**args)

    def train_batch(self):

        self.model.train()

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_kl = []
        train_mpiw = []
        train_crps = []

        train_cov= []
        train_wink = []


        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            # print(X.shape, label.shape)
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))

            if self.hour_day_month:
                X, hdm, label = self.split_hour_day_month(X,label)
                pred = self.model(X, hdm)
            else:
                pred = self.model(X, label)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                self._logger.info(f'check mask value {mask_value}')

            if type(pred) == tuple:
                pred, _ = pred
            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

            mid= torch.unsqueeze(pred[:,0,:,:],1)
            lower=torch.unsqueeze(pred[:,1,:,:],1)
            upper=torch.unsqueeze(pred[:,2,:,:],1)

            loss = self._loss_fn(mid, label, mask_value)
            lower_loss = torch.mean(torch.max((self.lower_bound - 1) * (label - lower), self.lower_bound * (label - lower)))
            upper_loss = torch.mean(torch.max((self.upper_bound - 1) * (label - upper), self.upper_bound * (label - upper)))
            loss = loss + lower_loss + upper_loss

            mae = masked_mae(mid, label, mask_value).item()
            mape = masked_mape(mid, label, mask_value).item()
            rmse = masked_rmse(mid, label, mask_value).item()
            crps = masked_crps(mid, label, mask_value).item()
            mpiw = masked_mpiw(lower, upper, mask_value).item()
            kl = masked_kl(mid, label, mask_value).item()

            wink = masked_wink(lower, upper, label).item()
            cov = masked_coverage(lower, upper, label).item()

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
            train_wink.append(wink)
            train_cov.append(cov)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mae), np.mean(train_mape), np.mean(train_rmse), np.mean(
            train_kl), np.mean(train_mpiw), np.mean(train_crps), np.mean(train_wink), np.mean(train_cov)

    def train(self):

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_kl, mtrain_mpiw, mtrain_crps, mtrain_wink, mtrain_cov = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_kl, mvalid_mpiw, mvalid_crps, mvalid_wink, mvalid_cov = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            # message = 'Epoch: {:03d}, Train Loss: {:.3f}, Train RMSE: {:.3f}, Train MAPE: {:.3f}, Valid Loss: {:.3f}, Valid RMSE: {:.3f}, Valid MAPE: {:.3f}, Train Time: {:.3f}s/epoch, Valid Time: {:.3f}s, LR: {:.4e}'
            message = ('Epoch: {:d}, T Loss: {:.3f},'
                       ' T MAE: {:.3f}, T RMSE: {:.3f}, T MAPE: {:.3f}, T KL: {:.3f}, T MPIW: {:.3f}, T CRPS: {:.3f}, T WINK: {:.3f}, T COV: {:.3f},'
                       ' V MAE: {:.3f}, V RMSE: {:.3f}, V MAPE: {:.3f}, V KL: {:.3f}, V MPIW: {:.3f}, V CRPS: {:.3f}, V WINK: {:.3f}, V COV: {:.3f},'
                       ' LR: {:.4e}')
            self._logger.info(message.format(epoch + 1, mtrain_loss,
                                             mtrain_mae, mtrain_rmse, mtrain_mape, mtrain_kl, mtrain_mpiw, mtrain_crps, mtrain_wink, mtrain_cov,
                                             mvalid_mae, mvalid_rmse, mvalid_mape, mvalid_kl, mvalid_mpiw, mvalid_crps, mvalid_wink, mvalid_cov,
                                             cur_lr))

            if mvalid_mae < min_loss:
                if mvalid_mae == 0:
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

        mids=[]
        lowers=[]
        uppers=[]
        labels = []
        with torch.no_grad():
            mode_=mode
            if mode=='export':
                mode_="test"
            for X, label in self._dataloader[mode_ + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                if self.hour_day_month:
                    X, hdm, label = self.split_hour_day_month(X, label)
                    pred = self.model(X, hdm)
                else:
                    pred = self.model(X, label)

                if type(pred) == tuple:
                    pred, _ = pred

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                mid = torch.unsqueeze(pred[:, 0, :, :], 1)
                lower = torch.unsqueeze(pred[:, 1, :, :], 1)
                upper = torch.unsqueeze(pred[:, 2, :, :], 1)

                mids.append(mid.squeeze(-1).cpu())
                lowers.append(lower.squeeze(-1).cpu())
                uppers.append(upper.squeeze(-1).cpu())

                labels.append(label.squeeze(-1).cpu())

        mids = torch.cat(mids, dim=0)
        lowers = torch.cat(lowers, dim=0)
        uppers = torch.cat(uppers, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = masked_mae(mid, label, mask_value).item()
            mape = masked_mape(mid, label, mask_value).item()
            rmse = masked_rmse(mid, label, mask_value).item()
            crps = masked_crps(mid, label, mask_value).item()
            mpiw = masked_mpiw(lower, upper, mask_value).item()
            kl = masked_kl(mid, label, mask_value).item()
            wink = masked_wink(lower, upper, label).item()
            cov = masked_coverage(lower, upper, label).item()
            return mae, mape, rmse, kl, mpiw, crps, wink, cov

        elif mode == 'test' or mode == 'export':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_kl = []
            test_mpiw = []
            test_crps = []
            test_wink = []
            test_cov = []

            self._logger.info(f'check mask value {mask_value}')
            for i in range(1):
                res = compute_all_metrics(mids[:, i, :], labels[:, i, :], mask_value, lowers[:, i, :], uppers[:, i, :])

                # log = ('Test Horizon: {:d}, '
                #        'MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}, WINK: {:.3f}, COV: {:.3f}')
                #
                # self._logger.info(log.format(i + 1, res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]))
                test_mae.append(res[0])
                test_rmse.append(res[1])
                test_mape.append(res[2])
                test_kl.append(res[3])
                test_mpiw.append(res[4])
                test_crps.append(res[5])
                test_wink.append(res[6])
                test_cov.append(res[7])

            self._logger.info(f"{self._save_path}")
            log = 'Average Test MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}, WINK: {:.3f}, COV: {:.3f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape),
                                         np.mean(test_kl), np.mean(test_mpiw), np.mean(test_crps), np.mean(test_wink), np.mean(test_cov)))

            if mode == 'export':
                # mae = torch.mean(test_mae[0].unsqueeze(0), axis=1)
                # mape = torch.mean(test_mape[0].unsqueeze(0), axis=1)
                # rmse = torch.mean(test_rmse[0].unsqueeze(0), axis=1)
                # kl = torch.mean(test_kl[0].unsqueeze(0), axis=1)
                # mpiw = torch.mean(test_mpiw[0].unsqueeze(0), axis=1)
                # crps = torch.from_numpy(test_crps[0])
                # crps = torch.mean(crps.unsqueeze(0), axis=1)
                #
                # wink=torch.mean(test_wink[0].unsqueeze(0), axis=1)
                # cov = torch.mean(test_cov[0].unsqueeze(0), axis=1)
                #
                # metrics = np.vstack((mae, mape, rmse, kl, mpiw, crps, wink, cov))
                # print(metrics.shape)
                # np.save(f"{self._save_path}/metrics.npy", metrics)

                mids.squeeze_(dim=1)
                lowers.squeeze_(dim=1)
                uppers.squeeze_(dim=1)
                labels.squeeze_(dim=1)

                mids.unsqueeze_(dim=0)
                lowers.unsqueeze_(dim=0)
                uppers.unsqueeze_(dim=0)
                labels.unsqueeze_(dim=0)

                result = np.vstack((mids, lowers, uppers, labels))
                self._logger.info(f'export shape (mids, lowers, uppers, labels): {result.shape}')
                np.save(f"{self._save_path}/preds_labels.npy", result)

    def cqr(self):
        mids=[]
        lowers=[]
        uppers=[]
        labels = []

        with torch.no_grad():
            for X, label in self._dataloader['val_loader'].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)

                if type(pred) == tuple:
                    pred, _ = pred

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                mid = torch.unsqueeze(pred[:, 0, :, :], 1)
                lower = torch.unsqueeze(pred[:, 1, :, :], 1)
                upper = torch.unsqueeze(pred[:, 2, :, :], 1)

                mids.append(mid.squeeze(-1).cpu())
                lowers.append(lower.squeeze(-1).cpu())
                uppers.append(upper.squeeze(-1).cpu())

                labels.append(label.squeeze(-1).cpu())

        mids = torch.cat(mids, dim=0)
        lowers = torch.cat(lowers, dim=0)
        uppers = torch.cat(uppers, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        nonconf_set=masked_nonconf(lowers, uppers,labels)
        bound=torch.quantile(nonconf_set, (1-self.alpha)*(1+1), dim=0)