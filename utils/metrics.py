import numpy as np
import properscoring as ps
import torch
from torch.distributions.laplace import Laplace
from torch.distributions.log_normal import LogNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

zero=torch.tensor(0.)

def masked_pinball(preds, labels, null_val, quantile):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.zeros_like(labels, dtype=torch.float)
    error = preds - labels
    smaller_index = error < 0
    bigger_index = 0 < error
    loss[smaller_index] = quantile * (abs(error)[smaller_index])
    loss[bigger_index] = (1 - quantile) * (abs(error)[bigger_index])

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)

def masked_mse(preds, labels, null_val):
    assert preds.shape == labels.shape
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val):
    assert preds.shape == labels.shape
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask

    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)

def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)*100 #percent

def masked_kl(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = labels * torch.log((labels+ 1e-5 )/ (preds + 1e-5))

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)

def masked_mpiw(lower, upper, null_val):
    return torch.mean(upper-lower)

def masked_wink(lower, upper, labels, alpha=0.1):
    score = upper-lower
    score += (2 / alpha) * torch.maximum(lower - labels, zero)
    score += (2 / alpha) * torch.maximum(labels - upper, zero)
    return torch.mean(score)

def masked_coverage(lower, upper, labels):
    in_the_range = torch.sum((labels >= lower) & (labels <= upper))
    coverage = in_the_range / labels.numel() * 100
    return coverage

def masked_nonconf(lower, upper, labels):
    return torch.maximum(lower-labels, labels-upper)



def masked_mpiw_ens(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    m = torch.mean(preds, dim=list(range(1, preds.dim())))
    # print(torch.min(preds),torch.quantile(m, 0.05),torch.mean(preds),torch.quantile(m, 0.95),torch.max(preds))

    upper_bound = torch.quantile(m, 0.95)
    lower_bound = torch.quantile(m, 0.05)
    loss=upper_bound-lower_bound

    return torch.mean(loss)#-torch.mean(torch.quantile(m, 0.8)-torch.quantile(m, 0.2))


def compute_all_metrics(preds, labels, null_val, lower=None, upper=None):
    mae = masked_mae(preds, labels, null_val)
    mape = masked_mape(preds, labels, null_val)
    rmse = masked_rmse(preds, labels, null_val)

    crps = masked_crps(preds, labels, null_val)
    mpiw = masked_mpiw_ens(preds, labels, null_val)
    kl = masked_kl(preds, labels, null_val)

    res=[mae,rmse,mape,kl,mpiw,crps]

    if lower is not None:
        res[4]=masked_mpiw(lower, upper, null_val)
        wink = masked_wink(lower, upper, labels)
        cov = masked_coverage(lower, upper, labels)
        res=res+[wink,cov]

    return res


def nb_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    n, p, pi = preds
    pi = torch.clip(pi, 1e-3, 1 - 1e-3)
    p = torch.clip(p, 1e-3, 1 - 1e-3)

    idx_yeq0 = labels <= 0
    idx_yg0 = labels > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = labels[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = labels[idx_yg0]

    lambda_ = 1e-4

    L_yeq0 = torch.log(pi_yeq0 + lambda_) + torch.log(lambda_ + (1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = torch.log(1 - pi_yg0 + lambda_) + torch.lgamma(n_yg0 + yg0) - torch.lgamma(yg0 + 1) - torch.lgamma(
        n_yg0 + lambda_) + n_yg0 * torch.log(p_yg0 + lambda_) + yg0 * torch.log(1 - p_yg0 + lambda_)

    loss = -torch.sum(L_yeq0) - torch.sum(L_yg0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss)


def nb_nll_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    n, p, pi = preds

    idx_yeq0 = labels <= 0
    idx_yg0 = labels > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = labels[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = labels[idx_yg0]

    index1 = p_yg0 == 1
    p_yg0[index1] = torch.tensor(0.9999)
    index2 = pi_yg0 == 1
    pi_yg0[index2] = torch.tensor(0.9999)
    index3 = pi_yeq0 == 1
    pi_yeq0[index3] = torch.tensor(0.9999)
    index4 = pi_yeq0 == 0
    pi_yeq0[index4] = torch.tensor(0.001)

    L_yeq0 = torch.log(pi_yeq0) + torch.log((1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = torch.log(1 - pi_yg0) + torch.lgamma(n_yg0 + yg0) - torch.lgamma(yg0 + 1) - torch.lgamma(
        n_yg0) + n_yg0 * torch.log(p_yg0) + yg0 * torch.log(1 - p_yg0)

    loss = -torch.sum(L_yeq0) - torch.sum(L_yg0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss)


def gaussian_nll_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds
    var = torch.pow(scale, 2)
    loss = (labels - loc) ** 2 / var + torch.log(2 * torch.pi * var)

    # pi = torch.acos(torch.zeros(1)).item() * 2
    # loss = 0.5 * (torch.log(2 * torch.pi * var) + (torch.pow(labels - loc, 2) / var))

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = torch.sum(loss)
    return loss


def laplace_nll_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds
    loss = torch.log(2 * scale) + torch.abs(labels - loc) / scale

    # d = torch.distributions.poisson.Poisson
    # loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = torch.sum(loss)
    return loss


def mnormal_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds

    dis = MultivariateNormal(loc=loc, covariance_matrix=scale)
    loss = dis.log_prob(labels)

    if loss.shape == mask.shape:
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # loss = -torch.sum(loss)
    loss = -torch.mean(loss)
    return loss


def mnb_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    mu, r = preds

    term1 = torch.lgamma(labels + r) - torch.lgamma(r) - torch.lgamma(labels + 1)
    term2 = r * torch.log(r) + labels * torch.log(mu)
    term3 = -(labels + r) * torch.log(r + mu)
    loss = term1 + term2 + term3


    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def normal_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds
    d = Normal(loc, scale)
    loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def lognormal_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds

    dis = LogNormal(loc, scale)
    loss = dis.log_prob(labels + 0.000001)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def tnormal_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds

    d = Normal(loc, scale)
    prob0 = d.cdf(torch.Tensor([0]).to(labels.device))
    loss = d.log_prob(labels) - torch.log(1 - prob0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def laplace_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds

    d = Laplace(loc, scale)
    loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def masked_crps(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # m, v = preds
    # if v.shape != m.shape:
    #     v = torch.diagonal(v, dim1=-2, dim2=-1)

    # loss = ps.crps_gaussian(labels, mu=m, sig=v)
    loss = ps.crps_ensemble(labels.cpu().numpy(), preds.cpu().detach().numpy())
    
    return loss.mean()




if __name__ == "__main__":
    tensor = torch.randn(64, 1, 11, 1)  # 64x1x11x1 随机张量
    lower = tensor - 0.5
    upper = tensor + 0.5
    labels = tensor + torch.randn(64, 1, 11, 1) * 0.3  # 添加噪声的真实值

    # 计算 MPIW 和 Winkler Score
    mpi_w = masked_mpiw(lower, upper, None)
    winkler_score = masked_wink(lower, upper, labels)

    print("MPIW:", mpi_w.item())
    print("Winkler Score:", winkler_score.item())
