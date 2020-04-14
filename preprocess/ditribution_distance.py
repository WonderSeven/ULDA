
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

def MMD(samples_A, samples_B, sigma=1, biased=True):
    alpha = 1 / (2 * sigma**2)
    B = samples_A.size(0)

    AA, BB = torch.mm(samples_A, samples_A.t()), torch.mm(samples_B, samples_B.t())
    AB = torch.mm(samples_A, samples_B.t())
    rA = (AA.diag().unsqueeze(0).expand_as(AA))
    rB = (BB.diag().unsqueeze(0).expand_as(BB))

    K = torch.exp(- alpha * (rA.t() + rA - 2*AA))
    L = torch.exp(- alpha * (rB.t() + rB - 2*BB))
    P = torch.exp(- alpha * (rA.t() + rB - 2*AB))

    if biased:
        return K.mean() + L.mean() - 2 * P.mean()

    beta = (1./(B*(B-1)))
    gamma = (2./(B*B))

    return beta*(torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)


def KL_divergence(samples_A, sample_B):
    return F.kl_div(samples_A, sample_B)
    pass

def Wasserstein_distance(sample_A, sample_B):
    # TODO: BUG
    if isinstance(sample_A, torch.Tensor):
        sample_A = sample_A.cpu().numpy()
    if isinstance(sample_B, torch.Tensor):
        sample_B = sample_B.cpu().numpy()
    return wasserstein_distance(sample_A, sample_B)


def FID(samples_A, smaples_B):
    
    pass


if __name__ == '__main__':
    x = torch.rand((8, 1600)).float()
    y = torch.rand((8, 1600)).float()

    mmd_dis = MMD(x, y)
    kl_dis = KL_divergence(x, y)
    wasser_dis = Wasserstein_distance(x, y)
    print(mmd_dis)
    print(kl_dis)
    print(wasser_dis)