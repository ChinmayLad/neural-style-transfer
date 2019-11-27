import torch

def gram_matrix(x):
    b, c, h, w = x.shape
    # b x c x h x w
    psi = x.reshape(b, c, h*w)
    # b x c x hw
    gm = torch.matmul(psi, psi.transpose(1,2))
    # b x c x c
    return gm/(c*h*w)

def total_variational(x, weights):
    return weights * (torch.sum(torch.abs(x[:, :, :, :-1] - x [:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x [:, :, 1:, :])))