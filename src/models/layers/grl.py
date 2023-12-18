import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseGradient(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output.neg() * alpha
        return grad_x, None


def scale(x, high, low):
    delta = high - low
    return (x * 2.0 * delta) + low - delta


class ReverseGradientScaled(Function):
    @staticmethod
    def forward(ctx, x, alpha, iter_num, max_iter, high, low):
        ctx.alpha = alpha
        ctx.high = high
        ctx.low = low
        ctx.iter_num = iter_num
        ctx.max_iter = max_iter
        return x

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        high = ctx.high
        low = ctx.low
        iter_num = ctx.iter_num
        max_iter = ctx.max_iter

        grad_x = None
        if ctx.needs_input_grad[0]:
            coeff = torch.sigmoid(torch.tensor(alpha * iter_num/max_iter, requires_grad=False))
            grad_x = grad_output.neg() * scale(coeff, high, low)

        return grad_x, None, None, None, None, None


reverse_scaled = ReverseGradientScaled.apply
reverse = ReverseGradient.apply


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1., high=1., low=0., max_iter=10000, scale=False):
        super().__init__()
        self.alpha = alpha
        self.iter_num = 0
        self.max_iter = max_iter * 4
        self.high = high
        self.low = low
        self.scale = scale

    def forward(self, x):
        if self.scale:
            self.iter_num += 1
            return reverse_scaled(x, self.alpha, self.iter_num, self.max_iter, self.high, self.low)
        else:
            return reverse(x, self.alpha)
