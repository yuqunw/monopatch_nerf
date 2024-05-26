import torch
from torch.autograd import Function
from .backend import _backend

class qff(Function):
    @staticmethod
    def forward(ctx, points, freqs, features):
        B = points.shape[0]
        F = freqs.shape[0]
        F8, C, R, _, _ = features.shape

        # outputs = Bx(F*8*C)
        outputs = torch.zeros((B, C*2*F), device=features.device, dtype=features.dtype)
        dy_dx_cache = torch.zeros((1), device=features.device, dtype=features.dtype)

        if not points.is_contiguous():
            points = points.contiguous()
        if not freqs.is_contiguous():
            freqs = freqs.contiguous()
        if not features.is_contiguous():
            features = features.contiguous()

        ctx.save_for_backward(points, freqs, features)
        ctx.dims = [B, F, C, R]
        _backend.qff_forward(points, freqs, features, outputs, dy_dx_cache, B, F, C, R)
        
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        points, freqs, features = ctx.saved_tensors
        B, F, C, R = ctx.dims

        if not grad_outputs.is_contiguous():
            grad_outputs = grad_outputs.contiguous()
        return qff_backward.apply(grad_outputs, points, freqs, features)



class qff_backward(Function):
    @staticmethod
    def forward(ctx, grad_outputs, points, freqs, features):
        B = points.shape[0]
        F = freqs.shape[0]
        F8, C, R, _, _ = features.shape


        # NxCxIHxIW
        grad_points = torch.zeros_like(points)
        grad_features= torch.zeros_like(features)

        _backend.qff_backward(grad_outputs, points, freqs, features, grad_points, grad_features, B, F, C, R)
        ctx.save_for_backward(grad_outputs, points, freqs, features)
        ctx.dims = [B, F, C, R]

        return grad_points, None, grad_features

    @staticmethod
    def backward(ctx, grad_grad_points, grad_grad_freq, grad_grad_features):
        # only using grad_grad_points for now
        # grad_grad_points = d(dL_dp)

        grad_outputs, points, freqs, features = ctx.saved_tensors
        B, F, C, R = ctx.dims

        if not grad_grad_points.is_contiguous():
            grad_grad_points = grad_grad_points.contiguous()
        if not grad_grad_features.is_contiguous():
            grad_grad_features = grad_grad_features.contiguous()

        # NxCxIHxIW
        grad2_features = torch.zeros_like(grad_grad_features)
        grad_grad_outputs = torch.zeros_like(grad_outputs)
        _backend.qff_backward_backward(grad_outputs, grad_grad_points, points, freqs, features, grad_grad_outputs, grad2_features, B, F, C, R)

        # grad2_grad_points and grad2_outputs not implemented
        grad2_points = None
        grad2_freqs = None
        return grad_grad_outputs, grad2_points, grad2_freqs, grad2_features

