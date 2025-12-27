import torch
from torch import nn
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainAttention(nn.Module):
    """ Wraps domain-adapting cross attention and MLP into a module.
        The operations are similar to those in Transformer, including normalization
        layers and dropout layers, while MLP is simplified as a linear layer.

    Args:
        d_model: total dimension of the model.
        n_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights.
    """

    def __init__(self, d_model, n_heads, dropout):
        super(DomainAttention, self).__init__()
        self.grl = GradientReversal()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, src, pos=None, padding_mask=None):
        cache_query_result = []
        for class_idx in range(query.shape[1]):
            query_c = query[:, class_idx, :].unsqueeze(1)  # (b,1,d)

            if padding_mask is None:
                kpm = None
            else:
                kpm = padding_mask[:, class_idx, :].bool()  # (b,S) True=屏蔽
                all_masked = kpm.all(dim=1)  # (b,)
                if all_masked.any():
                    kpm = kpm.clone()
                    kpm[all_masked] = False  # 避免全mask

            r_query_c, _ = self.cross_attn(
                query=query_c.transpose(0, 1),
                key=self.with_pos_embed(src, pos).transpose(0, 1),
                value=src.transpose(0, 1),
                key_padding_mask=kpm,
            )
            cache_query_result.append(r_query_c)

        r_query = torch.cat(cache_query_result, dim=0)
        query = query + self.dropout1(r_query.transpose(0, 1))
        query = self.norm1(query)
        query = query + self.dropout2(self.linear(query))
        query = self.norm2(query)
        return query