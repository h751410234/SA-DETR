import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from timm.models.layers import trunc_normal_

def calculate_mu_sig(x, eps=1e-6):
    mu = x.mean(dim=[2, 3])
    var = x.var(dim=[2, 3])
    sig = (var + eps).sqrt()
    mu = mu.detach()
    sig = sig.detach()
    return mu, sig

def momentum_update(old_value, new_value, momentum):
    update = momentum * old_value + (1 - momentum) * new_value
    return update



class StyleRepresentation(nn.Module):
    def __init__(self, num_prototype=2,
                 channel_size=64,
                 batch_size=4,
                 gamma=0.9,
                 dis_mode='abs',
                 channel_wise=False):
        super(StyleRepresentation, self).__init__()
        self.num_prototype = num_prototype
        self.channel_size = channel_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.dis_mode = dis_mode
        self.channel_wise = channel_wise
        self.style_mu = nn.Parameter(torch.zeros(self.num_prototype, self.channel_size),
                                     requires_grad=True)
        self.style_sig = nn.Parameter(torch.ones(self.num_prototype, self.channel_size),
                                      requires_grad=True)
        trunc_normal_(self.style_mu, std=0.02)

    def abs_distance(self, cur_mu, cur_sig, proto_mu, proto_sig, batch):
        cur_mu = cur_mu.view(batch, 1, self.channel_size)
        cur_sig = cur_sig.view(batch, 1, self.channel_size)
        proto_mu = proto_mu.view(1, self.num_prototype, self.channel_size)
        proto_sig = proto_sig.view(1, self.num_prototype, self.channel_size)

        cur_mu_sig = cur_mu / cur_sig
        proto_mu_sig = proto_mu / proto_sig
        distance = torch.abs(cur_mu_sig - proto_mu_sig)
        return distance

    def was_distance(self, cur_mu, cur_sig, proto_mu, proto_sig, batch):
        cur_mu = cur_mu.view(batch, 1, self.channel_size)
        cur_sig = cur_sig.view(batch, 1, self.channel_size)
        proto_mu = proto_mu.view(1, self.num_prototype, self.channel_size)
        proto_sig = proto_sig.view(1, self.num_prototype, self.channel_size)

        distance = (cur_mu - proto_mu).pow(2) + (cur_sig.pow(2) + proto_sig.pow(2) - 2 * cur_sig * proto_sig)
        return distance

    def kl_distance(self, cur_mu, cur_sig, proto_mu, proto_sig, batch):
        cur_mu = cur_mu.view(batch, 1, self.channel_size)
        cur_sig = cur_sig.view(batch, 1, self.channel_size)
        proto_mu = proto_mu.view(1, self.num_prototype, self.channel_size)
        proto_sig = proto_sig.view(1, self.num_prototype, self.channel_size)

        cur_mu = cur_mu.expand(-1, self.num_prototype, -1).reshape(batch * self.num_prototype, -1)
        cur_sig = cur_sig.expand(-1, self.num_prototype, -1).reshape(batch * self.num_prototype, -1)
        proto_mu = proto_mu.expand(batch, -1, -1).reshape(batch * self.num_prototype, -1)
        proto_sig = proto_sig.expand(batch, -1, -1).reshape(batch * self.num_prototype, -1)

        cur_distribution = torch.distributions.Normal(cur_mu, cur_sig)
        proto_distribution = torch.distributions.Normal(proto_mu, proto_sig)

        distance = torch.distributions.kl_divergence(cur_distribution, proto_distribution)
        distance = distance.reshape(batch, self.num_prototype, -1)
        return distance

    def forward(self, fea):
        batch = fea.shape[0]
        proto_mu = self.style_mu.data.clone()  #[self.num_prototype:数据集域个数,c]
        proto_sig = self.style_sig.data.clone()

        cur_mu, cur_sig = calculate_mu_sig(fea)  #[b,channel]


        if self.dis_mode == 'abs':
            distance = self.abs_distance(cur_mu, cur_sig, proto_mu, proto_sig, batch)
        elif self.dis_mode == 'was':
            distance = self.was_distance(cur_mu, cur_sig, proto_mu, proto_sig, batch)  #[batch,self.num_prototype, self.channel_size]
        elif self.dis_mode == 'kl':
            distance = self.kl_distance(cur_mu, cur_sig, proto_mu, proto_sig, batch)
        else:
            raise NotImplementedError('No this distance mode!')

        if not self.channel_wise:  #距离差均值，后续加权使用
            distance = distance.mean(dim=2) #[batch,self.num_prototype]

        alpha = 1.0 / (1.0 + distance)  #归一化数值
        # alpha = torch.exp(alpha) / torch.sum(torch.exp(alpha), dim=1, keepdim=True)
        alpha = F.softmax(alpha, dim=1)  #重新映射到0-1 #[batch,self.num_prototype]

        #
        if not self.channel_wise: #计算需要调制的 mu和sig
            # [batch,self.num_prototype] *  [self.num_prototype:数据集域个数,c] = [batch,channel]
            # 对不同域的分割特征进行加权
            mixed_mu = torch.mm(alpha, proto_mu)
            mixed_sig = torch.mm(alpha, proto_sig)

        else:
            proto_mu = proto_mu[None, ...]
            proto_sig = proto_sig[None, ...]
            mixed_mu = torch.sum(alpha * proto_mu, dim=1)
            mixed_sig = torch.sum(alpha * proto_sig, dim=1)


        #使用更新的均值方差，缩放提取特征，实现特征的映射(原版，全局更新)
        fea = ((fea - cur_mu[:, :, None, None]) / cur_sig[:, :, None, None]) * mixed_sig[:, :, None, None] + mixed_mu[:,
                                                                                                             :, None,
                                                                                                             None]
        #使用更新的均值方差，缩放提取特征，实现特征的映射(仅更新对应的源域特征，增广域不进行映射)（效果不好不考虑）
        #fea:[b,c,h,w]
        # if self.training:
        #     source_fea = fea[0:batch// 2,:,:,:]
        #     target_fea = fea[2:batch,:,:,:]
        #
        #     source_fea = ((source_fea - cur_mu[0:batch// 2, :, None, None]) / cur_sig[0:batch// 2, :, None, None]) * mixed_sig[0:batch// 2, :, None, None] + mixed_mu[0:batch// 2,
        #                                                                                                          :, None,
        #                                                                                                          None]
        #     fea = torch.cat([source_fea,target_fea],dim=0)
        # else:
        #     fea = ((fea - cur_mu[:, :, None, None]) / cur_sig[:, :, None, None]) * mixed_sig[:, :, None, None] + mixed_mu[:,
        #                                                                                                      :, None,
        #                                                                                                      None]
        if self.training: #更新学习的mu和sig
            #学习得到的mu和sig
            proto_mu_update = self.style_mu.data.clone()
            proto_sig_update = self.style_sig.data.clone()

            for dataset_id in range(self.num_prototype):
                #按batch_size分别更新对应数据集的mu和sig
                #假设batch_size = 2 , dateset_id =2 ，则训练一个batch总共输入4张图，前两张为第一域，后两张为第二域
                mu = cur_mu[dataset_id * self.batch_size:(dataset_id + 1) * self.batch_size, ...].mean(dim=0)
                sig = cur_sig[dataset_id * self.batch_size:(dataset_id + 1) * self.batch_size, ...].mean(dim=0)

                proto_mu_update[dataset_id] = momentum_update(old_value=proto_mu_update[dataset_id], new_value=mu,
                                                              momentum=self.gamma)
                proto_sig_update[dataset_id] = momentum_update(old_value=proto_sig_update[dataset_id], new_value=sig,
                                                               momentum=self.gamma)

            self.style_mu = nn.Parameter(proto_mu_update, requires_grad=False)
            self.style_sig = nn.Parameter(proto_sig_update, requires_grad=False)

            if dist.is_available() and dist.is_initialized():
                proto_mu = self.style_mu.data.clone()
                dist.all_reduce(proto_mu.div_(dist.get_world_size()))
                self.style_mu = nn.Parameter(proto_mu, requires_grad=False)

                proto_sig = self.style_sig.data.clone()
                dist.all_reduce(proto_sig.div_(dist.get_world_size()))
                self.style_sig = nn.Parameter(proto_sig, requires_grad=False)
        #
        return fea