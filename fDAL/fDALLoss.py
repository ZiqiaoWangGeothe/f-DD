# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F
import torch
from .utils import ConjugateDualFunction

__all__ = ["fDALLoss"]


class fDALLoss(nn.Module):
    def __init__(self, divergence_name, gamma):
        super(fDALLoss, self).__init__()

        self.lhat = None
        self.phistar = None
        self.phistar_gf = None
        self.multiplier = 1.
        self.internal_stats = {}
        self.domain_discriminator_accuracy = -1
        self.dg_name=divergence_name

        self.gammaw = gamma
        self.phistar_gf = lambda t: ConjugateDualFunction(divergence_name).fstarT(t)
        self.gf = lambda v: ConjugateDualFunction(divergence_name).T(v)

    def forward(self, y_s, y_t, y_s_adv, y_t_adv, K, t_star):
        # ---
        #
        #

        v_s = y_s_adv
        v_t = y_t_adv

        if K > 1:
            _, prediction_s = y_s.max(dim=1)
            _, prediction_t = y_t.max(dim=1)
            # This is not used here as a loss, it just a way to pick elements.

            # picking element prediction_s k element from y_s_adv.
            v_s = -F.nll_loss(v_s, prediction_s.detach(), reduction='none')
            # picking element prediction_t k element from y_t_adv.
            v_t = -F.nll_loss(v_t, prediction_t.detach(), reduction='none')

            # if t_star > 1 or t_star < 1:
            #     v_s= v_s * t_star
            #     v_t = v_t * t_star
            # else:
            #     v_s = v_s
            #     v_t = v_t

        if self.dg_name == "kl":
            #DV
            dst = self.gammaw * torch.mean(self.gf(v_s)) - torch.log(torch.mean(torch.exp(v_t)))
            # dst = self.gammaw * (torch.mean(self.gf(v_s)) * 0.95 + torch.mean(self.gf(v_t)) * 0.05) - torch.log(torch.mean(torch.exp(v_t))) * 0.95-torch.log(torch.mean(torch.exp(v_s))) * 0.05

        elif self.dg_name == "klabs":
            dst = torch.abs(torch.mean(self.gf(v_s)) - torch.log(torch.mean(torch.exp(v_t))))

        elif self.dg_name == "klrev":

            dst = self.gammaw * torch.mean(self.gf(v_t)) - torch.log(torch.mean(torch.exp(v_s)))

        elif self.dg_name == "optkl":
            tilted = torch.mean(torch.exp(v_t))
            pphi_mean = torch.mean(torch.exp(v_t)/tilted)
            pphi_var = torch.var(torch.exp(v_t)/tilted)
            numerator = torch.mean(v_s) - pphi_mean
            scale = 1 + numerator/pphi_var
            dst = scale * torch.mean(v_s) - torch.log(torch.mean(torch.exp(scale * v_t)))

        elif self.dg_name == "chi":
            v_mean = torch.mean(v_t)
            v_var = torch.mean((v_t-v_mean)**2)
            # sq = (torch.mean(v_s) - v_mean)**2
            # dst = sq / v_var
            dst = torch.mean(v_s) - v_mean - v_var/4

        elif self.dg_name == "chiabs":
            dst = torch.abs(torch.mean(self.gf(v_s)) - torch.mean(self.phistar_gf(v_t)))

        elif self.dg_name == "optchi":
            vt_mean = torch.mean(v_t)
            vt_var = torch.var(v_t)

            dst = torch.square(torch.mean(v_s) - vt_mean) / vt_var

            # vs_mean = torch.mean(v_s)
            # vs_var = torch.var(v_s)
            #
            # dst = torch.square(torch.mean(v_t) - vs_mean) / vs_var

        else:
            dst = self.gammaw * torch.mean(self.gf(v_s)) - torch.mean(self.phistar_gf(v_t))

        self.internal_stats['lhatsrc'] = torch.mean(v_s).item()
        self.internal_stats['lhattrg'] = torch.mean(v_t).item()
        self.internal_stats['acc'] = self.domain_discriminator_accuracy
        self.internal_stats['dst'] = dst.item()

        # we need to negate since the obj is being minimized, so min -dst =max dst.
        # the gradient reversar layer will take care of the rest
        return -self.multiplier * dst
