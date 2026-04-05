import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ASPP import ASPP
from model.backbone_utils import Backbone
from model.loss import WeightedDiceLoss



class SIAM(nn.Module):
    def __init__(self, channels, r=16, tau=0.05, k=5):
        super(SIAM, self).__init__()
        self.channels = channels
        self.tau = tau
        self.k = k
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU6(inplace=True),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, F_s, M_s, F_q):
        B, C, H, W = F_s.shape
        N = H * W
        M_s_tilde = F.interpolate(M_s, size=(H, W), mode='bilinear', align_corners=True)
        F_s_fg = F_s * M_s_tilde
        gap = F.adaptive_avg_pool2d(F_s, (1, 1)).view(B, C)
        g = self.gate(gap).view(B, C, 1, 1)
        F_s_ctx = g * F_s
        F_q_norm = F.normalize(F_q.view(B, C, N), p=2, dim=1)
        F_s_fg_norm = F.normalize(F_s_fg.view(B, C, N), p=2, dim=1)
        F_s_ctx_norm = F.normalize(F_s_ctx.view(B, C, N), p=2, dim=1)
        dist_fg = torch.bmm(F_q_norm.transpose(1, 2), F_s_fg_norm) / self.tau
        dist_ctx = torch.bmm(F_q_norm.transpose(1, 2), F_s_ctx_norm) / self.tau
        max_fg, _ = dist_fg.max(dim=-1)
        max_ctx, _ = dist_ctx.max(dim=-1)
        max_fg = max_fg.view(B, 1, H, W)
        max_ctx = max_ctx.view(B, 1, H, W)
        pad = self.k // 2
        A_fg = F.avg_pool2d(max_fg, kernel_size=self.k, stride=1, padding=pad)
        A_ctx = F.avg_pool2d(max_ctx, kernel_size=self.k, stride=1, padding=pad)
        M_coarse = self.fusion(torch.cat([A_fg, A_ctx], dim=1))
        M_coarse = torch.sigmoid(M_coarse)
        return M_coarse, F_s_fg


class ACPCM(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(ACPCM, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, F_s_fg, M_s_tilde, F_q, M_coarse):
        B, C, H, W = F_q.shape
        P_s = (F_s_fg.sum(dim=(2, 3))) / (M_s_tilde.sum(dim=(2, 3)) + self.eps)
        P_q = (F_q * M_coarse).sum(dim=(2, 3)) / (M_coarse.sum(dim=(2, 3)) + self.eps)
        F_q_norm = F.normalize(F_q, p=2, dim=1)
        P_s_norm = F.normalize(P_s, p=2, dim=1).view(B, C, 1, 1)
        P_q_norm = F.normalize(P_q, p=2, dim=1).view(B, C, 1, 1)
        S_self = (F_q_norm * P_q_norm).sum(dim=1, keepdim=True)
        S_cross = (F_q_norm * P_s_norm).sum(dim=1, keepdim=True)
        S = S_self + S_cross
        F_q_prime = F_q + self.gamma * (S * P_s.view(B, C, 1, 1))
        return F_q_prime







def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.criterion_dice = WeightedDiceLoss()
        self.print_freq = args.print_freq / 2
        self.pretrained = True
        self.classes = 2

        assert self.layers in [50, 101, 152]
        self.backbone = Backbone('resnet{}'.format(self.layers), train_backbone=False,
                                 return_interm_layers=True, dilation=[False, True, True])

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        embed_dim = reduce_dim
        self.init_merge_query = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 1, embed_dim, kernel_size=1, padding=0, bias=False)
        )
        self.init_merge_supp = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 1, embed_dim, kernel_size=1, padding=0, bias=False)
        )

        # swin transformer
        depths = (8,)
        num_heads = (8,)
        window_size = 8
        mlp_ratio = 1.
        self.window_size = window_size
        pretrain_img_size = 64


        scale = 0
        for i in range(len(depths)):
            scale += 2 ** i
        self.ASPP_meta = ASPP(scale * embed_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(scale * embed_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge_query.parameters()},
                {'params': model.init_merge_supp.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},
                {'params': model.cls_meta.parameters()},
            ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_swin = torch.optim.AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if "transformer" in n and p.requires_grad]}
            ], lr=6e-5
        )
        return optimizer, optimizer_swin

    def freeze_modules(self, model):
        for param in model.backbone.parameters():
            param.requires_grad = False

    def generate_prior(self, query_feat_high, final_supp_list, mask_list, fts_size):
        bsize, ch_sz, sp_sz, _ = query_feat_high.size()[:]
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat = tmp_supp_feat
            q = query_feat_high.flatten(2).transpose(-2, -1)
            s = tmp_supp_feat.flatten(2).transpose(-2, -1)

            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.permute(0, 2, 1)
            similarity = F.softmax(similarity, dim=-1)
            similarity = torch.bmm(similarity, tmp_mask.flatten(2).transpose(-2, -1)).squeeze(-1)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=fts_size, mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (corr_query_mask).mean(1, True)
        return corr_query_mask

    # que_img, sup_img, sup_mask, que_mask(meta), cat_idx(meta)
    def forward(self, x, s_x, s_y, y_m, cat_idx=None):
        x_size = x.size()  # bs, 3, 473, 473
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)  # (473 - 1) / 8 * 8 + 1 = 60
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)  # 60

        # Interpolation size for pascal/coco
        size = (64, 64)

        # ========================================
        # Feature Extraction - Query/Support
        # ========================================
        # Query/Support Feature
        with torch.no_grad():
            qry_bcb_fts = self.backbone(x)
            supp_bcb_fts = self.backbone(s_x.view(-1, 3, x_size[2], x_size[3]))

        query_feat_high = qry_bcb_fts['3']

        query_feat = torch.cat([qry_bcb_fts['1'], qry_bcb_fts['2']], dim=1)
        query_feat = self.down_query(query_feat)
        query_feat = F.interpolate(query_feat, size=size, mode='bilinear', align_corners=True)
        fts_size = query_feat.size()[-2:]
        supp_feat = torch.cat([supp_bcb_fts['1'], supp_bcb_fts['2']], dim=1)
        supp_feat = self.down_supp(supp_feat)
        supp_feat = F.interpolate(supp_feat, size=size, mode='bilinear', align_corners=True)

        mask_list = []
        supp_pro_list = []
        supp_feat_list = []
        final_supp_list = []
        supp_feat_mid = supp_feat.view(bs, self.shot, -1, fts_size[0], fts_size[1])
        supp_bcb_fts['3'] = F.interpolate(supp_bcb_fts['3'], size=size, mode='bilinear', align_corners=True)
        supp_feat_high = supp_bcb_fts['3'].view(bs, self.shot, -1, fts_size[0], fts_size[1])
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask = F.interpolate(mask, size=fts_size, mode='bilinear', align_corners=True)
            mask_list.append(mask)
            final_supp_list.append(supp_feat_high[:, i, :, :, :])
            supp_feat_list.append((supp_feat_mid[:, i, :, :, :] * mask).unsqueeze(-1))
            supp_pro = Weighted_GAP(supp_feat_mid[:, i, :, :, :], mask)
            supp_pro_list.append(supp_pro)

        # Support features/prototypes/masks
        supp_mask = torch.cat(mask_list, dim=1).mean(1, True)  # bs, 1, 60, 60
        supp_feat = torch.cat(supp_feat_list, dim=-1).mean(-1)  # bs, 256, 60, 60
        supp_pro = torch.cat(supp_pro_list, dim=2).mean(2, True)  # bs, 256, 1, 1
        supp_pro = supp_pro.expand_as(query_feat)  # bs, 256, 60, 60

        # Prior Similarity Mask
        corr_query_mask = self.generate_prior(query_feat_high, final_supp_list, mask_list, fts_size)

        # ========================================
        # Cross Swin Transformer
        # ========================================
        # Adapt query/support features with support prototype
        query_cat = torch.cat([query_feat, supp_pro, corr_query_mask], dim=1)  # bs, 512, 60, 60
        query_feat = self.init_merge_query(query_cat)  # bs, 256, 60, 60
        supp_cat = torch.cat([supp_feat, supp_pro, supp_mask], dim=1)  # bs, 512, 60, 60
        supp_feat = self.init_merge_supp(supp_cat)  # bs, 256, 60, 60

        # Swin transformer (cross)
        query_feat_list = []
        query_feat_list.extend(self.transformer(query_feat, supp_feat, supp_mask))
        fused_query_feat = []
        for idx, qry_feat in enumerate(query_feat_list):
            fused_query_feat.append(
                self.relu(
                    F.interpolate(qry_feat, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
                )
            )
        merge_feat = torch.cat(fused_query_feat, dim=1)

        # ========================================
        # Meta Output
        # ========================================
        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)

        # Interpolate
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)

        # ========================================
        # Loss
        # ========================================
        if self.training:
            main_loss = self.criterion_dice(meta_out, y_m.long())
            aux_loss1 = torch.zeros_like(main_loss)
            aux_loss2 = torch.zeros_like(main_loss)
            return meta_out.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return meta_out

class FewShotSegmentor(nn.Module):
    def __init__(self, channels=256):
        super(FewShotSegmentor, self).__init__()
        self.siam = SIAM(channels)
        self.acpcm = ACPCM(channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1)
        )

    def forward(self, F_s, M_s, F_q):
        out_size = F_q.shape[-2:]
        M_coarse, F_s_fg = self.siam(F_s, M_s, F_q)
        M_s_tilde = F.interpolate(M_s, size=out_size, mode='bilinear', align_corners=True)
        F_q_prime = self.acpcm(F_s_fg, M_s_tilde, F_q, M_coarse)
        logits = self.decoder(F_q_prime)
        return logits
