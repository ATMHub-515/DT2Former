import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, normal_
from typing import List
from .deformable_encoder import DeformableEncoderLayer, DeformableEncoder
from .deformable_decoder import DeformableDecoderLayer, DeformableDecoder
from .ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        n_feature_levels=4,
        n_heads=8,
        n_enc_points=4,
        n_dec_points=4,
        n_enc_layers=6,
        n_dec_layers=6,
        merge_det_track_layer=0,
        dropout=0.1,
        activation="ReLU",
        return_intermediate_dec=False,
        n_det_queries=300,
        extra_track_attn=False,
        two_stage=False,
        two_stage_num_proposals=300,
        use_checkpoint: bool = False,
        checkpoint_level: int = 2,
        use_dab: bool = False,
        visualize: bool = False,
    ):
        super(DeformableTransformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize
        encoder_layer = DeformableEncoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            n_levels=n_feature_levels,
            n_heads=n_heads,
            n_points=n_enc_points,
            sigmoid_attn=False,
        )
        decoder_layer = DeformableDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            n_levels=n_feature_levels,
            n_heads=n_heads,
            n_points=n_dec_points,
            sigmoid_attn=False,
            extra_track_attn=extra_track_attn,
            n_det_queries=n_det_queries,
            visualize=self.visualize,
        )
        self.encoder: DeformableEncoder = DeformableEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_enc_layers,
            use_checkpoint=self.use_checkpoint and self.checkpoint_level == 1,
        )
        self.decoder: DeformableDecoder = DeformableDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_dec_layers,
            return_intermediate=return_intermediate_dec,
            merge_det_track_layer=merge_det_track_layer,
            n_det_queries=n_det_queries,
            d_model=self.d_model,
            use_checkpoint=self.use_checkpoint,
            use_dab=self.use_dab,
            visualize=self.visualize,
        )
        self.level_embed = nn.Parameter(torch.Tensor(n_feature_levels, d_model))
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        elif use_dab:
            pass
        else:
            self.reference_points = nn.Linear(d_model, 2)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m.reset_parameters()
        if not self.two_stage:
            if self.use_dab:
                pass
            else:
                xavier_uniform_(self.reference_points.weight.data, gain=1.0)
                constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        srcs: List[torch.Tensor],
        masks: List[torch.Tensor],
        pos_embeds: List[torch.Tensor],
        query_embed,
        ref_pts,
        query_mask,
    ):
        assert self.two_stage or query_embed is not None
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            spatial_shapes.append(spatial_shape)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        if self.use_checkpoint and (self.checkpoint_level == 2 or self.checkpoint_level == 3):
            from torch.utils.checkpoint import checkpoint

            memory = checkpoint(
                self.encoder,
                src_flatten,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                lvl_pos_embed_flatten,
                mask_flatten,
                use_reentrant=False,
            )
        else:
            memory = self.encoder(
                src=src_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                pos=lvl_pos_embed_flatten,
                padding_mask=mask_flatten,
            )
        bs, _, c = memory.shape
        if self.two_stage:
            raise RuntimeError("Do not support two stage model for Deformable Transformer.")
        else:
            if self.use_dab:
                tgt = query_embed
                query_embed = None
            else:
                query_embed, tgt = torch.split(query_embed, c, dim=2)
            assert ref_pts is not None, "ref_pts should not be None."
            reference_points = ref_pts.sigmoid()
            init_reference_points = reference_points
        output, res_reference_points, inter_queries = self.decoder(
            tgt=tgt,
            reference_points=init_reference_points,
            src=memory,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=query_embed,
            query_mask=query_mask,
            src_padding_mask=mask_flatten,
        )
        return (output, init_reference_points, res_reference_points, inter_queries)

    def get_d_model(self):
        return self.d_model

    def get_n_dec_layers(self):
        return self.decoder.num_layers

    def set_refine_bbox_embed(self, bbox_embed: nn.Module):
        self.decoder.bbox_embed = bbox_embed
        return


def build(config: dict):
    return DeformableTransformer(
        d_model=config["HIDDEN_DIM"],
        d_ffn=config["FFN_DIM"],
        n_feature_levels=config["NUM_FEATURE_LEVELS"],
        n_heads=config["NUM_HEADS"],
        n_enc_points=config["NUM_ENC_POINTS"],
        n_dec_points=config["NUM_DEC_POINTS"],
        n_enc_layers=config["NUM_ENC_LAYERS"],
        n_dec_layers=config["NUM_DEC_LAYERS"],
        merge_det_track_layer=(
            0 if "MERGE_DET_TRACK_LAYER" not in config else config["MERGE_DET_TRACK_LAYER"]
        ),
        dropout=config["DROPOUT"],
        activation=config["ACTIVATION"],
        return_intermediate_dec=config["RETURN_INTER_DEC"],
        n_det_queries=config["NUM_DET_QUERIES"],
        extra_track_attn=config["EXTRA_TRACK_ATTN"],
        two_stage=False,
        use_checkpoint=config["USE_CHECKPOINT"],
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"],
        visualize=config["VISUALIZE"],
    )
