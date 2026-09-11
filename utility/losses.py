import torch
import torch.nn.functional as F


_EPS = 1e-8


def get_bpr_loss(user_embed, pos_embed, neg_embed):
    pos_scores = torch.sum(user_embed * pos_embed, dim=1)
    neg_scores = torch.sum(user_embed * neg_embed, dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + _EPS)
    return torch.mean(loss)


def get_weighted_bpr_loss(user_embed, pos_embed, neg_embed, pos_weight):
    """BPR with optional inverse-popularity weights on positive items.

    This is intentionally not the main contribution. It is provided as an
    ablation switch (--pop_bpr 1). We normalize weights inside each batch to
    avoid changing the global loss scale.
    """
    pos_scores = torch.sum(user_embed * pos_embed, dim=1)
    neg_scores = torch.sum(user_embed * neg_embed, dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + _EPS)
    weight = pos_weight.detach()
    weight = weight / torch.clamp(weight.mean(), min=_EPS)
    return torch.mean(weight * loss)


def get_reg_loss(*embeddings):
    reg_loss = 0
    for embedding in embeddings:
        reg_loss += 1 / 2 * embedding.norm(2).pow(2) / float(embedding.shape[0])
    return reg_loss


def get_InfoNCE_loss(embedding1, embedding2, temperature):
    embedding1 = F.normalize(embedding1, dim=-1)
    embedding2 = F.normalize(embedding2, dim=-1)

    pos_score = (embedding1 * embedding2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    total_score = torch.matmul(embedding1, embedding2.transpose(0, 1))
    total_score = torch.exp(total_score / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / (total_score + _EPS) + 1e-6)
    return torch.mean(cl_loss)


def get_neighbor_aggregate_loss(embedding1, embedding2, tau):
    """Original LightCCF NA loss.

    Kept for exact baseline compatibility. The original implementation uses
    exp((U I^T + U U^T) / tau) as the denominator score matrix.
    """
    embedding1 = F.normalize(embedding1, dim=-1)
    embedding2 = F.normalize(embedding2, dim=-1)

    pos_score = (embedding1 * embedding2).sum(dim=-1)
    pos_score = torch.exp(pos_score / tau)

    total_score = torch.matmul(embedding1, embedding2.transpose(0, 1)) + torch.matmul(embedding1, embedding1.transpose(0, 1))
    total_score = torch.exp(total_score / tau).sum(dim=1)

    na_loss = -torch.log(pos_score / (total_score + _EPS) + 1e-6)
    return torch.mean(na_loss)


def get_popularity_balanced_na_loss(user_embed,
                                    pos_embed,
                                    tau,
                                    pos_weight=None,
                                    group_id=None,
                                    gb_mix=0.0):
    """Popularity-aware Group-Balanced Neighborhood Aggregation loss.

    This loss modifies LightCCF at its core NA objective rather than adding an
    external regularizer.

    1) Positive attraction is reweighted by item inverse-popularity:
       tail items receive larger but clipped gradients, head items receive
       smaller but non-zero gradients.

    2) Denominator can be group-balanced by item popularity groups. In original
       LightCCF, all batch positive pairs enter the denominator uniformly; in a
       long-tailed batch, head-item pairs may dominate the contrastive repulsion.
       The balanced denominator gives head/middle/tail groups comparable total
       contribution. gb_mix mixes original and balanced denominators for stable
       first-stage experiments.

    Args:
        user_embed: [B, d] user representations.
        pos_embed: [B, d] positive item representations.
        tau: temperature.
        pos_weight: [B] weight of the positive item. If None, all ones.
        group_id: [B] item popularity group id, 0=tail, 1=middle, 2=head.
        gb_mix: 0 keeps original denominator; 1 uses pure group-balanced one.
    """
    user_embed = F.normalize(user_embed, dim=-1)
    pos_embed = F.normalize(pos_embed, dim=-1)

    batch_size = user_embed.shape[0]

    # Positive user-item attraction, same score as original LightCCF but written
    # in log form to make weighting mathematically clear and stable.
    pos_logit = (user_embed * pos_embed).sum(dim=-1) / tau
    if pos_weight is None:
        pos_weight = torch.ones_like(pos_logit)
    else:
        pos_weight = pos_weight.to(pos_logit.device).float()
        # Batch normalization keeps the loss scale close to original LightCCF.
        pos_weight = pos_weight / torch.clamp(pos_weight.mean(), min=_EPS)

    # Original denominator score matrix, aligned with LightCCF's implementation.
    score_mat = torch.matmul(user_embed, pos_embed.transpose(0, 1)) + torch.matmul(user_embed, user_embed.transpose(0, 1))
    exp_score = torch.exp(score_mat / tau)
    original_denom = exp_score.sum(dim=1)

    gb_mix = float(gb_mix)
    if group_id is None or gb_mix <= 0.0:
        denom = original_denom
    else:
        group_id = group_id.to(exp_score.device).long()
        balanced_denom = torch.zeros_like(original_denom)
        active_groups = 0
        for g in (0, 1, 2):
            mask = group_id.eq(g)
            group_count = int(mask.sum().item())
            if group_count > 0:
                # Re-scale each group to an equal virtual batch share.
                # This preserves denominator magnitude while equalizing group
                # gradient contribution.
                balanced_denom += exp_score[:, mask].sum(dim=1) * (batch_size / float(group_count))
                active_groups += 1
        if active_groups > 0:
            balanced_denom = balanced_denom / float(active_groups)
            denom = (1.0 - gb_mix) * original_denom + gb_mix * balanced_denom
        else:
            denom = original_denom

    loss = -pos_weight * pos_logit + torch.log(denom + _EPS)
    return torch.mean(loss)
