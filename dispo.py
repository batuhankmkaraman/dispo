import torch

def compute_dispo_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    eps_low_pos=1e9,
    eps_high_pos=5.0,
    eps_low_neg=1e9,
    eps_high_neg=5.0,
):
    """
    Naive DISPO loss:
      - separate clipping for positive vs negative advantages
      - detached clipped importance weights

    Args:
        old_log_prob: (bs, T)
        log_prob:     (bs, T)
        advantages:   (bs, T)
        response_mask:(bs, T)  # 1 for valid tokens, 0 otherwise

        eps_low_pos / eps_high_pos:
            clipping for positive-advantage tokens
            ratio clipped to [1 - eps_low_pos, 1 + eps_high_pos]

        eps_low_neg / eps_high_neg:
            clipping for negative-advantage tokens
            ratio clipped to [1 - eps_low_neg, 1 + eps_high_neg]

    Returns:
        pg_loss: scalar
        ratio: (bs, T)
        is_weight: (bs, T)
    """

    ratio = torch.exp(log_prob - old_log_prob)

    pos_mask = (advantages > 0).float()
    neg_mask = (advantages < 0).float()

    is_weight_pos = torch.clamp(
        ratio,
        min=1.0 - eps_low_pos,
        max=1.0 + eps_high_pos,
    )

    is_weight_neg = torch.clamp(
        ratio,
        min=1.0 - eps_low_neg,
        max=1.0 + eps_high_neg,
    )

    is_weight = pos_mask * is_weight_pos + neg_mask * is_weight_neg
    is_weight = is_weight.detach()

    pg_losses = -is_weight * advantages * log_prob

    denom = response_mask.sum().clamp_min(1.0)
    pg_loss = (pg_losses * response_mask).sum() / denom

    return pg_loss, ratio, is_weight