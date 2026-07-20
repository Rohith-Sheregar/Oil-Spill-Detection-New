"""
OOM probing + retry-with-halving for unattended Colab/Kaggle training runs.
An OOM mid-epoch on an unattended session just kills the run silently --
probe the safe batch size once at the start, and wrap the training step so
a late, data-dependent OOM (e.g. an unusually large connected component, or
just GPU memory fragmentation a few hundred steps in) degrades gracefully
instead of ending the whole session.
"""
import torch


def probe_max_batch_size(model, make_batch, start_bs=16, min_bs=1, device="cuda"):
    """make_batch(bs) -> a CPU batch of that size; this moves it to device
    and runs one forward pass. Call once before training starts."""
    bs = start_bs
    while bs >= min_bs:
        try:
            torch.cuda.empty_cache()
            x = make_batch(bs).to(device)
            with torch.no_grad():
                _ = model(x)
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs //= 2
    raise RuntimeError(
        "OOM even at batch size 1 -- reduce patch size (e.g. 256->128) "
        "instead of batch size; batch size 1 is already the floor."
    )


def train_step_with_oom_retry(model, optimizer, loss_fn, batch, min_bs=1):
    """
    Tries the full batch; on OOM, splits it in half and accumulates
    gradients across the two halves so the effective optimizer step matches
    the original batch (just slower for that one step). Re-raises if it
    OOMs even at min_bs -- at that point the problem is patch size, not
    batch size, and silently continuing would hide that.
    """
    x, y = batch
    bs = x.shape[0]
    try:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        return loss.item()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if bs <= min_bs:
            raise
        half = bs // 2
        optimizer.zero_grad()
        total_loss = 0.0
        for sub_x, sub_y in [(x[:half], y[:half]), (x[half:], y[half:])]:
            loss = loss_fn(model(sub_x), sub_y) * (sub_x.shape[0] / bs)
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        return total_loss
