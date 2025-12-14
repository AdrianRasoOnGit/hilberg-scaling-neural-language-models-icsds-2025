
import math
import numpy as np
import torch

from src.train.train_all_models import main as run_training

run_training()

@torch.no_grad()
def compute_model_loss_at_context(model, tokens, context_len, device, max_eval_tokens=200_000):
    """
    Compute model conditional entropy H(X_t | X_{t-context_len}^{t-1})
    by manually truncating the context at inference time.
    """
    model.eval()

    L = len(tokens)
    end = min(L - context_len - 1, max_eval_tokens)

    total_loss = 0.0
    count = 0

    for i in range(0, end, context_len):
        x = tokens[i:i+context_len]
        y = tokens[i+1:i+context_len+1]

        xb = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
        yb = torch.tensor(y, dtype=torch.long, device=device).unsqueeze(0)

        _, loss = model(xb, yb)
        total_loss += loss.item()
        count += 1

    avg_nats = total_loss / count
    return avg_nats / math.log(2)

def fit_model_exponent(context_values, loss_values):
    """
    Fit power-law exponent α from H(n) ∝ n^(α−1).
    """
    xs = [math.log(n) for n in context_values]
    ys = [math.log(Hn) for Hn in loss_values]

    alpha_slope, _ = np.polyfit(xs, ys, 1)
    beta = alpha_slope + 1

    return float(alpha_slope), float(beta)

def evaluate_model_scaling(model, eval_tokens, max_content, device):
    """
    Evaluate H(n) for n = 1, 2, 4, ..., max_context
    and compute Hilberg-style scaling exponents α, β.

    Returns:
        losses: dict {context_len: H(context_len)}
        alpha: power-law slope
        beta: alpha + 1 (Hilberg exponent form)
    """

    context_probe = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    valid_contexts = [n for n in context_probe if n <= max_content]

    losses = {}
    for n in valid_contexts:
        H_n = compute_model_loss_at_context(
            model=model,
            tokens=eval_tokens,
            context_len=n,
            device=device
        )
        losses[n] = H_n

    alpha, beta = fit_model_exponent(
        valid_contexts,
        [losses[n] for n in valid_contexts]
    )

    return losses, alpha, beta
