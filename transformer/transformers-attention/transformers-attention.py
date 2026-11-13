import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    QKT = torch.matmul(Q, K.transpose(-2,-1))
    scaled_score = QKT / (Q.shape[-1]**0.5)
    print(scaled_score)
    softmax_score = torch.softmax(scaled_score, 2)
    scaled_dot_product = torch.matmul(softmax_score, V)

    return scaled_dot_product
