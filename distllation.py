# losses.py

# ==============================================================================
# Required Imports
# ==============================================================================
import torch.nn.functional as F

# ==============================================================================
# Distillation Loss Functions
# ==============================================================================

# --- Type 1: Response-Based (Logits) Distillation ---
def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    """
    Calculates the response-based distillation loss.
    This loss is a weighted average of a "soft" loss (KL divergence between student
    and teacher logits) and a "hard" loss (cross-entropy with ground truth labels).

    Args:
        student_logits (torch.Tensor): The raw output logits from the student model.
        teacher_logits (torch.Tensor): The raw output logits from the teacher model.
        labels (torch.Tensor): The ground truth labels.
        alpha (float): The weight for the hard loss component.
        temperature (float): The temperature to soften the probability distributions.

    Returns:
        torch.Tensor: The final calculated response-based distillation loss.
    """
    # Soft loss (Kullback-Leibler Divergence)
    # The temperature softens the probabilities, encouraging the student to learn the
    # relative similarities in the teacher's predictions.
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss_soft = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    # Hard loss (Cross-Entropy)
    # This ensures the student still learns from the true labels.
    loss_hard = F.cross_entropy(student_logits, labels)

    # Combine the two losses
    return alpha * loss_hard + (1. - alpha) * loss_soft

# --- Type 2: Feature-Based Distillation ---
def feature_alignment_loss(student_features, teacher_features):
    """
    Calculates the feature-based distillation loss.
    This loss is the Mean Squared Error (MSE) between the feature representations
    of the student and teacher models, encouraging the student to mimic the
    teacher's internal representations.

    Args:
        student_features (torch.Tensor): The feature vectors from the student model.
        teacher_features (torch.Tensor): The feature vectors from the teacher model.

    Returns:
        torch.Tensor: The final calculated feature-based distillation loss.
    """
    return F.mse_loss(student_features, teacher_features)