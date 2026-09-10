import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

def save_model(model, model_save_path, model_name, epoch, optimizer, scheduler):
    """
    Function to save training model
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        },
        model_save_path + "_" + model_name + ".pth")


def update_best_metric(new_value, current_best, mode: str = "max"):
    changed_value = False
    
    if mode == "max" and (new_value > current_best):
        current_best = new_value
        changed_value = True
        
    if mode == "min" and (new_value < current_best):
        current_best = new_value
        changed_value = True
    
    return changed_value, current_best

def setup_scheduler(optimizer, num_epoch):
    """Linear warmup then cosine decay. Warmup shrinks for short runs so the
    cosine phase always has at least one epoch (T_max >= 1) - otherwise
    CosineAnnealingLR divides by zero."""
    warmup_epochs = min(5, max(1, num_epoch // 10))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, num_epoch - warmup_epochs), eta_min=1e-6
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

def plot_training_loss(train_loss, valid_loss, save_path="training_loss.pdf"):
    
    plt.plot(np.arange(1, len(train_loss)+1), train_loss, label="training_loss")
    plt.plot(np.arange(1, len(valid_loss)+1), valid_loss, label="validation_loss")
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

