import os
import torch

def save_checkpoint(model_path, epoch, model, optimizer, name='ssfsr_9layers', save_optim=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    if save_optim:
        state['optimizer'] = optimizer.state_dict()

    torch.save(state, os.path.join(model_path, name + '.pkl'))