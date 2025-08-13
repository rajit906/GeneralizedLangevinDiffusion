# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import os
import logging

def fix_optimizer_state(state_dict):
    for param_id, param_state in state_dict.items():
        if param_state and 'step' not in param_state:
            param_state['step'] = torch.tensor(0)
    return state_dict

def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        logging.warning(f'No checkpoint found at {ckpt_dir}. Returned the same state as input.')
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)

        # Patch optimizer state before loading
        loaded_state['optimizer']['state'] = fix_optimizer_state(loaded_state['optimizer']['state'])

        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
