import params
import os
import numpy as np
import torch
import random
from utils import load_data
from model.pretrain import pretrain_on_source
from model.adapt import adaptation

def init_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
    init_random_seed(params.seed)

    if not os.path.exists(params.output_dir):
        os.mkdir(params.output_dir)

    output_dir = os.path.join(params.output_dir, params.mode)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get source and target dataset
    src_data_loader, target_data_loader = load_data(params.mode, train=True)
    src_data_loader_eval, target_data_loader_eval = load_data(params.mode, train=False)

    # Pretrain on source dataset to get the source model(netF, netB, netC)
    pretrain_on_source(src_data_loader, src_data_loader_eval, output_dir)

    # train target model(netC fixed, update netF&B)
    adaptation(target_data_loader, target_data_loader_eval, output_dir)