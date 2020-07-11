from tqdm import tqdm

import network
import params
import torch.optim as optim
import os
import torch


def pretrain_on_source(src_data_loader, src_data_loader_eval, output_dir):
    ## set base network
    if params.mode == 'u2m':
        netF = network.LeNetBase().cuda()
    elif params.mode == 'm2u':
        netF = network.LeNetBase().cuda()
    elif params.mode == 's2m':
        netF = network.DTNBase().cuda()
    netB = network.feat_bootleneck(type=params.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=params.bottleneck).cuda()
    netC = network.feat_classifier(type=params.layer, class_num=params.class_num, bottleneck_dim=params.bottleneck).cuda()

    param_group = []
    learning_rate = params.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)

    acc_init = 0
    out_file = open(os.path.join(output_dir, 'log_pretrain.txt'), 'w')
    for epoch in range(params.epochs):
        # scheduler.step()
        netF.train()
        netB.train()
        netC.train()
        iter_source = iter(src_data_loader)
        for _, (inputs_source, labels_source) in enumerate(iter_source):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            outputs_source = netC(netB(netF(inputs_source)))
            classifier_loss = network.CrossEntropyLabelSmooth(num_classes=params.class_num, epsilon=params.smooth)(outputs_source,
                                                                                                       labels_source)
            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

        netF.eval()
        netB.eval()
        netC.eval()
        acc_s_tr, _ = network.cal_acc(src_data_loader, netF, netB, netC)
        acc_s_te, _ = network.cal_acc(src_data_loader_eval, netF, netB, netC)
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%'.format(params.mode, epoch + 1, params.epochs,
                                                                             acc_s_tr * 100, acc_s_te * 100)
        out_file.write(log_str + '\n')
        out_file.flush()
        print(log_str + '\n')

        if acc_s_te >= acc_init:
            acc_init = acc_s_te
            best_netF = netF.state_dict()
            best_netB = netB.state_dict()
            best_netC = netC.state_dict()

    torch.save(best_netF, os.path.join(output_dir, "source_F_val.pt"))
    torch.save(best_netB, os.path.join(output_dir, "source_B_val.pt"))
    torch.save(best_netC, os.path.join(output_dir, "source_C_val.pt"))

    return netF, netB, netC
