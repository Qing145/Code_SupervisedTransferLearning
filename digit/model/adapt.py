from tqdm import tqdm

import network
import params
import torch.optim as optim
import os
import torch


def adaptation(target_data_loader, target_data_loader_eval, output_dir):
    ## set base network

    #print(output_dir)
    netF = network.LeNetBase().cuda()

    netB = network.feat_bootleneck(type=params.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=params.bottleneck).cuda()
    netC = network.feat_classifier(type=params.layer, class_num=params.class_num, bottleneck_dim=params.bottleneck).cuda()

    modelpath = output_dir + '/source_F_val.pt'
    netF.load_state_dict(torch.load(modelpath))

    modelpath = output_dir + '/source_B_val.pt'
    netB.load_state_dict(torch.load(modelpath))

    modelpath = output_dir + '/source_C_val.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    netF.eval()
    netB.eval()
    acc, _ = network.cal_acc(target_data_loader_eval, netF, netB, netC)
    log_str = 'Task: {}, Before Training!  Accuracy = {:.2f}%'.format(params.mode, acc * 100)
    print(log_str + '\n')
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': params.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': params.lr}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
    out_file = open(os.path.join(output_dir, 'log_adaptation.txt'), 'w')
    for epoch in range(10):
        iter_test = iter(target_data_loader)

        netF.eval()

        netF.train()
        netB.train()
        for _, (inputs_test, target_label) in tqdm(enumerate(iter_test), leave=False):
            if inputs_test.size(0) == 1:
                continue
            inputs_test = inputs_test.cuda()
            pred = target_label.cuda()

            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)
            classifier_loss = network.CrossEntropyLabelSmooth(num_classes=params.class_num, epsilon=0)(outputs_test, pred)


            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

        netF.eval()
        netB.eval()
        acc, _ = network.cal_acc(target_data_loader_eval, netF, netB, netC)
        log_str = 'Task: {}, Adaptation training Iter:{}/{}; Accuracy = {:.2f}%'.format(params.mode, epoch + 1, params.epochs, acc * 100)
        out_file.write(log_str + '\n')
        out_file.flush()
        print(log_str + '\n')

    torch.save(netF.state_dict(), os.path.join(output_dir, "target_F.pt"))
    torch.save(netB.state_dict(), os.path.join(output_dir, "target_B.pt"))
    torch.save(netC.state_dict(), os.path.join(output_dir, "target_C.pt"))
    return netF, netB, netC
