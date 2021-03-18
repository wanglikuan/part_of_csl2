import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import time, random, math
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset, random_split, RandomSampler, BatchSampler
from torchsummary.torchsummary import summary

from data import get_permute_mnist, get_split_mnist, get_rotate_mnist, get_split_cifar
from models import cifar, mnist
from utils import EWC, splitEWC, ewc_train, normal_train, ours_first_train, our_train, test_model
 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='standard')
parser.add_argument('--model', type=str, default='AlexNet')
parser.add_argument('--dataset', type=str, default='permuted')
parser.add_argument('--split', type=int, default=3)
parser.add_argument('--bsz', type=int, default=128)
parser.add_argument('--num-task', type=int, default=5)
parser.add_argument('--first-lr', type=float, default=1e-2)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--lam', type=float, default=15)
parser.add_argument('--num-gpu', type=int, default=1)

parser.add_argument('--threshold', type=float, default=0.4) #
parser.add_argument('--FirstThreshold', type=float, default=0.4) #

args = parser.parse_args()

def model_retrieval():
    if 'cifar' in args.dataset:
        return {
            'ResNet18': cifar.ResNet18(100),
            'VGG': cifar.vgg16(100),
            #'LeNet': cifar.LeNet(100),
            #'CNN': cifar.CNN(100),
            'AlexNet': cifar.AlexNet(100)
        }[args.model]
    return {
        'ResNet18': mnist.ResNet18(),
        'VGG': mnist.VGG(),
        'LeNet': mnist.LeNet(),
        'CNN': mnist.CNN(),
        'AlexNet': mnist.AlexNet()
    }[args.model]

def layer_size(shape):
    ans = 1
    for num in shape:
        ans *= num
    return ans

def generate_cut_layer(cut_layer_idx, model):
    #input_datasize = (1, 28, 28)  # Mnist dataset
    input_datasize = (3, 32, 32)  # cifar 
    #cut_layer_idx = min(cut_layer_idx, len(summary(model, input_datasize, depth=1, verbose=0).summary_list))
    cut_layer_idx = min(cut_layer_idx, len(summary(model, input_datasize, depth=1, verbose=0).summary_list))
    # print(summary(model, input_datasize, depth=1, verbose=0))

    first_param_num, second_param_num, first_size, second_size, first_op, second_op = 0, 0, 0, 0, 0, 0
    for i, summary_list in enumerate(summary(model, input_datasize, depth=0, verbose=0).summary_list):
        if i < cut_layer_idx:
            first_param_num += summary_list.num_params
            first_size = first_size + layer_size(summary_list.output_size[1:]) + summary_list.num_params
            first_op = first_op + summary_list.macs
        else:
            second_param_num += summary_list.num_params
            second_size = second_size + layer_size(summary_list.output_size[1:]) + summary_list.num_params
            second_op = second_op + summary_list.macs

    if cut_layer_idx == 0:
        smashed_data_size = layer_size(summary(model, input_datasize, depth=0, verbose=0).summary_list[cut_layer_idx].input_size[1:])
        first_size, second_size = first_size + smashed_data_size, second_size + smashed_data_size
        cut_layer_gradient_size = summary(model, input_datasize, depth=0, verbose=0).summary_list[cut_layer_idx].num_params
    elif cut_layer_idx >= len(summary(model, input_datasize, depth=0, verbose=0).summary_list):
        smashed_data_size = layer_size(summary(model, input_datasize, depth=0, verbose=0).summary_list[-1].output_size[1:])
        first_size = first_size + layer_size(summary(model, input_datasize, depth=0, verbose=0).summary_list[0].input_size[1:])
        second_size = second_size + smashed_data_size
        cut_layer_gradient_size = 1
    else:
        smashed_data_size = layer_size(summary(model, input_datasize, depth=0, verbose=0).summary_list[cut_layer_idx].input_size[1:])
        first_size = first_size + layer_size(summary(model, input_datasize, depth=0, verbose=0).summary_list[0].input_size[1:])
        second_size = second_size + smashed_data_size
        cut_layer_gradient_size = summary(model, input_datasize, depth=0, verbose=0).summary_list[cut_layer_idx].num_params
    
    tot_param_num, real_cut_idx = first_param_num, 0
    for i, param in enumerate(model.parameters()):
        # print(tot_param_num, len(param.reshape(-1)))
        tot_param_num -= len(param.reshape(-1))
        if tot_param_num == 0:
            real_cut_idx = i+1
            break
    
    # return real_cut_idx, first_size, second_size, first_op, second_op, smashed_data_size, cut_layer_gradient_size
    return real_cut_idx

def models_copy(target_model, src_model, cut_idx=0):
    temp_param = [torch.zeros_like(param.data) if idx < cut_idx else torch.zeros_like(param.data) + param.data for idx, param in enumerate(src_model.parameters())]
    #如果小于cut_idx:归零；如果大于cut_idx:copy src_model的param
    for idx, param in enumerate(target_model.parameters()):
        if idx < cut_idx:  #跳过model的前几层，复制model的后几层，因为model的后几层（cut_idx ~ end）是server，前几层(0 ~ cut_idx-1)是client
            continue
        param.data = torch.zeros_like(param.data) + temp_param[idx]
    # target.load_state_dict()
    return target_model

def print_model(model:nn.Module):
    # for idx, param in enumerate(model.parameters()):
    #     print(idx, param.data)
    # print("===================")
    for idx, (key, value) in enumerate(model.state_dict().items()):
        print(idx, key, value)

def standard_process(train_loader, test_loader, labels, class_incremental, result_file='./standard.txt'):
    gpu = torch.device('cuda:0')
    new_model = model_retrieval().cuda(gpu)
    # print_model(new_model)
    models, cur_label = [copy.deepcopy(new_model) for _ in range(args.num_task)], []
    # time.sleep(10)
    temp_model = copy.deepcopy(new_model)
    cut_idx = generate_cut_layer(args.split, temp_model)
    optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(args.num_task)]

    for task in range(args.num_task):
        print('Training Task {}... Labels: {}'.format(task, labels[task]))
        model, optimizer = models[task] if task == 0 else models_copy(models[task], models[task-1], cut_idx), optimizers[task]
        # print_model(model)
        if task == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.first_lr
        for iteration in range(args.iterations):
            cur_label = cur_label + labels[task] if class_incremental else labels[task]
            loss = normal_train(model, cur_label, optimizer, train_loader[task], gpu)
            # print_model(model)
            print('Iteration: {}\tLoss:{}'.format(iteration, loss))
            for sub_task in range(task + 1):
                temp_model = copy.deepcopy(models[sub_task])
                temp_model = models_copy(temp_model, model, cut_idx)
                for i in range(task + 1):
                    cur_label = cur_label if class_incremental else labels[i]
                    acc = test_model(temp_model, cur_label, test_loader[i], gpu)
                    print('Device Task: {}\tTest Task: {}\tAccuracy: {}'.format(sub_task, i, acc))
                    with open(result_file, 'a') as f:
                        # Current server parameter (task) + device parameter (task) --> training a given task
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(task, iteration, loss, sub_task, i, acc))
            if iteration % 20 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95

def ewc_process_without_split(train_loader, test_loader, labels, class_incremental, online=False, result_file='./ewc_without_split.txt'):
    gpu = torch.device('cuda:0')
    model = model_retrieval().cuda(gpu)
    cur_label = []
    optimizer = optim.SGD(params=model.parameters(), lr=args.first_lr)

    ewcs = []
    for task in range(args.num_task):
        print('Training Task {}... Labels: {}'.format(task, labels[task]))
        if task == 0:
            cur_label = cur_label + labels[task] if class_incremental else labels[task]
            for iteration in range(args.iterations):
                loss = normal_train(model, cur_label, optimizer, train_loader[task], gpu)
                print('Iteration: {}\tLoss:{}'.format(iteration, loss))
                acc = test_model(model, cur_label, test_loader[task], gpu)
                print('Test Task: {}\tAccuracy: {}'.format(task, acc))
                with open(result_file, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(task, iteration, loss, task, task, acc))
                if iteration % 20 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.95
        else:
            cur_label = cur_label + labels[task] if class_incremental else labels[task]
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            for iteration in range(args.iterations):
                if online:
                    loss = ewc_train(model, cur_label, optimizer, train_loader[task], ewcs[-1:], args.lam, gpu)
                else:
                    loss = ewc_train(model, cur_label, optimizer, train_loader[task], ewcs, args.lam, gpu)
                print('Iteration: {}\tLoss:{}'.format(iteration, loss))
                for sub_task in range(task + 1):
                    cur_label = cur_label if class_incremental else labels[sub_task]
                    acc = test_model(model, cur_label, test_loader[sub_task], gpu)
                    print('Test Task: {}\tAccuracy: {}'.format(sub_task, acc))
                    with open(result_file, 'a') as f:
                        # Current server parameter (task) + device parameter (task) --> training a given task
                        f.write('{}\t{}\t{}\t{}\t{}\n'.format(task, iteration, loss, sub_task, acc))
                if iteration % 20 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.95
        ewcs.append(EWC(model, train_loader[task], gpu))

def ewc_process(train_loader, test_loader, labels, class_incremental, online=False, result_file='./ewc.txt'):
    gpu = torch.device('cuda:0')
    new_model = model_retrieval().cuda(gpu)
    models, cur_label = [copy.deepcopy(new_model) for _ in range(args.num_task)], []
    temp_model = copy.deepcopy(new_model)
    cut_idx = generate_cut_layer(args.split, temp_model)
    optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(args.num_task)]

    ewcs = []
    for task in range(args.num_task):
        print('Training Task {}... Labels: {}'.format(task, labels[task]))
        model, optimizer = models[task] if task == 0 else models_copy(models[task], models[task-1], cut_idx), optimizers[task] #model为tmp变量循环用，model：copy了上一次task中model的param（与cut_idx有关）
        if task == 0:
            cur_label = cur_label + labels[task] if class_incremental else labels[task]
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.first_lr
            for iteration in range(args.iterations):
                loss = normal_train(model, cur_label, optimizer, train_loader[task], gpu)
                print('Iteration: {}\tLoss:{}'.format(iteration, loss))
                acc = test_model(model, cur_label, test_loader[task], gpu)
                print('Device Task: {}\tTest Task: {}\tAccuracy: {}'.format(task, task, acc))
                with open(result_file, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(task, iteration, loss, task, task, acc))
                if iteration % 20 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.95
        else:
            cur_label = cur_label + labels[task] if class_incremental else labels[task]
            for iteration in range(args.iterations):
                if online:
                    loss = ewc_train(model, cur_label, optimizer, train_loader[task], ewcs[-1:], args.lam, gpu) #与normal_train相比多了ewcs[]与args.lam
                else:
                    loss = ewc_train(model, cur_label, optimizer, train_loader[task], ewcs, args.lam, gpu) #与online区别是ewcs[-1:]为ewcs只取最后一个元素构成的列表
                print('Iteration: {}\tLoss:{}'.format(iteration, loss))
                for sub_task in range(task + 1): #循环不同model
                    temp_model = copy.deepcopy(models[sub_task])
                    temp_model = models_copy(temp_model, model, cut_idx) #temp_model 用的是当前model后半部分，models[]前半部分
                    for i in range(task + 1): #循环不同task
                        cur_label = cur_label if class_incremental else labels[i]
                        acc = test_model(temp_model, cur_label, test_loader[i], gpu)
                        print('Device Task: {}\tTest Task: {}\tAccuracy: {}'.format(sub_task, i, acc))
                        with open(result_file, 'a') as f:
                            # Current server parameter (task) + device parameter (task) --> training a given task
                            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(task, iteration, loss, sub_task, i, acc))
                if iteration % 20 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.95
        ewcs.append(splitEWC(model, train_loader[task], cut_idx, gpu))

def our_process(train_loader, test_loader, labels, class_incremental, online=False, result_file='./our_process.txt'):
    gpu = torch.device('cuda:0')
    new_model = model_retrieval().cuda(gpu)
    models, cur_label = [copy.deepcopy(new_model) for _ in range(args.num_task)], []
    temp_model = copy.deepcopy(new_model)
    cut_idx = generate_cut_layer(args.split, temp_model)
    optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(args.num_task)]

    ewcs = []
    if_freeze, freeze_stat = 0, 0
    loss = 10
    for task in range(args.num_task):
        print('Training Task {}... Labels: {}'.format(task, labels[task]))
        model, optimizer = models[task] if task == 0 else models_copy(models[task], models[task-1], cut_idx), optimizers[task] #model为tmp变量循环用，model：copy了上一次task中model的param（与cut_idx有关）
        if task == 0:
            cur_label = cur_label + labels[task] if class_incremental else labels[task]
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.first_lr
            for iteration in range(args.iterations):
                # loss = ours_first_train(model, cur_label, optimizer, train_loader[task], gpu, cut_idx, freeze_stat, loss)
                # if loss > args.FirstThreshold:
                #     freeze_stat = 0
                # else:
                #     freeze_stat = 1
                # print('Iteration: {}\tLoss:{}\tfreeze_stat:{}'.format(iteration, loss, freeze_stat))
                loss = normal_train(model, cur_label, optimizer, train_loader[task], gpu)
                print('Iteration: {}\tLoss:{}'.format(iteration, loss))
                acc = test_model(model, cur_label, test_loader[task], gpu)
                print('Device Task: {}\tTest Task: {}\tAccuracy: {}'.format(task, task, acc))
                with open(result_file, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(task, iteration, loss, task, task, acc))
                if iteration % 20 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.95
        else:
            cur_label = cur_label + labels[task] if class_incremental else labels[task]
            for iteration in range(args.iterations):
                if online:
                    #loss = our_train(model, cur_label, optimizer, train_loader[task], ewcs[-1:], args.lam, gpu, cut_idx, args.threshold) #与normal_train相比多了ewcs[]与args.lam
                    loss = our_train(model, cur_label, optimizer, train_loader[task], ewcs[-1:], args.lam, gpu, cut_idx, if_freeze) #与normal_train相比多了ewcs[]与args.lam
                else:
                    loss = our_train(model, cur_label, optimizer, train_loader[task], ewcs, args.lam, gpu, cut_idx, args.threshold) #与online区别是ewcs[-1:]为ewcs只取最后一个元素构成的列表
                    # loss = our_train(model, cur_label, optimizer, train_loader[task], ewcs, args.lam, gpu, cut_idx, if_freeze) #与online区别是ewcs[-1:]为ewcs只取最后一个元素构成的列表
                #判断loss，loss若小于阈值，令变量if_freeze=1,传入下次our_train
                #our_train相比于ewc_train多两个参数：if_freeze和cut_idx
                if loss < args.threshold:
                    if_freeze = 1
                else:
                    if_freeze = 0
                #print('Iteration: {}\tLoss:{}'.format(iteration, loss))
                print('Iteration: {}\tLoss:{}\tif freeze:{}'.format(iteration, loss, if_freeze))
                for sub_task in range(task + 1): #循环不同model
                    temp_model = copy.deepcopy(models[sub_task])
                    temp_model = models_copy(temp_model, model, cut_idx) #temp_model 用的是当前model后半部分，models[]前半部分
                    for i in range(task + 1): #循环不同task
                        cur_label = cur_label if class_incremental else labels[i]
                        acc = test_model(temp_model, cur_label, test_loader[i], gpu)
                        print('Device Task: {}\tTest Task: {}\tAccuracy: {}'.format(sub_task, i, acc))
                        with open(result_file, 'a') as f:
                            # Current server parameter (task) + device parameter (task) --> training a given task
                            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(task, iteration, loss, sub_task, i, acc))
                if iteration % 20 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.95
        ewcs.append(splitEWC(model, train_loader[task], cut_idx, gpu))


if __name__ == '__main__':
    if args.dataset == 'permuted':
        # train_loader, test_loader, labels = get_permute_mnist(args.bsz, args.num_task, [1829, 241, 43, 12649, 4443])
        train_loader, test_loader, labels = get_permute_mnist(args.bsz, args.num_task)
    elif args.dataset == 'rotated':
        train_loader, test_loader, labels = get_rotate_mnist(args.bsz, args.num_task)
    elif 'mnist' in args.dataset:
        train_loader, test_loader, labels = get_split_mnist(args.bsz, args.num_task)
    elif 'cifar' in args.dataset: 
        train_loader, test_loader, labels = get_split_cifar(args.bsz, args.num_task)

    if args.method == 'ewc':
        ewc_process(train_loader, test_loader, labels, 'class' in args.dataset, result_file='./result/2095_t10_{}_{}_{}.txt'.format(args.method, args.dataset, args.split))
    elif args.method == 'online':
        ewc_process(train_loader, test_loader, labels, 'class' in args.dataset, True, result_file='./result/2095_t10_{}_{}_{}.txt'.format(args.method, args.dataset, args.split))
    elif args.method == 'split_free':
        ewc_process_without_split(train_loader, test_loader, labels, 'class' in args.dataset, False, result_file='./result/2095_t10_{}_{}.txt'.format(args.method, args.dataset))
    elif args.method == 'ours':
        our_process(train_loader, test_loader, labels, 'class' in args.dataset, result_file='./result/2095_t10_{}_{}_{}_{}_{}.txt'.format(args.method, args.dataset, args.split, args.FirstThreshold, args.threshold))
    else:
        standard_process(train_loader, test_loader, labels, 'class' in args.dataset, result_file='./result/2095_t10_i100_lr0.05_{}_{}_{}.txt'.format(args.method, args.dataset, args.split))
