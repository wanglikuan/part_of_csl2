from torchvision import datasets, models, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset, random_split, RandomSampler, BatchSampler
from torchsummary.torchsummary import summary

from SplitModelsForMnist import ResNet18, ResNet34, ResNet152, VGG, LeNet, CNN, AlexNet, ResNet101, ResNet50

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGG')
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--iid', type=bool, default=False)
parser.add_argument('--bsz', type=int, default=128)

args = parser.parse_args()

def model_retrieval():
    return {
        'ResNet18': ResNet18(),
        'ResNet34': ResNet34(),
        'ResNet50': ResNet50(),
        'ResNet101': ResNet101(),
        'ResNet152': ResNet152(),
        'VGG': VGG(),
        'LeNet': LeNet(),
        'CNN': CNN(),
        'AlexNet': AlexNet()
    }[args.model]

def load_MNIST_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    return trainset, testset

def layer_size(shape):
    ans = 1
    for num in shape:
        ans *= num
    return ans

def generate_cut_layer(cut_layer_idx, model):
    input_datasize = (1, 28, 28)  # Mnist dataset 
    cut_layer_idx = min(cut_layer_idx, len(summary(model, input_datasize, depth=1, verbose=0).summary_list))
    print(summary(model, input_datasize, depth=1, verbose=0))

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
    
    return real_cut_idx, first_size, second_size, first_op, second_op, smashed_data_size, cut_layer_gradient_size

def client_side(dataset, iterator, model, optimizer, criterion, device):
    start = time.time()
    
    model.train()
    try:
        data, target = next(iterator)
    except:
        iterator = iter(dataset)
        data, target = next(iterator)
    # data, target = Variable(data), Variable(target)
    data, target = Variable(data).cuda(device), Variable(target).cuda(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    # print(loss)
    loss.backward()
    # print(loss)
    optimizer.step()

    end = time.time()

    return loss.item(), end - start

def test_model(model, test_data, device, criterion=nn.CrossEntropyLoss()):
    correct, total, test_loss = 0, 0, 0.0
    # model.eval()
    with torch.no_grad():
        i = 1
        for data, target in test_data:
            i = i + 1
            if i > 10:
                break
            # data, target = Variable(data), Variable(target)
            data, target = Variable(data).cuda(device), Variable(target).cuda(device)
            output = model(data)
            test_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= 10
    acc = correct / total
    # acc = format(correct / total, '.4%')
    return test_loss, acc

# device settings 
cpu, gpu = torch.device('cpu'), torch.device('cuda')
device = gpu
device_size = 1000 #MB
bandwidth = 100 #Mbps

# Initialization 
cut_layer_idx = args.split
real_cut_idx, first_size, second_size, first_op, second_op, smashed_data_size, cut_layer_gradient_size = generate_cut_layer(cut_layer_idx, model_retrieval())

workers_num = 8
iteration = 2000
batchsize = args.bsz
print("Current Model: {}".format(args.model))
print("Current Cut Layer: {}".format(cut_layer_idx))
print("Model size on device: {:.6f} MB".format(first_size * 32 / 8 / 1000 / 1000))
print("Model size on server: {:.6f} MB".format(second_size * 32 / 8 / 1000 / 1000))
print("Number of operations on device: {}".format(first_op))
print("Number of operations on server: {}".format(second_op))
print("Number of smashed data: {}".format(smashed_data_size))
print("Size of transmission part of a gradient: {}".format(cut_layer_gradient_size))
print("Bandwidth: {} Mbps".format(bandwidth))
print("Batch size: {}".format(batchsize))
learning_rate = 0.1

trainset, testset = load_MNIST_dataset()
if args.iid:
    worker_alloc = [int(len(trainset)/workers_num) for _ in range(workers_num)]
    worker_trainsets = random_split(trainset, worker_alloc)
    worker_trainsets = [DataLoader(worker_trainsets[i], batch_size=batchsize) for i in range(workers_num)]
else:
    worker_alloc = [int((i+1) * len(trainset)/(25*workers_num)) for i in range(workers_num)]
    worker_trainsets = random_split(trainset, worker_alloc + [len(trainset) - sum(worker_alloc)])
    worker_trainsets = [DataLoader(worker_trainsets[i], batch_size=batchsize) for i in range(workers_num)]
worker_iterators = [iter(worker_trainsets[i]) for i in range(workers_num)]
# worker_models = [ResNet18() for i in range(workers_num)]
worker_models = [model_retrieval().cuda(device) for i in range(workers_num)]
worker_optimizers = [optim.SGD(model.parameters(), lr=learning_rate) for model in worker_models]
worker_criteria = [torch.nn.CrossEntropyLoss() for i in range(workers_num)]

tot_time_cost, tot_communication_cost, tot_calculation = 0, 0, 0
test_dataset = DataLoader(testset, batch_size=100)
test_iterator = iter(test_dataset)
iteration_list, time_cost_list, communication_cost_list, loss_list = [], [], [], []

if args.iid:
    f = open('./result/IID _{}_cut_layer_{}_result.csv'.format(args.model, cut_layer_idx), 'w')
else:
    f = open('./result/NonIID_{}_cut_layer_{}_result.csv'.format(args.model, cut_layer_idx), 'w')

for t in range(iteration):
    time_cost, communication_cost, loss, calculation = 0, 0, 0, 0
    for i in range(workers_num):
        client_loss, client_time_cost = client_side(worker_trainsets[i], worker_iterators[i], worker_models[i], worker_optimizers[i], worker_criteria[i], device)
        time_cost = max(time_cost, client_time_cost)
        loss += client_loss * worker_alloc[i] / sum(worker_alloc)
    communication_cost = 2 * smashed_data_size * batchsize * workers_num * 32 
    calculation = 2 * first_op * batchsize * workers_num
    time_cost = time_cost + communication_cost/(bandwidth*1000*1000)

    # Here we integral the public part 
    temp_param = [torch.zeros_like(param.data) for param in worker_models[0].parameters()]
    for i in range(workers_num):
        for idx, param in enumerate(worker_models[i].parameters()):
            if idx < real_cut_idx:
                continue
            temp_param[idx] = temp_param[idx] + param.data * worker_alloc[i] / sum(worker_alloc)

    for i in range(workers_num):
        for idx, param in enumerate(worker_models[i].parameters()):
            if idx < real_cut_idx:
                continue
            param.data = temp_param[idx]

    tot_time_cost, tot_communication_cost, tot_calculation = tot_time_cost + time_cost, tot_communication_cost + communication_cost, tot_calculation + calculation
    iteration_list.append(t)
    time_cost_list.append(tot_time_cost)
    communication_cost_list.append(tot_communication_cost)
    loss_list.append(loss)

    test_loss, test_accuracy = 0.0, 0.0
    for i in range(workers_num):
        worker_test_loss, worker_test_accuracy = test_model(worker_models[i], test_dataset, device)
        test_loss += worker_test_loss * worker_alloc[i] / sum(worker_alloc)
        test_accuracy += worker_test_accuracy * worker_alloc[i] / sum(worker_alloc)
        print("Worker {}\tLoss: {:.4f}\tAccuracy: {:.4%}".format(i, worker_test_loss, worker_test_accuracy))

    f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(t, time_cost, tot_time_cost, 
            communication_cost, tot_communication_cost, calculation, tot_calculation, loss, test_loss, test_accuracy))
    print('Iteration: {}\tTraining Loss: {:.4f}\tTest Loss: {:.4f}\tTest Accuracy: {:.4%}'.format(t, loss, test_loss, test_accuracy))

    if t % 20 == 0:
        for i in range(workers_num):
            for param_group in worker_optimizers[i].param_groups:
                param_group['lr'] *= 0.95

f.close()

print()
print()
print("========================================")
print()
print()