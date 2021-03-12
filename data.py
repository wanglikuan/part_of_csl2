import numpy as np
import torch, numbers
from random import Random
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import MNIST, CIFAR100
from torchvision import transforms

def data_catogerize(data:Dataset, seed=1234):
    data_dict = {}
    for idx in range(len(data)):
        _, target = data.__getitem__(idx)
        if target not in data_dict:
            data_dict[target] = []
        data_dict[target].append(idx)
    rng = Random()
    rng.seed(seed)
    for key in data_dict.keys():
        rng.shuffle(data_dict[key]) #shuffle将序列的所有元素随机排序
    return data_dict

def worker_labels(targets, num_workers, seed=1234):
    labels, temp = [], [target for target in targets]

    for i in range(num_workers):
        a = int(len(temp)/(num_workers - i))
        labels.append(temp[0:a])  #取targets前 0~a个
        temp = temp[a:]  #去掉targets前0~a个
    return labels #返回workers数量的labels

class NonIID_DataPartitioner(object):

    def __init__(self, train_data, test_data, sizes, classes=-1, seed=1234):
        self.train_data, self.test_data = train_data, test_data
        self.train_partitions, self.test_partitions = [], []
        train_data_dict, test_data_dict = data_catogerize(train_data), data_catogerize(test_data)
        rng = Random()
        rng.seed(seed)
        self.labels = labels = worker_labels(sorted(train_data_dict.keys()), len(sizes))

        for idx, ratio in enumerate(sizes): #相当于循环workers，idx为workers编号
            part_train_len, part_test_len, train_partition, test_partition = int(ratio * len(train_data)), int(ratio * len(test_data)), [], []
            for j, label in enumerate(labels[idx]):
                a, b = int(part_train_len / (len(labels[idx]) - j)), int(part_test_len / (len(labels[idx]) - j)) #a是
                train_partition.extend(train_data_dict[label][0:a])
                test_partition.extend(test_data_dict[label][0:b])
                train_data_dict[label], test_data_dict[label] = train_data_dict[label][a:], test_data_dict[label][b:]
                part_train_len, part_test_len = part_train_len - a, part_test_len - b
            rng.shuffle(train_partition)
            rng.shuffle(test_partition)
            self.train_partitions.append(train_partition)
            self.test_partitions.append(test_partition)
        
    def use(self, partition):
        return Subset(self.train_data, self.train_partitions[partition]), Subset(self.test_data, self.test_partitions[partition]), self.labels[partition]


def partition_dataset(train_dataset, test_dataset, workers, balanced, p=(1.0, 0.0)):
    """ Partitioning Data """
    workers_num = len(workers)
    if balanced:
        partition_sizes = [1.0 * p[0] / workers_num for _ in range(workers_num)]
    else: 
        partition_sizes = [float((i+1) * 2 / ((1+workers_num) * workers_num)) * p[0] for i in range(workers_num)]
    
    partition = NonIID_DataPartitioner(train_dataset, test_dataset, partition_sizes)
    return partition

def select_dataset(workers: list, rank: int, partition, batch_size: int):
    workers_num = len(workers)
    partition_dict = {workers[i]: i for i in range(workers_num)}
    train_partition, test_partition, label = partition.use(partition_dict[rank])
    return DataLoader(train_partition, batch_size=batch_size, shuffle=False), DataLoader(test_partition, batch_size=batch_size, shuffle=False), label

def get_split_mnist(bsz, num_task):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(root='data/mnist_data', train=True, download=True, transform=transform)
    testset = MNIST(root='data/mnist_data', train=False, download=True, transform=transform)

    workers = list(range(num_task))
    partitioner = partition_dataset(trainset, testset, workers, True)
    train_loader, test_loader, labels = {}, {}, {}
    for i in workers:
        train_loader[i], test_loader[i], labels[i] = select_dataset(workers, i, partitioner,bsz)
    return train_loader, test_loader, labels

def get_split_cifar(bsz, num_task):
    train_transform = transforms.Compose([ 
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])
    test_transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])
    trainset = CIFAR100('data/cifar_data', train=True, download=True, transform=train_transform)
    testset = CIFAR100('data/cifars_data', train=False, download=True, transform=test_transform)

    workers = list(range(num_task))
    partitioner = partition_dataset(trainset, testset, workers, True)
    train_loader, test_loader, labels = {}, {}, {}
    for i in workers:
        train_loader[i], test_loader[i], labels[i] = select_dataset(workers, i, partitioner,bsz)
    return train_loader, test_loader, labels

class PermutedMNISTDataLoader(MNIST):
    def __init__(self, source='data/mnist_data', train = True, shuffle_seed = None):
        super(PermutedMNISTDataLoader, self).__init__(source, train, download=True)
        
        self.train = train
        self.num_data = 0
        
        self.permuted_data = torch.stack(
            [img.type(dtype=torch.float32).view(-1)[shuffle_seed].view(1, 28, 28) / 255.0
                for img in self.data])
        self.num_data = self.permuted_data.shape[0]
            
    def __getitem__(self, index):
        input, label = self.permuted_data[index], self.targets[index] 
        return input, label

    def getNumData(self):
        return self.num_data

def get_permute_mnist(bsz, num_task, seeds=None):
    train_loader, test_loader, labels = {}, {}, {}
    
    train_data_num, test_data_num = 0, 0
    
    for i in range(num_task):
        shuffle_seed = np.arange(28*28)
        if i > 0:
            if seeds is not None and len(seeds) == num_task:
                np.random.seed(seeds[i])
            np.random.shuffle(shuffle_seed)
        
        train_PMNIST_DataLoader = PermutedMNISTDataLoader(train=True, shuffle_seed=shuffle_seed)
        test_PMNIST_DataLoader = PermutedMNISTDataLoader(train=False, shuffle_seed=shuffle_seed)
        
        train_data_num += train_PMNIST_DataLoader.getNumData()
        test_data_num += test_PMNIST_DataLoader.getNumData()
        
        train_loader[i] = DataLoader(train_PMNIST_DataLoader, batch_size=bsz, shuffle=False)
        test_loader[i] = DataLoader(test_PMNIST_DataLoader, batch_size=bsz, shuffle=False)
        # labels[i] = train_PMNIST_DataLoader.targets
        labels[i] = list(range(10))
    
    # return train_loader, test_loader, int(train_data_num/num_task), int(test_data_num/num_task)
    return train_loader, test_loader, labels


class Rotation(object):
    def __init__(self, degree, resample=False, expand=False, center=None):
        self.degree = degree

        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img):
        
        def rotate(img, angle, resample=False, expand=False, center=None):
            """Rotate the image by angle and then (optionally) translate it by (n_columns, n_rows)
            Args:
            img (PIL Image): PIL Image to be rotated.
            angle ({float, int}): In degrees degrees counter clockwise order.
            resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
            expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
            center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
            """
                
            return img.rotate(angle, resample, expand, center)

        angle = self.degree

        return rotate(img, angle, self.resample, self.expand, self.center)

def get_rotate_mnist(bsz, num_task):
    train_loader, test_loader, labels = {}, {}, {}
    for i in range(num_task):
        transform = transforms.Compose([Rotation(180/num_task*i), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        trainset = MNIST(root='data/mnist_data', train=True, download=True, transform=transform)
        testset = MNIST(root='data/mnist_data', train=False, download=True, transform=transform)
        train_loader[i] = DataLoader(trainset, batch_size=bsz, shuffle=False)
        test_loader[i] = DataLoader(testset, batch_size=bsz, shuffle=False)
        labels[i] = list(range(10))
    return train_loader, test_loader, labels