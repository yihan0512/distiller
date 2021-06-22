import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import os
import sys
sys.path.append('../../')

from distiller.models import create_model

from torch.utils.tensorboard import SummaryWriter

import ipdb



if __name__ == "__main__":
    arch = sys.argv[1]
    writer_dir = sys.argv[2]
    dataset = 'cifar10'


    writer = SummaryWriter(os.path.join('runs', writer_dir))

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='../../../data.cifar10', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=16)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = create_model(False, dataset, arch, False)
    print(model)

    batch = next(iter(testloader))
    images, labels = batch
    images, labels = images.cuda(), labels.cuda()
    outputs = model(images.cuda())

    writer.add_graph(model, images)
    writer.close()

#     ipdb.set_trace()


    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)


    # for i, data in enumerate(testloader, 0):
    #     images, labels = data
    #     images = images.cuda()
    #     labels = labels.cuda()

    #     optimizer.zero_grad()

    #     outputs = model(images)

    #     loss = criterion(outputs, labels)

    #     loss.backward()
    #     optimizer.step()
