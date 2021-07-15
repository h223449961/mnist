import torchvision
import torch.utils.data as da
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])#Edit location
])
data_train = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)
data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)
#Data Loading
data_loder_train = da.DataLoader(dataset=data_train, batch_size=64, shuffle = True)
data_loder_test = da.DataLoader(dataset=data_test, batch_size = 64, shuffle = True)
images, lables = next(iter(data_loder_train))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
for i in range(64):
    print(lables[i], end=" ")
    i += 1
    if i%8 is 0:
        print(end='\n')
plt.imshow(img)
plt.show()
