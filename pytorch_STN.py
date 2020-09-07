import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def resize_image(image, size):
    iw, ih = image.width, image.height
    w, h = size

    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (255, 255, 255))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        return self.stn(x)


img = Image.open('/Users/sunjiancheng/Desktop/7月市场数据集/红富士苹果/{红富士苹果_3}_0104.jpg').convert('RGB')
img = resize_image(img, (224, 224))
loader = transforms.Compose([transforms.ToTensor()])
img = loader(img).unsqueeze(0).to(torch.float)
net = Net()
out = net(img)
print(out)
