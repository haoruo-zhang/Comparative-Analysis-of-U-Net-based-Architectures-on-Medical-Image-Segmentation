import torch
import torch.nn as nn


def conv_block(dim_in,
               dim_out,
               kernel_size=3,
               stride=1,
               padding=1,
               bias=True):
    """
    Functions for adding the convolution layers.

    Args:
    - dim_in (int): Number of input channels.
    - dim_out (int): Number of output channels.
    - kernel_size (int, optional): Size of the convolutional kernel (default is `3`).
    - stride (int, optional): Stride of the convolution (default is `1`).
    - padding (int, optional): Padding added to the input (default is `1`).
    - bias (bool, optional): The bias. Default is True.

    Returns:
    - (nn.Sequential): the convolutional block
    """
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(dim_out),
        nn.LeakyReLU(0.1),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(dim_out),
        nn.LeakyReLU(0.1)
    )


def upsample(ch_coarse, ch_fine,
             kernel_size=4,
             stride=2,
             padding=1):
    """
    Functions for adding the upsampling layers.

    Args:
    - ch_coarse (int): Number of input channels
    - ch_fine (int): Number of output channels
    - kernel_size (int, optional): Size of the convolutional kernel (default is `4`).
    - stride (int, optional): Stride of the convolution (default is `2`).
    - padding (int, optional): Padding added to the input (default is `1`).

    Returns:
    - (nn.Sequential): the upsampling module
    """
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, kernel_size, stride, padding, bias=False),
        nn.ReLU())


class UNet(nn.Module):
    """
    The U-Net model.
    """

    def __init__(self, n_out, n_in=1):
        """
        Args:
        - n_in (int): Number of channels the image has.
        - n_out (int): Number of classes.
        """
        super(UNet, self).__init__()
        # Downgrade stages
        self.conv1 = conv_block(n_in, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)
        # Upgrade stages
        self.conv4m = conv_block(1024, 512)
        self.conv3m = conv_block(512, 256)
        self.conv2m = conv_block(256, 128)
        self.conv1m = conv_block(128, 64)
        # Maxpool
        self.max_pool = nn.MaxPool2d(2)
        # Upsample layers
        self.upsample54 = upsample(1024, 512)
        self.upsample43 = upsample(512, 256)
        self.upsample32 = upsample(256, 128)
        self.upsample21 = upsample(128, 64)
        self.convfinal = nn.Conv2d(64, n_out, kernel_size=1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass.

        Args:
        - x (tensor): The data batch.

        Returns:
        - output (tensor): Result of forward pass.
        """
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))

        conv5m_out_ = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
        conv4m_out = self.conv4m(conv5m_out_)
        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)
        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)

        output = self.convfinal(conv1m_out)
        return output


if __name__ == "__main__":
    tensor = torch.randn(1, 1, 224, 224)
    model = UNet(2)
    output = model(tensor)
    print(output.shape)
    print("finish")
