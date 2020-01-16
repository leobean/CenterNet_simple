import torch.nn as nn
import torch

class block(torch.nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.conv31 = torch.nn.Conv2d(64,64,kernel_size=(3,3),padding=0, stride=1, dilation=1, groups=1, bias=False)
    def forward(self, input):
        out = self.conv31(input)
        return out



class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (0, -1)
            hor_pad_or_crop = (-1, 0)

            ver_conv_padding = (1, 0)
            hor_conv_padding = (0, 1)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride, padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride, padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

            self.corner_conv = four_corner_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=1)
            self.corner_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            #vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            #horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            corner_outputs = self.corner_conv(input)
            corner_outputs = self.corner_bn(corner_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs + corner_outputs

import torch.nn.functional as F
import torch

def four_corner(out_channels, in_channels, kernel_size):
    '''
    :param four_weight:  for 3*3 kernel, the four corner point's weight, which need calculate grad
    :param five_weight:  for 3*3 kernel, other five point("十" shape)'s weight, which is zero, need not calculate grad
    :return: a 3*3 kernel
    w 0 w
    0 0 0
    w 0 w
    '''
    kernel_size1, kernel_size2 = kernel_size, kernel_size
    if isinstance(kernel_size, tuple):
        kernel_size1 = kernel_size[0]
        kernel_size2 = kernel_size[1]
    new_kernel = torch.randn(out_channels, in_channels, kernel_size1, kernel_size2, requires_grad=True)
    with torch.no_grad():
        new_kernel[:, :, 0, 1] = 0.
        new_kernel[:, :, 1, 0] = 0.
        new_kernel[:, :, 1, 1] = 0.
        new_kernel[:, :, 1, 2] = 0.
        new_kernel[:, :, 2, 1] = 0.

    return new_kernel

class four_corner_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(four_corner_conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

    def forward(self, x):

        kernel = four_corner(self.out_channels, self.in_channels, self.kernel_size)
        kernel = kernel.float()   # [channel_out, channel_in, kernel, kernel]
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out



# if __name__ == '__main__':
#     mat = torch.ones((1,64,256,256))
#     print(mat.size())
#     mod = ACBlock(64,64,3,padding=1,stride=2)
#     #mod = block()
#     print(mod(mat).size())


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Solution:
    def numIslands(self, grid) -> int:
        if grid == []:
            return 0
        if grid[0] == []:
            return 0
        import queue
        height = len(grid)
        width = len(grid[0])
        visited = [[0 for i in range(width)] for j in range(height)]
        q = queue.Queue()
        num_lands = 0
        direction = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        # q.put(grid[0][0])
        for i in range(height):
            for j in range(width):
                if visited[i][j] == 1:
                    continue
                if grid[i][j] == '0':
                    visited[i][j] = 1
                    continue
                else:
                    visited[i][j] = 1
                    pt = Point(i, j)
                    q.put(pt)
                    while (not q.empty()):
                        this_point = q.get()
                        for direct in direction:
                            next_point = [this_point.x + direct[0], this_point.y + direct[1]]
                            if (next_point[0] < height and next_point[0] >= 0 and next_point[1] < width and next_point[1] >= 0):
                                if (visited[next_point[0]][next_point[1]] == 0 and grid[next_point[0]][next_point[1]] == '1'):
                                    visited[next_point[0]][next_point[1]] = 1
                                    q.put(Point(next_point[0], next_point[1]))
                    num_lands += 1
        return num_lands


if __name__ == '__main__':
    s = Solution()
    a = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
    ans = s.numIslands(a)
    print(ans)
