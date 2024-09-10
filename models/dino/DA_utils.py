import torch
from torch import nn
import torch.nn.functional as F

def decompose_features(srcs,masks,poss):

    B, _, _, _ = srcs[0].shape
    #source
    srcs_source = []
    masks_source = []
    poss_source = []

    #target
    # srcs_target = []
    # masks_target = []
    # poss_target = []


    for i in range(len(srcs)):
        # source
        srcs_source.append(srcs[i][:B//2,:,:,:])
        masks_source.append(masks[i][:B//2,:,:])
        poss_source.append(poss[i][:B//2,:,:,:])

        # srcs_target.append(srcs[i][B // 2:, :, :, :])
        # masks_target.append(masks[i][B // 2:, :, :])
        # poss_target.append(poss[i][B // 2:, :, :, :])



    return srcs_source,masks_source,poss_source,srcs,masks,poss

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class DA_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def get_valid_feature(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_h = valid_H
    valid_w = valid_W
    valid_ratio = torch.stack([valid_h, valid_w], -1)
    return valid_ratio



class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

def mask_to_box_single_img(boxes, labels, size,image_size,num_classes= None):
    mask = torch.zeros((num_classes,image_size[0], image_size[1])).cuda()

    img_w, img_h = size[1], size[0]
    img_w, img_h = torch.tensor([img_w]), torch.tensor([img_h])
    scale_fct = torch.stack([img_w, img_h, img_w, img_h]).view(1, 4).cuda()


    if len(boxes.size()) != 3:
        boxes = boxes.unsqueeze(0)

    boxes = boxes * scale_fct
    boxes = boxes[0]
    for box, label in zip(boxes, labels):
        x, y, w, h = box
        xmin, xmax = x - w / 2, x + w / 2
        ymin, ymax = y - h / 2, y + h / 2
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
        mask[label,ymin:ymax, xmin:xmax] = 1

    #debug save_mask
   #  np_mask = masks.cpu().numpy()
   #  save_path = '/data/jianhonghan/code/第三篇域泛化/code/创新实验代码/DINO-main-DA域泛化_多视角_改版/test.jpg'
   #  print(save_path)
   #  print('********')
   #  print(np_mask.shape)
   #  cv2.imshow('a',np_mask * 255)
   # # cv2.waitKey()
   #  cv2.imwrite(save_path,np_mask * 255)
    mask_label = torch.sum(mask.flatten(1),dim=1) != 0
    return mask,mask_label #label用于判断输入图像存在的类别