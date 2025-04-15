
import torch
from utils import utils_image as util

import time
from Main.Models.network_scunet import SCUNet as net

class Denoiser:
    def __init__(self, model_path):
        self.n_channels = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = net(in_nc = self.n_channels, config = [4, 4, 4, 4, 4, 4, 4], dim = 64)

        model.load_state_dict(torch.load(model_path), strict = True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        self.model = model.to(self.device)

    def run_img(self, input_img):

        img_L = util.uint2tensor4(input_img)
        img_L = img_L.to(self.device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        # img_E = utils_model.test_mode(model, img_L, refield=64, min_size=512, mode=2)

        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        img_E = self.model(img_L)
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        elapsed_time = time.time() - start_time
        img_E = util.tensor2uint(img_E)

        return img_E