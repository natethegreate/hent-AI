import sys
import os.path
import cv2
import numpy as np
import torch
import architecture
import math

# ESRGAN class allows abstraction of warmup and inference.
class esrgan():
    # hw = cpu, or cuda
    def __init__(self, model_path=None, hw='cpu'):
        assert model_path
        if hw=='cpu':
            self.device = torch.device('cpu')  
        if hw=='cuda':
            self.device = torch.device('cuda')
        self.model = architecture.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                                mode='CNA', res_scale=1, upsample_mode='upconv')
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model = self.model.to(self.device)
        print('Model warmup complete')

    # Function to run esrgan on single image, and single output.
    def run_esrgan(self, test_img_folder=None, out_filename=None, mosaic_res=1):
        assert out_filename
        assert test_img_folder

        img = cv2.imdecode(np.fromfile(test_img_folder, np.uint8), cv2.IMREAD_UNCHANGED)
        # from alex: codes to shrink image if memory is an issue
        # GPUmem = torch.cuda.get_device_properties(0).total_memory
        # Sx = int(1.2*img.shape[0]/mosaic_res)
        # Sy = int(1.2*img.shape[1]/mosaic_res)
        # maxres = math.sqrt((Sx*Sy)/(GPUmem*0.00008))
        # if maxres < 1:
        #     maxres = 1
        # img = cv2.resize(img, (int(Sx/maxres),int(Sy/maxres)))

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)

        # image to device
        img_LR = img_LR.to(self.device)

        output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        # cv2.imwrite(out_filename, output)
        is_success, im_buf_arr = cv2.imencode(".png", output)
        im_buf_arr.tofile(out_filename)
