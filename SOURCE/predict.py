import cv2
import numpy as np
from tqdm.auto import tqdm
import torch

import config
import utils


def run_predict(epoch, dataloader, generator, max_predict_num=-1):
    total_predict_num = len(dataloader)

    if max_predict_num > 0:
        total_predict_num = min(total_predict_num, max_predict_num)

    progress_bar = tqdm(enumerate(dataloader), total=total_predict_num)

    with torch.no_grad():
        for idx, (input_L, _, input_file_info) in progress_bar:
            if idx == total_predict_num:
                break

            input_L = input_L.to(config.DEVICE)
            input_L3 = input_L.repeat(1, 3, 1, 1)

            fake_AB, _ = generator(input_L3)

            fake_LAB = torch.cat([input_L, fake_AB], dim=1)

            fake_LAB = fake_LAB.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            fake_img = cv2.cvtColor(fake_LAB.astype(np.uint8), cv2.COLOR_LAB2BGR)

            input_L3 = input_L3.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            input_L3 = input_L3.astype(np.uint8)

            color_img = None

            if 'color_file_path' in input_file_info:
                color_file_path = str(input_file_info['color_file_path'][0])
                color_img = cv2.imread(color_file_path, cv2.IMREAD_COLOR)
                color_img = cv2.resize(color_img, (config.IMAGE_SIZE, config.IMAGE_SIZE))

            utils.save_pred_output_to_file(epoch, idx, input_L3, fake_img, color_img)
