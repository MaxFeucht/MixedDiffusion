import os 
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
from torchvision.datasets import CelebA

def create_dirs(**kwargs):

    vae_flag = "vae" if kwargs["vae"] else ""
    # Check if directory for imgs exists
    for i in range(10000):
        imgpath = f'./imgs/{kwargs["dataset"]}_{kwargs["degradation"]}_{vae_flag}/run_{i}'
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
            break
    
    modelpath = f'./models/{kwargs["dataset"]}_{kwargs["degradation"]}_{vae_flag}'
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    return imgpath, modelpath


def save_video(samples, save_dir, nrow, name="process.mp4"):
    """ Saves a video from Pytorch tensor 'samples'. 
    Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the video"""

    padding = 0
    imgs = []

    for idx in range(len(samples)):
        sample = samples[idx].cpu().detach().numpy()
        sample = np.clip(sample * 255, 0, 255)
        image_grid = make_grid(torch.Tensor(sample), nrow, padding=padding).numpy(
        ).transpose(1, 2, 0).astype(np.uint8)
        image_grid = cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR)
        imgs.append(image_grid)

    video_size = tuple(reversed(tuple(s for s in imgs[0].shape[:2])))
    writer = cv2.VideoWriter(os.path.join(save_dir,name), cv2.VideoWriter_fourcc(*'mp4v'),
                             30, video_size)
    
    for i in range(len(imgs)):
        image = cv2.resize(imgs[i], video_size, fx=0,
                           fy=0, interpolation=cv2.INTER_CUBIC)
        writer.write(image)
    writer.release()


def save_gif(samples, save_dir, nrow, name="process.gif"):
    """ Saves a gif from Pytorch tensor 'samples'. Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the gif"""

    imgs = []

    for idx in range(len(samples)):
        s = samples[idx].cpu().detach().numpy()
        s = np.clip(s * 255, 0, 255).astype(np.uint8)
        image_grid = make_grid(torch.Tensor(s), nrow, padding=0)
        im = Image.fromarray(image_grid.permute(
            1, 2, 0).to('cpu', torch.uint8).numpy())
        imgs.append(im)

    imgs[0].save(os.path.join(save_dir,name), save_all=True,
                 append_images=imgs[1:], duration=0.5, loop=0)
    


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True