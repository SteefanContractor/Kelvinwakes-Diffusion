#%%
import torch 
from modules import UNet
from ddpm import Diffusion
from utils import plot_images

# %%
device="cuda"
model = UNet().to(device)
ckpt = torch.load('models/DDPM_Uncondtional_grayscale/ckpt.pt')
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
x = diffusion.sample(model, n=6)

#%%
x32 = diffusion.sample(model, n=32)

#%%
import importlib
import sys
importlib.reload(sys.modules['utils'])
from utils import plot_images
plot_images(x)
# %%
plot_images(x32)
# %%
