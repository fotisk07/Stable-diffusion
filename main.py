import torch
import torch.nn.functional as F
from model import SimpleModel
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from tqdm import tqdm
import matplotlib.pyplot as plt

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def reverse_transform(img):
    img = img.detach().cpu().numpy()
    img = img.squeeze()
    return img


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod):
  # add noise
  x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image

def calculate_loss(denoise_model, x_start, t):
    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod)
    x_denoised = denoise_model(x_noisy)
    
    # calculate loss
    loss = F.mse_loss(x_denoised, x_start)

    return loss


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())

    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MNIST(root='./data', download=True, transform=Compose([Resize((28, 28)), ToTensor(), Lambda(lambda x: (x - 1/2) * 2), ]))
train_data = DataLoader(dataset, batch_size=32, shuffle=True)



mymodel = SimpleModel(1)
mymodel.to(device)

print("Training on device:", device)
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001)

timesteps = 300

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


epochs = 1
for epoch in range(epochs):
    print("Epoch:", epoch)
    for step, batch in enumerate(train_data):
      optimizer.zero_grad()
      batch_size = batch[0].shape[0]
      batch = batch[0].to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,)).long()

      loss = calculate_loss(mymodel, batch, t)

      if step % 500 == 0:
        print("Loss:", loss.item())


      loss.backward()
      optimizer.step()


samples = sample(mymodel, 28, batch_size=16, channels=1)
random_index = 5
plt.imshow(samples[-1][random_index].reshape(28, 28, 1), cmap="gray")
plt.show()
