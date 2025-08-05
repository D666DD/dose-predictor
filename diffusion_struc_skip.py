import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def cosine_similarity(X,Y):
    '''
    compute cosine similarity for each pair of image
    Input shape: (batch,channel,H,W)
    Output shape: (batch,1)
    '''
    b, c, h, w = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)
    corr = norm(X)*norm(Y)#(B,C,H*W)
    similarity = corr.sum(dim=0).mean(dim=1)
    return similarity
def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, condition):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        #print("x_t:", x_t.min(), x_t.max())
        #############################################################################################
        #print("shape!", x_t.shape, condition.shape)
        x_recon1, x_recon2 = self.model(x_t, condition, t)
        #print("x_recon:", x_recon.min(), x_recon.max())
        #print(x_recon.shape)
        #print(F.mse_loss(x_recon, noise, reduction='none'))
        loss_mse1 = F.mse_loss(x_recon1, noise, reduction='none').mean()
        loss_mse2 = F.mse_loss(x_recon2, noise, reduction='none').mean()

        loss= loss_mse2
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        
        self.alphas = 1. - self.betas

        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x_T, condition):
        x_t = x_T
        eta=0.0
        timesteps = torch.linspace(self.T-1, 0, 8, dtype=int)
        timesteps = timesteps.tolist()
        
        for i, time_step in enumerate(timesteps[:-1]):
            t_prev = timesteps[i+1]
    
            # 1. predict eps
    
            t_tensor = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            _, eps = self.model(x_t, condition, t_tensor)
    
            alpha_t = self.alpha_cumprod[time_step]
            alpha_prev = self.alpha_cumprod[t_prev]
    
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    
            # 2. estimate x0
            x0_hat = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
    
            # DDIM sampling
            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
    
            noise = torch.zeros_like(x_t) if eta == 0 else torch.randn_like(x_t)

    
            x_t = torch.sqrt(alpha_prev) * x0_hat + \
                  torch.sqrt(1 - alpha_prev - sigma_t ** 2) * eps + \
                  sigma_t * noise
    
        return x_t

