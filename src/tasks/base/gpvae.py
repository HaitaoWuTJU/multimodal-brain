import gpytorch

class GPVAE(VAE):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super(GPVAE, self).__init__(input_dim, latent_dim, hidden_dim)
        
        # GP Kernel for latent space
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x, t):
        # Encode input x into latent space
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Apply GP to latent space
        gp_latent = self.apply_gp(z, t)

        # Decode GP-transformed latent representation
        recon_x = self.decode(gp_latent)
        return recon_x, mu, logvar

    def apply_gp(self, z, t):
        # Use GP to transform latent space based on time t
        latent_gp = gpytorch.distributions.MultivariateNormal(self.mean_module(t), self.covar_module(t))
        return latent_gp.rsample(torch.Size([z.size(0)]))

    def training_step(self, batch, batch_idx):
        x, t = batch
        recon_x, mu, logvar = self(x, t)
        loss = self.vae_loss(recon_x, x, mu, logvar)
        self.log("train_loss", loss)
        return loss