import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        self.activation = torch.nn.LeakyReLU()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            torch.nn.BatchNorm2d(out_channel),
            self.activation,
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
        )

        if stride == 1 and in_channel == out_channel:
            self.identity = torch.nn.Identity()
        else:
            self.identity = torch.nn.Sequential(
                torch.nn.Conv2d(in_channel, out_channel, 1, stride, 0),
                torch.nn.BatchNorm2d(out_channel),
            )

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.activation(output + self.identity(input))

        return output


class BaseVAE(torch.nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def loss(self, *args):
        raise NotImplementedError


class BetaVAE(BaseVAE):
    def __init__(
        self,
        in_channel=3,
        latent_channel=64,
        hidden_channels=[32, 64, 128, 256],
        input_dim=64,
        beta=1.0,
        is_log_mse=False,
        dataset="celeba",
    ):
        if dataset == "celeba":
            in_channel = 3
            latent_channel = 64
            hidden_channels = [32, 64, 128, 256]
            input_dim = 64
        elif dataset == "mnist":
            in_channel = 1
            latent_channel = 32
            hidden_channels = [32, 64, 128]
            input_dim = 28

        super(BetaVAE, self).__init__()

        self.latent_channel = latent_channel
        self.beta = beta
        self.is_log_mse = is_log_mse

        fc_dim = input_dim
        transpose_padding = []
        for _ in range(len(hidden_channels)):
            transpose_padding.append((fc_dim + 1) % 2)
            fc_dim = (fc_dim - 1) // 2 + 1
        transpose_padding.reverse()

        # make encoder
        self.encoder = []
        last_channel = in_channel

        for channel in hidden_channels:
            self.encoder.append(
                torch.nn.Sequential(
                    ResidualBlock(last_channel, channel, 2),
                    ResidualBlock(channel, channel, 1),
                )
            )
            last_channel = channel

        self.encoder.append(
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(last_channel * (fc_dim**2), latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        self.encoder = torch.nn.Sequential(*self.encoder)

        # make decoder
        hidden_channels.reverse()

        self.decoder = []
        last_channel = hidden_channels[0]

        self.decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(latent_channel, last_channel * (fc_dim**2)),
                torch.nn.BatchNorm1d(last_channel * (fc_dim**2)),
                torch.nn.LeakyReLU(),
                torch.nn.Unflatten(1, (last_channel, fc_dim, fc_dim)),
                ResidualBlock(last_channel, last_channel, 1),
            )
        )

        for channel, pad in zip(hidden_channels[1:], transpose_padding[:-1]):
            self.decoder.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(last_channel, channel, 3, 2, 1, pad),
                    torch.nn.BatchNorm2d(channel),
                    torch.nn.LeakyReLU(),
                )
            )
            last_channel = channel

        self.decoder.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    last_channel, last_channel, 3, 2, 1, transpose_padding[-1]
                ),
                torch.nn.BatchNorm2d(last_channel),
                torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d(last_channel, in_channel, 3, 1, 1),
            )
        )
        self.decoder = torch.nn.Sequential(*self.decoder)

        hidden_channels.reverse()

    def encode(self, input):
        ret = self.encoder(input)
        return ret.split(ret.shape[1] // 2, 1)

    def decode(self, input):
        return self.decoder(input)

    def sample(self, batch_size, device="cuda"):
        return self.decode(torch.randn([batch_size, self.latent_channel]).to(device))

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        return self.decode(z), mu, log_var

    def loss(self, input, output, mu, log_var, eps=1e-5):
        loss_recon = (
            ((input - output) ** 2).mean(dim=0).sum()
            if not self.is_log_mse
            else (
                0.5
                * torch.ones_like(input[0]).sum()
                * (
                    (
                        2 * torch.pi * ((input - output) ** 2).mean(1).mean(1).mean(1)
                        + eps
                    ).log()
                    + 1
                )
            ).mean()
        )
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()

        return loss_recon + loss_reg * self.beta, loss_recon.detach(), loss_reg.detach()
