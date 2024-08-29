from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import torchvision
import os
import subprocess
import gc
import model as Model
import numpy as np
import pandas as pd


def train_and_test(
    model: Model.BaseVAE,
    epochs=50,
    batch_size=512,
    device="cuda",
    dataset="mnist",
    evaluation=True,
):

    transforms = None
    if dataset == "mnist":
        transforms = [
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.RandomResizedCrop((28, 28), (0.9, 1), (0.9, 1.1)),
            torchvision.transforms.ToTensor(),
        ]
    elif dataset == "celeba":
        transforms = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.CenterCrop(148),
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
        ]
    transforms = torchvision.transforms.Compose(transforms)

    dataset_train = None
    dataset_test = None
    if dataset == "mnist":
        dataset_train = torchvision.datasets.MNIST(
            root="C:/dataset/", transform=transforms
        )
        dataset_test = torchvision.datasets.MNIST(
            root="C:/dataset/", transform=transforms, train=False
        )
    elif dataset == "celeba":
        dataset_train = torchvision.datasets.CelebA(
            root="C:/dataset/", transform=transforms
        )
        dataset_test = torchvision.datasets.CelebA(
            root="C:/dataset/", transform=transforms, split="test"
        )

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, epochs * len(loader_train) // 2 + 1
    )

    name = type(model).__name__ + datetime.now().strftime(" %m%d %H%M")
    name += " beta=" + str(float(model.beta))
    name += " log=" + str(model.is_log_mse)

    writer = SummaryWriter(log_dir="runs/" + name)
    os.makedirs("./result/" + name, exist_ok=True)

    # Main loop
    for epoch in tqdm(range(epochs), leave=False, desc=name):
        model.train()
        loss_total = 0.0
        loss_recon_total = 0.0
        loss_reg_total = 0.0

        # Train loop
        for x, y in tqdm(loader_train, leave=False, desc="Train"):
            x = x.to(device)
            y = y.to(device)

            result = model(x)

            loss, loss_recon, loss_reg = model.loss(x, *result)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_total += float(loss)
            loss_recon_total += float(loss_recon)
            loss_reg_total += float(loss_reg)

        writer.add_scalar("loss/train", loss_total / len(loader_train), epoch)
        writer.add_scalar("recon/train", loss_recon_total / len(loader_train), epoch)
        writer.add_scalar("reg/train", loss_reg_total / len(loader_train), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        model.eval()
        loss_total = 0

        with torch.no_grad():
            # Validation loop
            for x, y in tqdm(loader_test, leave=False, desc="Evaluate"):
                x = x.to(device)
                y = y.to(device)

                result = model(x)
                loss = model.loss(x, *result)[0]
                loss_total += float(loss)

            # Save reconstruction example
            for _ in tqdm(range(1), leave=False, desc="Test"):
                x, _ = next(iter(loader_test))
                x = x.to(device)

                result, _, _ = model(x)

                save_image(
                    x[:256],
                    "./result/" + name + "/" + str(epoch) + "_origin.png",
                    normalize=True,
                    nrow=16,
                )
                save_image(
                    result[:256].clip(0, 1),
                    "./result/" + name + "/" + str(epoch) + "_recon.png",
                    normalize=True,
                    nrow=16,
                )

                # Save sampled example
                result = model.sample(batch_size, device)

                save_image(
                    result[:256].clip(0, 1),
                    "./result/" + name + "/" + str(epoch) + "_sample.png",
                    normalize=True,
                    nrow=16,
                )

        writer.add_scalar("loss/test", loss_total / len(loader_test), epoch)
        if epoch % 10 == 9:
            torch.save(
                model.state_dict(), "./result/" + name + "/model_" + str(epoch) + ".pt"
            )

    writer.close()

    # [FID, likelihood, distortion, rate, test-likelihood, test-distortion, test-rate]
    dat = [0.0] * 7

    # Generate samples to calculate FID score
    if evaluation:
        MC_COUNT = 1  # Number of Monte-Carlo sampling

        with torch.no_grad():
            # Calculate train set likelihood
            for _ in range(MC_COUNT):
                for x, y in tqdm(loader_train, leave=False, desc="Train"):
                    x = x.to(device)
                    y = y.to(device)

                    result = model(x)

                    loss, loss_recon, loss_reg = model.loss(x, *result)

                    dat[1] += float(loss)
                    dat[2] += float(loss_recon)
                    dat[3] += float(loss_reg)

            dat[1] /= len(loader_train) * MC_COUNT
            dat[2] /= len(loader_train) * MC_COUNT
            dat[3] /= len(loader_train) * MC_COUNT

            # Calculate test set likelihood
            for _ in range(MC_COUNT):
                for x, y in tqdm(loader_test, leave=False, desc="Test"):
                    x = x.to(device)
                    y = y.to(device)

                    result = model(x)

                    loss, loss_recon, loss_reg = model.loss(x, *result)

                    dat[4] += float(loss)
                    dat[5] += float(loss_recon)
                    dat[6] += float(loss_reg)

            dat[4] /= len(loader_test) * MC_COUNT
            dat[5] /= len(loader_test) * MC_COUNT
            dat[6] /= len(loader_test) * MC_COUNT

        with torch.no_grad():
            os.makedirs("./result/" + name + "/generation", exist_ok=True)

            SAMPLE_ITERATION = 50
            for i in tqdm(range(SAMPLE_ITERATION), leave=False, desc="Generate"):
                x = model.sample(batch_size, device).clip(0, 1)

                for j in range(batch_size):
                    save_image(
                        x[j],
                        "./result/"
                        + name
                        + "/generation/"
                        + str(i * batch_size + j)
                        + ".png",
                        normalize=True,
                        nrow=1,
                    )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Calculate FID via `pytorch_fid` lib
        try:
            import pytorch_fid

            fid = os.popen(
                f'python -m pytorch_fid C:/fid/{dataset}.npz "./result/{name}/generation/" --device cuda:0 --batch-size 512'
            ).read()
            # print(fid)
            dat[0] = float(fid.split(" ")[-1])

        except ModuleNotFoundError:
            print("Please install `pytorch_fid` to show FID score")

    dat = np.array(dat)
    np.savetxt("./result/" + name + "/result.csv", dat, delimiter=",")

    return dat


if __name__ == "__main__":
    REPEAT = 5

    for dataset in ["celeba", "mnist"]:
        for is_log in [True, False]:
            for beta in [0.01, 0.1, 1.0, 10.0, 100.0]:
                mn = np.array([0.0] * 7)
                dat = []
                for _ in range(REPEAT):
                    dat.append(
                        train_and_test(
                            model=Model.BetaVAE(
                                beta=beta, is_log_mse=is_log, dataset=dataset
                            ),
                            dataset=dataset,
                        )
                    )
                    mn += dat[-1]
                mn /= REPEAT

                std = np.array([0.0] * 7)
                for d in dat:
                    std += (d - mn) ** 2
                std /= REPEAT
                std = np.sqrt(std)

                print("DATASET:", dataset, "IS_LOG:", is_log, "BETA:", beta)
                print(mn, "+-", std)

                result = np.zeros((2, 7))
                result[0] = mn
                result[1] = std
                np.savetxt(
                    f"./result/dataset={dataset} is_log={is_log} beta={beta} cnt={REPEAT}.csv",
                    result,
                    delimiter=",",
                )
