import argparse
import datetime
import math
import torch
import torch.nn as nn
import tqdm
from pathlib import Path
import numpy as np
from Sim_LSCSnet_model import LPTNetIF


# -----------------------------------------------------------------
def cosine_ramp(epoch, start, end, target):
    """
    0 before 'start', cosine ramp to 'target' between start…end,
    flat ('target') afterwards.
    """
    if epoch < start:
        return 0.0
    t = min((epoch - start) / (end - start), 1.0)  # 0 → 1
    return target * 0.5 * (1 - math.cos(math.pi * t))  # cosine ease-in


def psnr(pred, tgt, maxv=1):
    mse = (pred - tgt).pow(2).mean()
    return 10 * torch.log10(maxv ** 2 / mse)


def sam_torch(x, y, eps=1e-12, degrees=True):
    dot = (x * y).sum(dim=-1)
    nx = x.norm(dim=-1)
    ny = y.norm(dim=-1)
    denom = torch.clamp(nx * ny, min=eps)
    cosang = torch.clamp(dot / denom, -1.0, 1.0)
    ang = torch.arccos(cosang)
    return torch.rad2deg(ang) if degrees else ang


def generate_K(num_epochs, start, end):
    # Initialize variables
    dim = 6950
    K_initial = int(np.round(dim * 0.001))  # Initial K value as an integer

    # Define a non-linear progression (e.g., exponential)
    x = np.linspace(0, 1, num_epochs - (start + end))  # Linearly spaced values between 0 and 1
    progression = np.power(x, 2)  # Use a quadratic curve for slower start and faster finish

    # Scale the progression to fit the range between K_initial and dim
    K_array = K_initial + progression * (dim - K_initial)

    # Round and convert to integers
    K_array = np.round(K_array).astype(int)
    K_array[-1] = dim

    # Ensure the last int(end) values are exactly equal to dim
    M_array = np.ones([end, ]) * dim

    # Creating a 3x1 array of zeros
    zeros_array = np.zeros([start, ])

    # Concatenate along the first axis (vertically)
    K_array = np.concatenate((zeros_array, K_array, M_array)).astype(int)

    return K_array


# -------------------------------
#   Dip-aware weighted MSE
# -------------------------------
def iw_mse(pred, target, *, beta=4.0, gamma=5.0,
           max_w=20.0, normalize=True):
    """
    Intensity-weighted MSE (row-1 loss).
    pred, target : (..., L) tensors in [0,1] transmittance
    beta         : overall gain (>0)
    gamma        : sharpness  (>=1)
    max_w        : optional upper cap to keep gradients stable
    normalize    : divide by mean(w) so the scale ≈ plain MSE
    """
    # dip depth (0 at baseline, 1 at full absorption)
    depth = 1.0 - target
    weights = 1.0 + beta * depth.pow(gamma)

    if max_w is not None:
        weights = torch.clamp(weights, max=max_w)

    if normalize:
        weights = weights / weights.mean()

    # detach so β/γ don’t get gradients, but keep dtype/device
    weights = weights.detach()

    return (weights * (pred - target).pow(2)).mean()


def total_variation(x, p=2):
    diff = x[:, 1:] - x[:, :-1]
    return diff.abs().mean() if p == 1 else diff.pow(2).mean()


def generate_universal_samp_map(all_samp_maps, K):
    # Reshape all_samp_maps to [total_samples, input_dim] if needed
    all_samp_maps = all_samp_maps.reshape(all_samp_maps.shape[0], -1)

    # Calculate the variance across the samples for each index (axis 0 is along samples)
    variances = np.var(all_samp_maps, axis=0)

    # Calculate the mean across the samples for each index
    mean_samp_map = np.mean(all_samp_maps, axis=0)

    # Get indices of the K smallest variances (most certain)
    highest_certain_indices = None
    if K != 0:
        sorted_indices = np.argsort(variances)
        highest_certain_indices = sorted_indices[:K]

    # Create a new universal sample map
    universal_samp_map = mean_samp_map

    # Convert to tensor and reshape to [1, input_dim]
    universal_samp_map = torch.tensor(universal_samp_map).reshape(1, -1)

    return universal_samp_map, highest_certain_indices


# -----------------------------------------------------------------
def build_loader(pt_path, batch, device, shuffle):
    """expects .pt file with keys 'sp' (B,6950)"""
    blob = torch.load(pt_path, map_location=device, weights_only=False)
    t = blob['data']['data']
    blob_sp = torch.zeros((len(t), 6950))
    for i, sp in enumerate(t):
        blob_sp[i, :] = sp
    blob_sp *= torch.ones(blob_sp.shape) * 10       # Times 10 so data will be normalized to 1 instead of 0.1
    return torch.utils.data.DataLoader(blob_sp, batch_size=batch,
                                       shuffle=shuffle, drop_last=True)


# -----------------------------------------------------------------
def train_one_epoch(model, dl, optm, entropy_lambda, sum_lambda, tv_lambda, device):
    model.train()
    running = 0
    all_samp_maps = []

    beta = 5.0  # ⇐ tune
    gamma = 4.0  # ⇐ tune
    entropy_sum = 0
    sum_sum = 0
    tv_sum = 0

    for sp_ref in tqdm.tqdm(dl, leave=False):
        sp_ref = sp_ref.to(device)

        sp_bin_pred, samp_map = model(sp_ref)

        mse_iw = iw_mse(sp_bin_pred, sp_ref, beta=beta, gamma=gamma)
        entropy = (samp_map * (1 - samp_map)).mean()
        mask_sum = (samp_map.sum() - model.M) ** 2
        tv = total_variation(sp_bin_pred)

        loss = mse_iw + entropy_lambda * entropy + sum_lambda * mask_sum + tv_lambda * tv

        optm.zero_grad()
        loss.backward()
        optm.step()
        running += loss.item()
        entropy_sum += entropy_lambda * entropy
        sum_sum += sum_lambda * mask_sum
        tv_sum += tv_lambda * tv
        all_samp_maps.append(samp_map.detach().cpu().numpy())

    all_samp_maps = np.concatenate(all_samp_maps, axis=0)
    return running / len(dl), all_samp_maps, entropy_sum / len(dl), sum_sum / len(dl), tv_sum / len(dl)


def validate(model, dl, device):
    model.eval()
    ps, loss = 0, 0
    with torch.no_grad():
        for sp_ref in dl:
            sp_ref = sp_ref.to(device)
            if model.uni_samp_map is not None and len(model.highest_certain_indices) == len(sp_ref[0, :]):
                if_sig = torch.fft.ifft(sp_ref, dim=-1)
                if_sig *= model.get_binary_samp_map(model.uni_samp_map)
                sp_masked = torch.fft.fft(if_sig, dim=-1).abs()
            else:
                sp_masked = sp_ref
            sp_bin, _ = model(sp_masked)
            loss += nn.functional.mse_loss(sp_bin, sp_ref).item()
            ps += psnr(sp_bin, sp_ref).item()
    n = len(dl)
    return loss / n, ps / n


# -----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",
                   help="folder with train.pt and val.pt containing SP tensors",
                   default=r"spectrogram_dataset_small\spectrogram_dataset_small")
    p.add_argument("--save_dir",
                   default=r"saved_models_1d")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--M", type=int, default=200)
    p.add_argument("--lr", type=float, default=4.11e-4)
    p.add_argument("--starting_epoch", type=int, default=15)
    p.add_argument("--ending_epoch", type=int, default=7)
    p.add_argument("--entropy_lambda", type=float, default=11e-2)
    p.add_argument("--sum_lambda", type=float, default=25e-8)
    p.add_argument("--tv_lambda", type=float, default=2e-1)
    opt = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # --- data ----------------------------------------------------
    tr_pt = Path(opt.data_path + "_train.pt")   # changed from train_undersampled to test small batch
    va_pt = Path(opt.data_path + "_valid.pt")
    train_dl = build_loader(tr_pt, opt.batch_size, device, shuffle=True)
    valid_dl = build_loader(va_pt, opt.batch_size, device, shuffle=False)
    blob = torch.load(tr_pt, map_location=device, weights_only=False)
    wavenumbers = blob['wavenumber']

    # --- model ---------------------------------------------------
    model = LPTNetIF(iters=2, M=opt.M).to(device)
    optm = torch.optim.Adam(model.parameters(), opt.lr)

    # --- training loop ------------------------------------------
    best_psnr = 0
    save_dir = Path(opt.save_dir) / datetime.date.today().isoformat()
    save_dir.mkdir(exist_ok=True, parents=True)

    end = opt.ending_epoch
    K_array = generate_K(opt.epochs, opt.starting_epoch, end)

    warmup_start = 5  # first 5 epochs → λ = 0
    warmup_end = 30  # cosine warm-up finishes here

    entropy_max = opt.entropy_lambda  # plateau value
    sum_max = opt.sum_lambda
    tv_max = opt.tv_lambda

    for epoch in range(1, opt.epochs + 1):

        entropy_lambda = cosine_ramp(epoch, warmup_start, warmup_end, entropy_max)
        sum_lambda = cosine_ramp(epoch, warmup_start, warmup_end, sum_max)
        tv_lambda = cosine_ramp(epoch, warmup_start, warmup_end, tv_max)

        tr_loss, all_samp_maps, entropy_loss, sum_loss, tv_loss = train_one_epoch(model, train_dl, optm,
                                                                                  entropy_lambda, sum_lambda, tv_lambda,
                                                                                  device)
        va_loss, va_psnr_bin = validate(model, valid_dl, device)
        print(f"Ep {epoch:02d} | train {tr_loss:.4e} | "
              f"val {va_loss:.4e} | entropy {entropy_loss:.6f} | sum_loss {sum_loss:.6f} | tv_loss {tv_loss:.6f} | PSNR binary {va_psnr_bin:.2f} dB | K {K_array[epoch - 1]}")

        if opt.starting_epoch < epoch < (opt.epochs - 1):
            universal_samp_map, highest_certain_indices = generate_universal_samp_map(all_samp_maps, K_array[epoch - 1])
            model.highest_certain_indices = torch.tensor(highest_certain_indices, dtype=torch.long, device=device)
            model.uni_samp_map = universal_samp_map.to(device)

        if best_psnr < va_psnr_bin:
            best_psnr = va_psnr_bin

        if epoch == opt.epochs - opt.ending_epoch:
            best_psnr = 0

        if epoch > opt.epochs - opt.ending_epoch:
            state_dict = model.state_dict()
            uni = model.uni_samp_map
            highest = model.highest_certain_indices

    print("Finished, best val PSNR:", best_psnr)
    torch.save({
        'model_state_dict': state_dict,
        'uni_samp_map': uni,
        'highest_certain_indices': highest,
        'wavenumbers': wavenumbers
    }, save_dir / f"lpt_sim_final_M{model.M} with {best_psnr:.2f}dB.pth")
    print("Saved final model")


# -----------------------------------------------------------------
if __name__ == "__main__":
    main()


# Example run in google colab:
# !python /content/drive/MyDrive/EE_Project/Sim_Main_train_script.py \
#       --data_path='/content/drive/MyDrive/EE_Project/spectrogram_dataset' \
#       --save_dir='/content/drive/MyDrive/EE_Project/saved_models_1d' \
#       --starting_epoch 10  --epochs 15 --lr 4.11e-4 --M 200 --ending_epoch 4 --entropy_lambda 11e-2 --sum_lambda 5e-8 --tv_lambda 2e-1