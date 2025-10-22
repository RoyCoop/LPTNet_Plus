import argparse
import torch, matplotlib.pyplot as plt, numpy as np
from Sim_LSCSnet_model import LPTNetIF


def psnr(pred, tgt, maxv=1):
    mse = (pred - tgt).pow(2).mean()
    return 10 * torch.log10(maxv ** 2 / mse)


def load_dataset(pt_path, device):
    """expects .pt file with keys 'sp' (B,6950)"""
    blob = torch.load(pt_path, map_location=device, weights_only=False)
    t = blob['data']['data']
    blob_sp = torch.zeros((len(t), 6950))
    for i, sp in enumerate(t):
        blob_sp[i, :] = sp
    blob_sp *= torch.ones(blob_sp.shape)*10
    return blob_sp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",
                   help="folder with train.pt and val.pt containing SP tensors",
                   default=r"saved_models_1d\lpt_sim_final_M200 with 29.87dB.pth")
    p.add_argument("--data_path",
                   default=r"spectrogram_dataset_small\spectrogram_dataset_small_test.pt")
    opt = p.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LPT_WEIGHTS = opt.model_dir
    TEST_PT = opt.data_path
    sps = load_dataset(TEST_PT, DEVICE)
    M = int(LPT_WEIGHTS[LPT_WEIGHTS.find('_M') + 2:LPT_WEIGHTS.find(' w')])
    model = LPTNetIF(iters=2, M=M).to(DEVICE)
    checkpoint = torch.load(LPT_WEIGHTS, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        uni_samp_map = checkpoint['uni_samp_map'].to(DEVICE)  # move to GPU if needed
        highest_certain_indices = checkpoint['highest_certain_indices'].to(DEVICE)
        model.uni_samp_map = uni_samp_map
        model.highest_certain_indices = highest_certain_indices
    except:
        pass
    model.eval()  # → deterministic mask
    for p in model.parameters():
        p.requires_grad_(False)

    # ────────────────────────────────────────────────────────────────
    # 3.  Run N samples through the network
    # ────────────────────────────────────────────────────────────────
    wn = checkpoint['wavenumbers']
    wn_hi = wn
    wn_lo = wn
    sps_true = load_dataset(TEST_PT, DEVICE)
    ifs = torch.fft.ifft(sps_true, dim=-1)
    if_sig = ifs.clone()
    if_sig = torch.mul(if_sig, model.get_binary_samp_map(model.uni_samp_map))
    sp_sig = torch.fft.fft(if_sig, dim=-1).abs()

    start = 0
    N = min(2, sps_true.size(0) - start)  # rows to plot

    psnr_list = np.zeros((11,))
    sp_recons = torch.zeros((11, 6950))
    with torch.no_grad():
        for i in range(N):
            idx = start + i
            #sp_masked = sps_true[idx, :].unsqueeze(0)
            #sp_bin, samp_bin = model(sp_masked)
            sp_bin, samp_bin = model(sp_sig[idx, :].unsqueeze(0))

            sp_recon = sp_bin.squeeze().cpu().numpy()
            samp_bin = samp_bin.T.squeeze()
            samp_bin = model.get_binary_samp_map(samp_bin)
            samp_bin = torch.fft.fftshift(samp_bin)

            sp_recons[i, :] = torch.asarray(sp_recon)

            psnr_val = psnr(sp_bin, sps_true[i, :])  # returns a plain float
            print(psnr_val)
            psnr_list[i] = psnr_val

#    return wn_lo, wn_hi, sps, sp_recons, ifs, samp_bin

    def psnr_torch(pred, tgt, maxv=None):
        pred_t = torch.tensor(pred)
        tgt_t = torch.tensor(tgt)
        if maxv is None:
            maxv = float(tgt_t.abs().max().item() if tgt_t.numel() else 1.0)
        mse = torch.mean((pred_t - tgt_t) ** 2)
        return float(10.0 * torch.log10((maxv ** 2) / (mse + 1e-12)))

    # Figure: one row per example
    N_cmp = 2
    fig_cmp, axs = plt.subplots(N_cmp, 1, figsize=(12, 3.2 * N_cmp), sharex=True)

    for i, ax in enumerate(axs):
        y_true = sps_true[i, :]
        y_model = sp_recons[i, :]

        # PSNRs
        psnr_model = psnr_torch(y_model, y_true)

        # Plot
        ax.plot(wn_hi, y_true, linewidth=1.8, color='b', label="Original", alpha=0.9)
        ax.plot(wn_hi, y_model, linewidth=1.3, color='orange', label=f"Model (PSNR {psnr_model:.2f} dB)", alpha=0.9)



        ax.set_ylabel("Absorption")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axs[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    fig_cmp.tight_layout()
    plt.show()


# np.savetxt("LPT.csv", psnr_list, delimiter=",", fmt="%f")  # column)

if __name__ == "__main__":
    main()

