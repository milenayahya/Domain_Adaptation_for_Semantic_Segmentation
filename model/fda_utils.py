# Main Reference: https://github.com/YanchaoYang/FDA/blob/master/utils/__init__.py
import torch
import numpy as np

# Using the implementation provided in this issue, https://github.com/YanchaoYang/FDA/issues/40
# since the original implementation was compatible with an older version of PyTorch.


def extract_ampl_phase(fft_im):
    # fft_im: size should be b x 3 x h x w
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha


def low_freq_mutate_A(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)  # get b
    if __name__ == "__main__":
        print(f"A - {b=}, {h=}, {w=}, {L=}")
    if b == 0:
        raise ValueError("L is too small")
    # When rFFT is used only half of the space needs to be updated
    # because of the symmetry along the last dimension
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):

    # get fft of both source and target
    fft_src = torch.fft.fft2(src_img.clone())
    fft_trg = torch.fft.fft2(trg_img.clone())
    fft_src_clone, fft_trg_clone = fft_src.clone(), fft_trg.clone()
    # extract amplitude and phase of both ffts
    amp_src, pha_src = torch.abs(fft_src_clone), torch.angle(fft_src_clone)
    amp_trg, pha_trg = torch.abs(fft_trg_clone), torch.angle(fft_trg_clone)

    # replace the low frequency amplitude part of source with that from target
    amp_src_mutated = low_freq_mutate_A(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    real = torch.cos(pha_src.clone()) * amp_src_mutated.clone()
    imag = torch.sin(pha_src.clone()) * amp_src_mutated.clone()
    fft_src_recomposed = torch.complex(real=real, imag=imag)

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.ifft2(fft_src_recomposed, s=[imgH, imgW])
    src_in_trg = torch.real(src_in_trg)
    src_in_trg = torch.clamp(src_in_trg, 0, 255)

    return src_in_trg


def FDAEntropyLoss(x: "torch.Tensor", eta: float):
    """Computes the robust entropy loss for a given input tensor.

    Parameters:
    - x (torch.Tensor): Input tensor of shape [B, C, H, W], where B is the batch size, C is the number of channels, H is the height, and W is the width.
    - eta (float): A scalar value controlling the degree of robustness in the entropy computation.

    Returns:
    - torch.Tensor: A scalar value representing the mean robust entropy loss.
    """

    P = torch.softmax(x, dim=1)  # [B, 19, H, W]
    logP = torch.log_softmax(x, dim=1)  # [B, 19, H, W]
    PlogP = P * logP  # [B, 19, H, W]
    ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
    ent = ent / 2.9444  # chanage when classes is not 19
    # compute robust entropy
    ent = ent**2.0 + 1e-8
    ent = ent**eta
    return ent.mean()
    
if __name__ == "__main__":
    # Visualize the FDA transformation, import the images
    from Datasets.cityscapes_torch import Cityscapes, CITYSCAPES_CROP_SIZE
    from Datasets.gta5 import GTA5, GTA5_CROP_SIZE
    from Datasets.transformations import *
    from torch.utils.data import DataLoader
    import torchvision

    # Cityscapes
    cityscapes = Cityscapes(
        mode="train",
        transforms=OurCompose([OurResize(CITYSCAPES_CROP_SIZE), OurToTensor()]),
    )
    # GTA
    gta5 = GTA5(
        mode="train",
        transforms=OurCompose(
            [
                OurResize(CITYSCAPES_CROP_SIZE),
                OurToTensor(),
            ]
        ),
    )

    # Create DataLoaders with batchsize 1
    cityscapes_loader = DataLoader(cityscapes, batch_size=1, shuffle=True)
    gta5_loader = DataLoader(gta5, batch_size=1, shuffle=True)

    # get first batch and save images
    cityscapes_batch = next(iter(cityscapes_loader))
    gta5_batch = next(iter(gta5_loader))
    c_img, c_lbl = cityscapes_batch
    gta_img, gta_lbl = gta5_batch
    original = gta_img[0, :, :, :]
    torchvision.utils.save_image(original, "./logs/original.png")
    L = 0.006
    transformedA = FDA_source_to_target(gta_img.clone(), c_img.clone(), L=L)

    target = c_img[0, :, :, :]
    # Save them to folder
    torchvision.utils.save_image(transformedA[0, :, :, :], "./logs/transformedA.png")
    # torchvision.utils.save_image(transformedB[0, :, :, :], "./logs/transformedB.png")
    torchvision.utils.save_image(target, "./logs/target.png")