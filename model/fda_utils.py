# Main Reference: https://github.com/YanchaoYang/FDA/blob/master/utils/__init__.py
import torch
import numpy as np


# def extract_ampl_phase(fft_im):
#     # fft_im: size should be bx3xhxw
#     fft_amp = torch.abs(fft_im)
#     fft_pha = torch.angle(fft_im)
#     return fft_amp, fft_pha


# def low_freq_mutate(amp_src, amp_trg, L=0.1):
#     _, _, h, w = amp_src.size()
#     # multiply w by 2 because we have only half the space as rFFT is used
#     w *= 2
#     b = (np.floor(np.amin((h, w)) * L)).astype(int)  # get b
#     if b > 0:
#         # When rFFT is used only half of the space needs to be updated
#         # because of the symmetry along the last dimension
#         amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]  # top left
#         amp_src[:, :, h - b + 1 : h, 0:b] = amp_trg[
#             :, :, h - b + 1 : h, 0:b
#         ]  # bottom left
#     return amp_src


# def FDA_source_to_target(src_img: "torch.Tensor", trg_img: "torch.Tensor", L=0.01) -> "torch.Tensor":
#     """
#     This function performs Frequency Domain Adaptation (FDA) on the source image using the target image.
#     FDA aims to transfer the style of the target image to the source image while preserving the content.

#     Parameters:
#     - src_img (torch.Tensor): The source image tensor of shape (B, C, H, W)
#     - trg_img (torch.Tensor): The target image tensor of the same shape as src_img.
#     - L (float, optional): A parameter controlling the degree of style transfer (represents the "Beta" value on the paper). Default is 0.1.

#     Returns:
#     - torch.Tensor: The resulting image tensor after applying FDA, of the same shape as src_img.
#     """

#     # get fft of both source and target
#     fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
#     fft_trg = torch.fft.rfft2(trg_img.clone(), dim=(-2, -1))

#     # extract amplitude and phase of both ffts
#     amp_src, pha_src = extract_ampl_phase(fft_src.clone())
#     amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

#     # replace the low frequency amplitude part of source with that from target
#     amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

#     # recompose fft of source
#     real = torch.cos(pha_src.clone()) * amp_src_.clone()
#     imag = torch.sin(pha_src.clone()) * amp_src_.clone()
#     fft_src_ = torch.complex(real=real, imag=imag)

#     # get the recomposed image: source content, target style
#     _, _, imgH, imgW = src_img.size()
#     src_in_trg = torch.fft.irfft2(fft_src_, dim=(-2, -1), s=[imgH, imgW])

#     return src_in_trg


# # Numpy Versions


# def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
#     a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
#     a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

#     _, h, w = a_src.shape
#     b = (np.floor(np.amin((h, w)) * L)).astype(int)
#     c_h = np.floor(h / 2.0).astype(int)
#     c_w = np.floor(w / 2.0).astype(int)

#     h1 = c_h - b
#     h2 = c_h + b + 1
#     w1 = c_w - b
#     w2 = c_w + b + 1

#     a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
#     a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
#     return a_src


# def FDA_source_to_target_np(src_img, trg_img, L=0.1):
#     # exchange magnitude
#     # input: src_img, trg_img

#     src_img_np = src_img  # .cpu().numpy()
#     trg_img_np = trg_img  # .cpu().numpy()

#     # get fft of both source and target
#     fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
#     fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))

#     # extract amplitude and phase of both ffts
#     amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
#     amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

#     # mutate the amplitude part of source with target
#     amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

#     # mutated fft of source
#     fft_src_ = amp_src_ * np.exp(1j * pha_src)

#     # get the mutated image
#     src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
#     src_in_trg = np.real(src_in_trg)

#     return src_in_trg


def extract_ampl_phase(fft_im):
    # fft_im: size should be b x 3 x h x w
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    # multiply w by 2 because we have only half the space as rFFT is used
    w *= 2
    # multiply by 0.5 to have the maximum b for L=1 like in the paper
    b = (np.floor(0.5 * np.amin((h, w)) * L)).astype(int)     # get b
    if b > 0:
        print(f"{L=} {b=} {h=} {w=}")
        # When rFFT is used only half of the space needs to be updated
        # because of the symmetry along the last dimension
        amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]      # top left
        amp_src[:, :, h-b+1:h, 0:b] = amp_trg[:, :, h-b+1:h, 0:b]    # bottom left
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # get fft of both source and target
    fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
    fft_trg = torch.fft.rfft2(trg_img.clone(), dim=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    real = torch.cos(pha_src.clone()) * amp_src_.clone()
    imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(real=real, imag=imag)

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(fft_src_, dim=(-2, -1), s=[imgH, imgW])

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

if __name__ == '__main__':
    
    # Visualize the FDA transformation, import the images
    from Datasets.cityscapes_torch import Cityscapes, CITYSCAPES_CROP_SIZE
    from Datasets.gta5 import GTA5, GTA5_CROP_SIZE
    from Datasets.transformations import *
    from torch.utils.data import DataLoader
    import torchvision
    # Cityscapes
    cityscapes = Cityscapes(mode='train', transforms=OurCompose([OurResize(CITYSCAPES_CROP_SIZE), OurToTensor()]))
    # GTA
    gta5 = GTA5(mode='train', transforms=OurCompose(
            [
                OurResize(CITYSCAPES_CROP_SIZE),
                OurToTensor(),
            ]
        ),)

    # Create DataLoaders with batchsize 1
    cityscapes_loader = DataLoader(cityscapes, batch_size=1, shuffle=True)
    gta5_loader = DataLoader(gta5, batch_size=1)

    # get first batch and save images
    cityscapes_batch = next(iter(cityscapes_loader))
    gta5_batch = next(iter(gta5_loader))
    c_img, c_lbl = cityscapes_batch
    gta_img, gta_lbl = gta5_batch

    transformed = FDA_source_to_target(gta_img, c_img, L=.006)

    original = gta_img[0, :, :, :]
    transformed = transformed[0, :, :, :]
    target = c_img[0, :, :, :]
    # Save them to folder
    torchvision.utils.save_image(original, 'original.png')
    torchvision.utils.save_image(transformed, 'transformed.png')
    torchvision.utils.save_image(target, 'target.png')

