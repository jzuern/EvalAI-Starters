from PIL import Image
import numpy as np
# import subprocess
# import sys
# from skimage.metrics import structural_similarity as ssim
# from skimage.io import imread
from scipy import signal
from scipy import ndimage



def get_metrics_single_image(annotation, submission) -> dict:
    
    if isinstance(annotation, str):
        print("Loading annotation image")
        annotation = Image.open(annotation)
    if isinstance(submission, str):
        print("Loading submission image")
        submission = Image.open(submission)
        
    # if either of the two has an alpha channel, remove it
    if submission.mode == "RGBA":
        submission = submission.convert("RGB")
    if annotation.mode == "RGBA":
        annotation = annotation.convert("RGB")
        
    # if shape is different resize to the target shape
    if submission.size != annotation.size:
        submission = submission.resize(annotation.size)
        
    # convert to np
    submission = np.array(submission)
    annotation = np.array(annotation)
    
    # convert to float and normalize to [0, 1]
    submission = submission / 255.0
    annotation = annotation / 255.0
    
    # add some random noise to the submission
    submission = submission + np.random.normal(0, 0.1, submission.shape)
        
    print("3 inside get_metrics")
    
    psnr_value = compute_psnr(submission, annotation)
    ssim_value = np.mean(compute_ssim(submission, annotation))

    metrics = {}
    metrics["psnr"] = psnr_value
    metrics["ssim"] = ssim_value
    # metrics["lpips"] = 1.0
    # metrics["fid"] = 1.0 
    # metrics["total"] = 0.25 * (metrics["psnr"] / 30. + metrics["ssim"] + (1 - metrics["lpips"]) + (1 - metrics["fid"]))
    metrics["total"] = 0.5 * (metrics["psnr"] / 30. + metrics["ssim"])
    
    print(4, metrics)
    
    return metrics


def compute_psnr(
    pred: np.array,
    target: np.array,
) -> np.array:

    assert pred.shape == target.shape, "The predicted and target images must have the same shape."
    assert pred.dtype == target.dtype, "The predicted and target images must have the same data type."
    
    # Compute the MSE
    mse = np.mean((pred - target) ** 2)
    
    # Compute the PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))

    return psnr


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def compute_ssim_single_channel(img1, img2, cs_map=False):
    
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64) * 255.
    img2 = img2.astype(np.float64)  * 255.
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    
    print(window.shape, img1.shape, img2.shape)

    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
        

def compute_ssim(img1, img2, cs_map=False):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    if img1.ndim == 2:
        return compute_ssim_single_channel(img1, img2, cs_map)
    
    elif img1.ndim == 3:
        if cs_map:
            ret = []
            for i in range(img1.shape[2]):
                ret.append(compute_ssim_single_channel(img1[...,i], img2[...,i], cs_map))
            return np.array(ret)
        else:
            return np.array([compute_ssim_single_channel(img1[...,i], img2[...,i], cs_map) for i in range(img1.shape[2])])
    else:
        raise ValueError('Wrong input image dimensions')
                         
                         
                         
        
if __name__ == "__main__":
    
    
    image_original = "/Users/jannikzurn/Downloads/PSNR-example-base.png"
    image_modified = "/Users/jannikzurn/Downloads/PSNR-example-comp-10.jpg"
    
    get_metrics(image_original, image_modified)
    
    print("done")