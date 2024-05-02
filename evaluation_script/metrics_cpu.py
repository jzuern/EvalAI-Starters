from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim


def get_metrics(test_annotation_file: str, user_submission_file: str) -> dict:
    
    print("1 inside get_metrics")
    print("user_submission_file", user_submission_file) 
    print("test_annotation_file", test_annotation_file)
    
    pred = Image.open(user_submission_file)
    target = Image.open(test_annotation_file)
    
    # if either of the two has an alpha channel, remove it
    if pred.mode == "RGBA":
        pred = pred.convert("RGB")
    if target.mode == "RGBA":
        target = target.convert("RGB")
        
    print("2 inside get_metrics")

    # if shape is different resize to the target shape
    if pred.size != target.size:
        pred = pred.resize(target.size)
        
    # convert to np
    pred = np.array(pred)
    target = np.array(target)
    
    # convert to float and normalize
    pred = pred.astype(np.float32) / 255.0
    target = target.astype(np.float32) / 255.0
    
    
    print("3 inside get_metrics")

    metrics = {}
    metrics["psnr"] = compute_psnr(pred, target).item()
    metrics["ssim"] = 0.0
    metrics["lpips"] = 0.0
    metrics["fid"] = 0.0 
    metrics["total"] = 0.25 * (metrics["psnr"] / 30. + metrics["ssim"] + (1 - metrics["lpips"]) + (1 - metrics["fid"]))
    
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



