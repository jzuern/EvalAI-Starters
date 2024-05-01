from PIL import Image
import numpy as np


def get_metrics(test_annotation_file: str, user_submission_file: str) -> dict:
    
    pred = Image.open(user_submission_file)
    target = Image.open(test_annotation_file)
    
    # if either of the two has an alpha channel, remove it
    if pred.mode == "RGBA":
        pred = pred.convert("RGB")
    if target.mode == "RGBA":
        target = target.convert("RGB")
        
    # if shape is different resize to the target shape
    if pred.size != target.size:
        pred = pred.resize(target.size)
        
    # convert to np
    pred = np.array(pred)
    target = np.array(target)
            
    metrics = {}
    metrics["psnr"] = compute_psnr(pred, target, mask=None).item()
    
    print(metrics)
    
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
    psnr = 10 * np.log10(1 / mse)

    return psnr



