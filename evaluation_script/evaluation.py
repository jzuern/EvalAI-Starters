import os
import zipfile
import numpy as np
import argparse
from PIL import Image

from metrics import get_metrics_single_image


def check_valid_image(path: str) -> bool:
    """
    Check if the image at the given path is a valid image file.
    """
    try:
        image = Image.open(path)
        image.verify()
        return True
    except Exception as e:
        print(f"Image {path} is not valid: {e}")
        return False


def evaluate_submission(zipfile_pred, zipfile_target):
    
    tmp_folder_pred = '/tmp/wayve_scene_reconstruction_benchmark_pred_eval'
    tmp_folder_target = '/tmp/wayve_scene_reconstruction_benchmark_target_eval'
    
    os.makedirs(tmp_folder_pred, exist_ok=True)
    os.makedirs(tmp_folder_target, exist_ok=True)

    test_camera_names = ['ff']
    per_image_evaluation_metrics = ['ssim', 'psnr']
    

    with zipfile.ZipFile(zipfile_pred, 'r') as zip_pred:
        zip_pred.extractall(tmp_folder_pred)
    with zipfile.ZipFile(zipfile_target, 'r') as zip_target:
        zip_target.extractall(tmp_folder_target)
        
    tmp_folder_pred = os.path.join(tmp_folder_pred, 'wayve_scene_reconstruction_benchmark/scenes')
    tmp_folder_target = os.path.join(tmp_folder_target, 'wayve_scene_reconstruction_benchmark/scenes')

    
    # Init the metrics dict. The dict structure will follow the directory structure of the unpacked zip files
    metrics_dict = {}
    
    # iterate over all scenes in the root folder
    for scene in os.listdir(tmp_folder_pred):
        metrics_dict[scene] = {}
        
        imfiles_pred_list = []
        imfiles_target_list = []

        # iterate over all cameras in the scene
        for camera in os.listdir(os.path.join(tmp_folder_target, scene)):
            if camera not in test_camera_names:
                continue
                        
            metrics_dict[scene][camera] = {}
                        
            # iterate over all images in the camera folder
            for image in os.listdir(os.path.join(tmp_folder_target, scene, camera)):
                imfile_target = os.path.join(tmp_folder_target, scene, camera, image)
                imfile_pred = os.path.join(tmp_folder_pred, scene, camera, image)
                
                if check_valid_image(imfile_target) and check_valid_image(imfile_pred):
                    metrics_dict[scene][camera][image] = get_metrics_single_image(imfile_pred, imfile_target)
                    
                    imfiles_pred_list.append(imfile_pred)
                    imfiles_target_list.append(imfile_target)
                                
            # iterate over all images for the camera and average metrics
            for metric in per_image_evaluation_metrics:
                metrics_dict[scene][camera][metric] = np.mean([metrics_dict[scene][camera][image][metric] for image in metrics_dict[scene][camera]])
                
        # iterate over all cameras in the scene and average all per_camera metric
        for metric in per_image_evaluation_metrics:
            metrics_dict[scene][metric] = np.mean([metrics_dict[scene][camera][metric] for camera in metrics_dict[scene]])
            
    # Get metrics for the whole dataset
    for metric in per_image_evaluation_metrics:
        metrics_dict[metric] = np.mean([metrics_dict[scene][metric] for scene in metrics_dict])
        
    print(metrics_dict)

    # Clean up temporary folder
    os.system(f"rm -rf {tmp_folder_pred}")
    os.system(f"rm -rf {tmp_folder_target}")
    
    return metrics_dict



