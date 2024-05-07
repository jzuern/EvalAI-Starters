import os
import zipfile
import numpy as np
import argparse
from PIL import Image
import json

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
    
    # Clean up temporary folder
    os.system(f"rm -rf {tmp_folder_pred}")
    os.system(f"rm -rf {tmp_folder_target}")

    
    os.makedirs(tmp_folder_pred, exist_ok=True)
    os.makedirs(tmp_folder_target, exist_ok=True)

    test_camera_names = ['ff']
    per_image_evaluation_metrics = ['ssim', 'psnr']
    allowed_image_formats = ['png', 'jpg', 'jpeg']
    

    with zipfile.ZipFile(zipfile_pred, 'r') as zip_pred:
        zip_pred.extractall(tmp_folder_pred)
    with zipfile.ZipFile(zipfile_target, 'r') as zip_target:
        zip_target.extractall(tmp_folder_target)
        
    tmp_folder_pred = os.path.join(tmp_folder_pred, 'wayve_scene_reconstruction_benchmark/scenes')
    tmp_folder_target = os.path.join(tmp_folder_target, 'wayve_scene_reconstruction_benchmark/scenes')
    
    # define the content for an empty image submission
    non_result = {
        "psnr": 0.0,
        "ssim": 0.0,
    }
    
    image_counter = 0
    
    # Init the metrics dict. The dict structure will follow the directory structure of the unpacked zip files
    metrics_dict = {}
        
    # iterate over all scenes in the root folder
    for scene in os.listdir(tmp_folder_target):
        metrics_dict[scene] = {}
        
        # iterate over all cameras in the scene
        for camera in os.listdir(os.path.join(tmp_folder_target, scene)):
            if camera not in test_camera_names:
                continue
                        
            # init the dict for the camera
            metrics_dict[scene][camera] = {}
                        
            # iterate over all images in the camera folder
            for image in os.listdir(os.path.join(tmp_folder_target, scene, camera)):
                imfile_target = os.path.join(tmp_folder_target, scene, camera, image)
                imfile_pred = os.path.join(tmp_folder_pred, scene, camera, image)
                
                if check_valid_image(imfile_target) and check_valid_image(imfile_pred):
                    image_result = get_metrics_single_image(imfile_pred, imfile_target)
                    image_counter += 1          
                    print(f"Image {image_counter} processed")       
                else:
                    image_result = non_result
                    
                metrics_dict[scene][camera][image] = image_result
                                
            # iterate over all images for the camera and average metrics
            images_in_scene = list(image for image in metrics_dict[scene][camera] if image.split('.')[-1] in allowed_image_formats)
            for metric in per_image_evaluation_metrics:
                metrics_dict[scene][camera][metric] = np.mean([metrics_dict[scene][camera][image][metric] for image in images_in_scene])
                
        # iterate over all cameras in the scene and average all per_camera metrics
        cameras_in_scene = list(camera for camera in metrics_dict[scene] if camera in test_camera_names)
        for metric in per_image_evaluation_metrics:
            metrics_dict[scene][metric] = np.mean([metrics_dict[scene][camera][metric] for camera in cameras_in_scene])
            
    # Get metrics for the whole dataset
    scenes_in_dataset = list(scene for scene in metrics_dict)
    for metric in per_image_evaluation_metrics:
        metrics_dict[metric] = np.mean([metrics_dict[scene][metric] for scene in scenes_in_dataset])
        

    print (json.dumps(metrics_dict, indent=2))


    # Clean up temporary folder
    os.system(f"rm -rf {tmp_folder_pred}")
    os.system(f"rm -rf {tmp_folder_target}")
    
    return metrics_dict



