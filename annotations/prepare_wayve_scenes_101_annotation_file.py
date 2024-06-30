"""
In this file, we take an existing wayve_scenes_101 directory and create a zipfile for it.
The zipfile is then used to evaluate the submission. 
It only contains every 50th image for evaluation purposes.

Original wayve_scenes_101 directory size: ca. 500 MB x 101 = 50.5 GB
New wayve_scenes_101 directory size: ca. 10 MB x 101 = 1 GB
"""

wayve_scenes_101_dir =       "/Users/jannikzurn/data/wayve_scenes_101"
wayve_scenes_101_dir_small = "/Users/jannikzurn/data/wayve_scenes_101_small/wayve_scenes_101"

cameras = ["front-forward", "left-forward", "right-forward", "left-backward", "right-backward"]

every_n_image = 50

import os 
import zipfile
from PIL import Image

# iterate over scene directories in the original wayve_scenes_101 directory
for scene in sorted(os.listdir(wayve_scenes_101_dir)):
    if not os.path.isdir(os.path.join(wayve_scenes_101_dir, scene)):
        continue
    
    print(f"Processing scene {scene}")
    
    scene_dir = os.path.join(wayve_scenes_101_dir, scene)
    scene_dir_small = os.path.join(wayve_scenes_101_dir_small, scene)
    
    os.makedirs(scene_dir_small, exist_ok=True)
    
    # iterate over camera directories in the scene directory
    for camera in cameras:
        camera_dir = os.path.join(scene_dir, 'images', camera)
        camera_dir_small = os.path.join(scene_dir_small, 'images', camera)
        
        os.makedirs(camera_dir_small, exist_ok=True)
        
        # iterate over image files in the camera directory
        for i, image in enumerate(os.listdir(camera_dir)):
            if i % every_n_image == 0:
                image_path = os.path.join(camera_dir, image)
                image_path_small = os.path.join(camera_dir_small, image)
                
                os.system(f"cp {image_path} {image_path_small}")
                
                # make the image in image_path_small smaller (downsampling by factor 4)
                im = Image.open(image_path_small)
                im = im.resize((im.width // 4, im.height // 4))
                im.save(image_path_small)
                
                
# create a zipfile for the small wayve_scenes_101 directory
zipfile_path = wayve_scenes_101_dir_small + "/wayve_scenes_101.zip"

# write the small wayve_scenes_101 directory to the zipfile. The unpacked should have the wayve_scenes_101     
with zipfile.ZipFile(zipfile_path, 'w') as zipf:
    for root, dirs, files in os.walk(wayve_scenes_101_dir_small):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), wayve_scenes_101_dir_small))