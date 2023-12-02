import os
import shutil
import mmengine
import os.path as osp

def copy_images(source_folder, destination_folder, replacer):
    # Iterate through all subdirectories in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Get the full path of the current file
            file_path = os.path.join(root, file)
            if "inst_" not in file_path:
                file_name, file_extension = os.path.splitext(file)
                new_file_name = file_name.replace(replacer, "") + file_extension
                destination_path = os.path.join(destination_folder, new_file_name)
                shutil.copy(file_path, destination_path)

copy_images("idd20kII/leftImg8bit/train", "idd20kII/train_img", "_leftImg8bit")
copy_images("idd20kII/gtFine/train", "idd20kII/train_label", "_gtFine_labellevel3Ids")
copy_images("idd20kII/leftImg8bit/val", "idd20kII/val_img", "_leftImg8bit")
copy_images("idd20kII/gtFine/val", "idd20kII/val_label", "_gtFine_labellevel3Ids")
print("Image copied successfully")


# define dataset root and directory for images and annotations
data_root = 'idd20kII'
train_img_dir = 'train_img'
train_ann_dir = 'train_label'
val_img_dir = 'val_img'
val_ann_dir = 'val_label'

# split train/val set randomly -> Change this later
split_dir = 'splits'
mmengine.mkdir_or_exist(osp.join(data_root, split_dir))

filename_list = [osp.splitext(filename)[0] for filename in mmengine.scandir(
    osp.join(data_root, train_img_dir), suffix='.jpg')]
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
  f.writelines(line + '\n' for line in filename_list)

filename_list = [osp.splitext(filename)[0] for filename in mmengine.scandir(
    osp.join(data_root, val_img_dir), suffix='.jpg')]
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
  f.writelines(line + '\n' for line in filename_list)

print("Splits created successfully")