
import pathlib

import os


working_dir_path = pathlib.Path().absolute()


features_PATH = os.path.join(str(working_dir_path), 'features')

images_PATH = os.path.join(str(working_dir_path), 'images')


models_PATH = os.path.join(str(working_dir_path), 'models')

recordings_PATH = os.path.join(str(working_dir_path), 'recordings')
