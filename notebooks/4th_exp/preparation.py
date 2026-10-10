
# %%
# Paket-pakettttt
import gdown
import subprocess
from pathlib import Path
import os

# %%
cwd = Path.cwd()
project_root_dir = cwd.parents[1]
dataset_root_dir = project_root_dir / "datasets"

### modifiable
dataset_name = "MMSD2.0" # ganti dataset lewat sini
dataset_dir = dataset_root_dir / dataset_name
dataset_data_dir = dataset_dir / "data"
###

# symlink
project_dataset_dir = cwd / "dataset"
if not project_dataset_dir.exists():
    relative_target = os.path.relpath(dataset_data_dir, start=project_dataset_dir.parent)
    project_dataset_dir.symlink_to(relative_target)

# # %%
# # donwload MMSD2.0 images
# gdown.download(url='https://drive.google.com/uc?id=1mK0Nf-jv_h2bgHUCRM4_EsdTiiitZ_Uj', output=project_dataset_dir.as_posix(), quiet=False)
# gdown.download(url='https://drive.google.com/uc?id=1AOWzlOz5hmdO39dEmzhQ4z_nabgzi7Tu', output=project_dataset_dir.as_posix(), quiet=False)
# gdown.download(url='https://drive.google.com/uc?id=1dJERrVlp7DlNSXk-uvbbG6Rv7uvqTOKd', output=project_dataset_dir.as_posix(), quiet=False)
# gdown.download(url='https://drive.google.com/uc?id=1pODuKC4gP6-QDQonG8XTqI8w8ds68mE3', output=project_dataset_dir.as_posix(), quiet=False)

# # pastiin udh install 7z dari package manager
# subprocess.run(["7z", "x", "./dataset/dataset_image.zip", "-o./dataset"])

# # %%
# # download whitelist
# gdown.download("https://drive.google.com/file/d/18yU3HaSvBNYml2EfKn-uG7vUKXGDMt6d/view?usp=drive_link", output=project_dataset_dir.as_posix(), quiet=False)