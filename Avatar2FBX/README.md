# Avatar2FBX

Convert SMPL-like Avatar to FBX format.

- Based on [Smplx2FBX](https://github.com/mrhaiyiwang/Smplx2FBX)
- Based on FBX SDK Python 2020.2
- For scientific research purpose only bounded by license: https://smpl-x.is.tue.mpg.de/modellicense

## TODO
- [ ] Support any pose and any beta SMPL models.
- [ ] Add animation.

## Usage

### 1. Setup conda Environment
```
# Common
conda create -n smpl python=3.6
conda activate smpl
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit==10.1.243 -c pytorch
conda install tqdm

# Specific
pip install 'smplx[all]'
pip install chumpy
```
You will also need to download and setup SMPL models in folder `../smpl_models` as described [here](../README.md).


### 2. Install Python FBX SDK
- Download FBX Python SDK from https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3
- Unzip the downloaded file and follow the instructions in `Install_FbxSdk.txt`


### 3. Edit paths
Edit the following paths according to your environment:
- In `export_fbx.py` and `fbx_utils.py`:
```
sys.path.append('/path/to/fbxsdk/FBX202031_FBXPYTHONSDK_LINUX/lib/Python37_x64')
```


### 4. Convert models
- Add all of your `.ply` files in the folder: `meshes`
- Run the script `export_fbx.py`, the output fbx models will be saved into the folder: `outputs`.
- Note that we currently only supports 'stand pose' and 'zero beta'.