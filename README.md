# MonoPatchNeRF: Improving Neural Radiance Fields with Patch-based Monocular Guidance

This is the official repo for PyTorch implementation of paper "MonoPatchNeRF: Improving Neural Radiance Fields with Patch-based Monocular Guidance".

### [Paper](https://arxiv.org/abs/2404.08252) | [Project](https://yuqunw.github.io/MonoPatchNeRF/)

## Setup
### Prerequest
We test our repo with a single Nvidia RTX 3090Ti. Please decrease the target batch size if GPU memory is smaller.
### Environment
- Clone the repository locally: `git clone https://github.com/yuqunw/monopatch_nerf.git`
- Create and activate environment: `conda create -n monopatchnerf python=3.9` and `conda activate monopatchnerf`.
- Install torch and torchvision: `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia`
- Install other required package: `pip install -r requirements.txt`

### Dataset
- Download our preprocessed ETH3D dataset from [here](https://drive.google.com/drive/folders/1eejC2a1Mf47AK2SCAZEYpastNkJZodqa?usp=share_link). Please refer to [ETH3D website](https://www.eth3d.net) for the original high resolution dataset

### Custom Dataset:
- Prepare images with sparse models processed by [colmap](https://colmap.github.io)
- Install our pip package for [omnidata](https://github.com/EPFL-VILAB/omnidata) and [ADE20K semantic segmentation](https://github.com/CSAILVision/semantic-segmentation-pytorch) by running `pip install git+https://github.com/leejaeyong7/OmnidataModels` and `pip install git+http://github.com/leejaeyong7/ADE20KSegmenter`. Note that we only provide an API, and all the models and weights are entirely attributed to the original authors.
- Prepare monocular depth, normals, masks, transforms file, initialization, and depth and SfM points alignment (for density restriction): `python scripts/preprocess_eth3d.py -i ${image_folder} -o ${output_folder} -s ${sparse_folder}`



## Usage
### Training
```
python train.py --data_path "${DATA_DIR}/${scene}" \
                --output_path ${OUTPUT_DIR} \
                --experiment_name ${scene}
```
The default setting uses all proposed components. Run `python train.py -h` for more options and instructions.

### Rendering
Render all input views with checkpoints:
```
python scripts/eval.py --model_checkpoint_file "${OUTPUT_DIR}/${scene}/checkpoints/model/model_steps_${num_iters}.ckpt" \
                       --grid_checkpoint_file "${OUTPUT_DIR}/${scene}/checkpoints/grid/grid_steps_${num_iters}.ckpt" \
                       --data_path "${DATA_DIR}/${scene}/" \
                       --output_path "${OUTPUT_DIR}/${scene}/output" \
                       --full True
```
### Point Cloud Fusion
Fuse point clouds with input views' poses and depths:
```
python scripts/fusion.py --output_path ${OUTPUT_PATH}/${scene}/output \
                         --sparse_path ${SPARSE_PATH}/${scene}/sparse
```
The fused point cloud is `${OUTPUT_PATH}/${scene}/results/fused.ply`

### Evaluate
Evaluate the rendered RGB images and fused point clouds:
```
python scripts/report.py --input_path "${DATA_DIR}/${scene}" \
                         --output_path "${CHECKPOINT_DIR}/${scene}/output" \
                         --gt_path "${GT_DIR}/${scene}" 
```
The result is `${OUTPUT_PATH}/${scene}/results/restuls.json`, containing PSNR, SSIM, LPIPS for novel view synthesis, and F1, precision, and recall for point cloud evaluation.


