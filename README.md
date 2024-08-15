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
python scripts/train.py --data_path "${DATA_DIR}/${scene}" \
                --output_path "${OUTPUT_DIR}" \
                --experiment_name "${scene}"
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
python scripts/fusion.py --output_path "${OUTPUT_PATH}/${scene}/output" \
                         --min_views 2 \
                         --threshold 2.0
```
The fused point cloud is `${OUTPUT_PATH}/results/fused.ply`. We use a loose threshold and views for ETH3D scenes. However, if the scene is denser, then the min_views can be larger and fusion threshold can be smaller, e.g., `--min_views=5` and `--threshold=0.5` for TnT. Can specify colmap sparse folder to accelerate the fusion for denser view, e.g., `--sparse_path ${SPARSE_DIR}/${scene}/sparse`.
### Evaluation
Install the point cloud [evaluation program of ETH3D](https://github.com/ETH3D/multi-view-evaluation), download the [ground truth point cloud](https://www.eth3d.net/data/multi_view_training_dslr_scan_eval.7z), change the corresponding path `eth3d_evaluation_bin` in `scripts/report.py`, and run the evaluation for rendered RGB images and fused point clouds:
```
python scripts/report.py --input_path "${DATA_DIR}/${scene}" \
                         --output_path "${OUTPUT_PATH}/${scene}/output" \
                         --gt_path "${GT_DIR}/${scene}/dslr_scan_eval" 
```
The results are in `${OUTPUT_PATH}/${scene}/output/results/restuls.json`, containing PSNR, SSIM, LPIPS for novel view synthesis, and F1, precision, and recall for point cloud evaluation.

### Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{wu2024monopatchnerf,
  title={MonoPatchNeRF: Improving Neural Radiance Fields with Patch-based Monocular Guidance},
  author={Wu, Yuqun and Lee, Jae Yong and Zou, Chuhang and Wang, Shenlong and Hoiem, Derek},
  journal={arXiv preprint arXiv:2404.08252},
  year={2024}
}
```
If you find the QFF representation helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{lee2022qff,
  title={Qff: Quantized fourier features for neural field representations},
  author={Lee, Jae Yong and Wu, Yuqun and Zou, Chuhang and Wang, Shenlong and Hoiem, Derek},
  journal={arXiv preprint arXiv:2212.00914},
  year={2022}
}
```