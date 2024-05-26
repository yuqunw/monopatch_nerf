#!/bin/bash
SCENES=(Auditorium Barn Church Courtroom Meetingroom Ballroom Courthouse Museum)
OUTPUT_PATH=/mnt/data/cluster_results/tnt/ours_d_dg_n_ssim_ncc_cull_200000_full
GT_PATH=/mnt/data2/eth3d_ground_truths/
SPARSE_PATH=/mnt/data/tnt_processed
mkdir -p ${EVAL_PATH}


for scene in "${SCENES[@]}" 
do
    echo "$ablation, $scene"
    python scripts/fuse_tnt.py --output_path ${OUTPUT_PATH}/${scene}/output --sparse_path ${SPARSE_PATH}/${scene}/sparse
done