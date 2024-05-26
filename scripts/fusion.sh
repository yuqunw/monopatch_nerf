#!/bin/bash
SCENES=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
INPUT_PATH=/mnt/data/eth3d_processed
GT_PATH=/mnt/data2/eth3d_ground_truths/
MONO_PATH=/mnt/data2/yuqun_fix/fix/ours_mono
ACMMP_PATH=/mnt/data2/yuqun_fix/fix/ours_acmmp
mkdir -p ${EVAL_PATH}


for scene in "${SCENES[@]}" 
do
    echo "$ablation, $scene"
    python scripts/fusion.py --output_path /mnt/data/cluster_results/fix/ours_acmmp/${scene}/output
done

for scene in "${SCENES[@]}" 
do
    echo "$ablation, $scene"
    python scripts/fusion.py --output_path /mnt/data/cluster_results/fix/ours_mono/${scene}/output
done
