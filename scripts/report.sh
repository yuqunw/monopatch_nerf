#!/bin/bash
# SCENES=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
SCENES=(terrains)
ABLATIONS=(default_qff)
INPUT_PATH=/mnt/data/eth3d_neuralangelo
OUTPUT_PATH=/mnt/data/eth3d_outputs/ablations/neuralangelo_eth3d
GT_PATH=/mnt/data/eth3d_ground_truths/
mkdir -p ${EVAL_PATH}

for ablation in "${ABLATIONS[@]}" 
do
    echo $ablation
    for scene in "${SCENES[@]}" 
    do
	    echo "$ablation, $scene"
        python scripts/report.py --input_path ${INPUT_PATH}/${scene} --output_path ${OUTPUT_PATH}/${scene}/output --gt_path ${GT_PATH}/${scene} --threshold 2.0
    done
done
