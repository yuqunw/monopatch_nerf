#!/bin/bash
SCENES=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
ABLATIONS=(default_qff)
RESULT_PATH=/mnt/data1/eth3d_outputs/ablations
EVAL_PATH=/mnt/data1/eth3d_results/ablations
mkdir -p ${EVAL_PATH}

for ablation in "${ABLATIONS[@]}" 
do
    echo $ablation
    for scene in "${SCENES[@]}" 
    do
	    echo "$ablation, $scene"
        python scripts/report.py --scene $scene --result_path ${RESULT_PATH}/${ablation} --eval_path ${EVAL_PATH}/${ablation}/${scene}.json --threshold 2.0
    done
done
