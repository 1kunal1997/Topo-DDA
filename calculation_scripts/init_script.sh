#!/bin/bash

for n in $(seq 1 400);
do
    echo $n
    python generate_config.py "config.json" "configs/config$n.json" $n
    python TopoOptCalculation.py "configs/config$n.json"
    full_path=$(jq -r '.full_path' "configs/config$n.json")
    python plotScript.py $full_path
    ffmpeg -framerate 10 -pattern_type glob -i "$full_path/Plots/Structures/*.png" -c:v libx264 -pix_fmt yuv420p "$full_path/out.mp4"
    ffmpeg -framerate 10 -pattern_type glob -i "$full_path/Plots/E-Field_ZSlice/*.png" -c:v libx264 -pix_fmt yuv420p "$full_path/E_field.mp4"
done

