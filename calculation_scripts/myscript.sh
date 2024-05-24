#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit -1
fi

config_file=$1

for n in $(seq 0.1 0.1 0.1); 
do
    echo $n
    jq ".coeff_const = $n" $config_file > config_temp.json
    python TopoOptCalculation.py config_temp.json
    full_path=$(jq '.full_path' config_temp.json)
    python plotScript.py "$(eval printf '%s' $full_path)"
done
rm -f config_temp.json
#jq '.' $config_file > config_temp.json

