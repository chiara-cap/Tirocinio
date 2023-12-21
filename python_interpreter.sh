#!/bin/bash
python="/homes/ccapellino/.conda/envs/clip/bin/python3"
n_gpu=${GPU:-1}
if [[ $1 == "--version" ]] || [[ $1 == "-V" ]]; then
	    $python "$1"
    elif [[ $@ == *"generator3.py"* ]] || [[ $@ == *"import socket"* ]] || [[ $n_gpu == 0 ]] || [[ $@ == *"packaging_tool.py"* ]]; then
	        $python "$@"
	else
		    /usr/bin/srun -Q --immediate=10 --partition=all_serial --gres=gpu:$n_gpu $python "$@"
fi
