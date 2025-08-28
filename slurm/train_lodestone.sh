#!/bin/bash
#BSUB -g /d.goldfarb/compute
#BSUB -G compute-d.goldfarb
#BSUB -a 'docker(dennisgoldfarb/pytorch_ris:lightning)'
#BSUB -J lodestone-train
#BSUB -q general
#BSUB -R "gpuhost rusage[mem=40GB] span[hosts=1]"
#BSUB -gpu 'num=1:gmem=6G:gmodel=NVIDIAA40'
#BSUB -n 8
#BSUB -W 1440
#BSUB -M 40G
#BSUB -o /scratch1/fs1/d.goldfarb/Lodestone/logs/train.%J.out.txt
#BSUB -e /scratch1/fs1/d.goldfarb/Lodestone/logs/train.%J.err.txt

REPO_DIR=/storage1/fs1/d.goldfarb/Active/Projects/Lodestone

cd ${REPO_DIR}
PYTHONPATH=src \
    /bin/python3 -m lodestone.train --config config.json
