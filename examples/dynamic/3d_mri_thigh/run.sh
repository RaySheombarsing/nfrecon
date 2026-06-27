#!/bin/bash

repo_dir="/dir/to/nfrecon"
out_dir="/dir/to/desired/output"
data_path="dir/to/undersampled/kspace"

export PYTHONPATH=$PYTHONPATH:${repo_dir}

uv run --project ${repo_dir} nfrecon reconstruct --multirun \
    --config-dir "${repo_dir}/examples/dynamic/3d_mri_thigh/configs" \
    setup.out_dir=${out_dir} \
    setup.use_mlflow=True \
    setup.device="cuda:0" \
    data=thigh \
    data.data_path=${data_path} \
    samplers=thigh \
    model=thigh \
    loss=thigh \
    optimizer=thigh \
    hydra.sweep.dir=${out_dir}
