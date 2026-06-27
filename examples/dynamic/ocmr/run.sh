#!/bin/bash

repo_dir="/dir/to/nfrecon"
out_dir="/dir/to/desired/output"
data_path="dir/to/undersampled/kspace"

export PYTHONPATH=$PYTHONPATH:${repo_dir}
acc=8

uv run --project ${repo_dir} nfrecon reconstruct --multirun \
    --config-dir "${repo_dir}/examples/dynamic/ocmr/configs" \
    setup.out_dir=${out_dir} \
    setup.use_mlflow=True \
    setup.device="cuda:0" \
    data=ocmr \
    data.data_path=${data_path} \
    data.data_keys.kspace="kspace_rectilinear_gro_acc_${acc}" \
    data.data_keys.kspace_coords="kspace_coords_rectilinear_gro_acc_${acc}" \
    samplers=ocmr \
    model=ocmr \
    loss=ocmr \
    optimizer=ocmr \
    hydra.sweep.dir=${out_dir}
