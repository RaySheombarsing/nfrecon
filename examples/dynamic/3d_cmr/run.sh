#!/bin/bash

repo_dir="/dir/to/nfrecon"
out_dir="/dir/to/desired/output"
data_path="dir/to/undersampled/kspace"

export PYTHONPATH=$PYTHONPATH:${repo_dir}

uv run --project ${repo_dir} nfrecon reconstruct --multirun \
    --config-dir "${repo_dir}/examples/dynamic/3d_cmr/configs" \
    setup.out_dir=${out_dir} \
    setup.use_mlflow=True \
    setup.device="cuda:0" \
    data=3d_cmr \
    data.data_path=${data_path} \
    samplers=3d_cmr \
    model=3d_cmr \
    loss=3d_cmr \
    optimizer=3d_cmr \
    hydra.sweep.dir=${out_dir}
