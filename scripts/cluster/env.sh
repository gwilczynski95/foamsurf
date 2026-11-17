# Environment of the FoamSurf jobs on the GMUM cluster (sourced by the sbatch scripts).
#   repo: /shared/results/gmtobias/repos/foamsurf   env: /shared/results/gmtobias/conda_envs/foamsurf
REPOSITORY=/shared/results/gmtobias/repos/foamsurf
ENVIRONMENT=/shared/results/gmtobias/conda_envs/foamsurf
PYTHON=$ENVIRONMENT/bin/python
export CONDA_ENV_PATH=$ENVIRONMENT CUDA_HOME=$ENVIRONMENT
export OPTIX_DIRECTORY=/shared/results/gmtobias/repos/iris/iris/sampler/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
export TORCH_CUDA_ARCH_LIST=8.9 CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCH_EXTENSIONS_DIR=$REPOSITORY/.torch_extensions   # PPISP's JIT extension, built once by build.sbatch
export PATH=$ENVIRONMENT/bin:/usr/local/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset VIRTUAL_ENV
