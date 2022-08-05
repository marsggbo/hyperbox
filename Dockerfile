# Build: docker build -t project_name .
# Run: docker run --gpus all -it --rm project_name

# Build from official Nvidia PyTorch image
# GPU-ready with Apex for mixed-precision support
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/
FROM nvcr.io/nvidia/pytorch:22.07-py3


# Copy all files
ADD . /workspace/hyperbox
WORKDIR /workspace/hyperbox


# Create hyperbox
RUN conda env create -f conda_env_gpu.yaml -n hyperbox
RUN conda init bash


# Set hyperbox to default virtual environment
RUN echo "source activate hyperbox" >> ~/.bashrc
