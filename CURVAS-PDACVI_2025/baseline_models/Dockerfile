FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64 
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

# Set up the nnU-Net environment
ENV nnUNet_raw="/opt/algorithm/nnUNet_raw/"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed/"
ENV nnUNet_results="/opt/algorithm/nnUNet_results/"
ENV nnUNet_n_proc_DA="8"

RUN mkdir -p $nnUNet_raw \
    && mkdir -p $nnUNet_results \
    && mkdir -p $nnUNet_preprocessed

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources/nnUNet_results $nnUNet_results
COPY --chown=user:user resources/nnUNet_preprocessed $nnUNet_preprocessed
COPY --chown=user:user resources/nnUNet_raw $nnUNet_raw
COPY --chown=user:user resources/nnUNetTrainerCustom.py /home/user/.local/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
