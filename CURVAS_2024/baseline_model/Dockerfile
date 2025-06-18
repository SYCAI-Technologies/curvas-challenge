FROM --platform=linux/amd64 pytorch/pytorch

ENV PYTHONUNBUFFERED 1

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

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
