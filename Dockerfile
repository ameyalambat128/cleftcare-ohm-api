# Multi-stage build for Kaldi + Python application
# Stage 1: Extract Kaldi components from official image
FROM --platform=linux/amd64 kaldiasr/kaldi:latest AS kaldi-base

# Stage 2: Build application with Kaldi support
FROM --platform=linux/amd64 python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KALDI_ROOT=/opt/kaldi
ENV PATH=$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin:$KALDI_ROOT/src/gmmbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/featbin:$KALDI_ROOT/src/online2bin:$KALDI_ROOT/src/ivectorbin:$PATH

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for Kaldi and audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    bc \
    gawk \
    perl \
    libblas3 \
    liblapack3 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Kaldi binaries and scripts from the Kaldi image
COPY --from=kaldi-base /opt/kaldi/src/bin /opt/kaldi/src/bin
COPY --from=kaldi-base /opt/kaldi/src/fstbin /opt/kaldi/src/fstbin
COPY --from=kaldi-base /opt/kaldi/src/gmmbin /opt/kaldi/src/gmmbin
COPY --from=kaldi-base /opt/kaldi/src/nnet3bin /opt/kaldi/src/nnet3bin
COPY --from=kaldi-base /opt/kaldi/src/featbin /opt/kaldi/src/featbin
COPY --from=kaldi-base /opt/kaldi/src/online2bin /opt/kaldi/src/online2bin
# Include ivectorbin which provides ivector-extract and related tools
COPY --from=kaldi-base /opt/kaldi/src/ivectorbin /opt/kaldi/src/ivectorbin
# Copy all shared library directories (the symlinks in lib/ point to these)
COPY --from=kaldi-base /opt/kaldi/src/lib /opt/kaldi/src/lib
COPY --from=kaldi-base /opt/kaldi/src/base /opt/kaldi/src/base
COPY --from=kaldi-base /opt/kaldi/src/chain /opt/kaldi/src/chain
COPY --from=kaldi-base /opt/kaldi/src/cudamatrix /opt/kaldi/src/cudamatrix
COPY --from=kaldi-base /opt/kaldi/src/decoder /opt/kaldi/src/decoder
COPY --from=kaldi-base /opt/kaldi/src/feat /opt/kaldi/src/feat
COPY --from=kaldi-base /opt/kaldi/src/fstext /opt/kaldi/src/fstext
COPY --from=kaldi-base /opt/kaldi/src/gmm /opt/kaldi/src/gmm
COPY --from=kaldi-base /opt/kaldi/src/hmm /opt/kaldi/src/hmm
COPY --from=kaldi-base /opt/kaldi/src/ivector /opt/kaldi/src/ivector
COPY --from=kaldi-base /opt/kaldi/src/lat /opt/kaldi/src/lat
COPY --from=kaldi-base /opt/kaldi/src/lm /opt/kaldi/src/lm
COPY --from=kaldi-base /opt/kaldi/src/matrix /opt/kaldi/src/matrix
COPY --from=kaldi-base /opt/kaldi/src/nnet2 /opt/kaldi/src/nnet2
COPY --from=kaldi-base /opt/kaldi/src/nnet3 /opt/kaldi/src/nnet3
COPY --from=kaldi-base /opt/kaldi/src/online2 /opt/kaldi/src/online2
COPY --from=kaldi-base /opt/kaldi/src/transform /opt/kaldi/src/transform
COPY --from=kaldi-base /opt/kaldi/src/tree /opt/kaldi/src/tree
COPY --from=kaldi-base /opt/kaldi/src/util /opt/kaldi/src/util
COPY --from=kaldi-base /opt/kaldi/tools/openfst /opt/kaldi/tools/openfst
COPY --from=kaldi-base /opt/kaldi/egs/wsj/s5/steps /app/steps
COPY --from=kaldi-base /opt/kaldi/egs/wsj/s5/utils /app/utils

# Set LD_LIBRARY_PATH for Kaldi shared libraries
ENV LD_LIBRARY_PATH=/opt/kaldi/src/lib:/opt/kaldi/tools/openfst/lib:$LD_LIBRARY_PATH

# Copy the rest of the application code
COPY . /app/

# Make scripts executable
RUN chmod +x /app/steps/* /app/utils/* 2>/dev/null || true
RUN find /app/steps -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
RUN find /app/utils -name "*.pl" -exec chmod +x {} \; 2>/dev/null || true

# Expose the port (optional)
EXPOSE 8080

# Use the exec form of CMD to run your application
CMD ["python", "app.py"]
