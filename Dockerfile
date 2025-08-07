ARG BUILD_FROM
FROM $BUILD_FROM

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    linux-headers \
    alsa-lib-dev \
    libffi-dev \
    openssl-dev \
    python3-dev \
    py3-pip \
    cmake \
    git

# Install Python packages
RUN pip3 install --no-cache-dir --upgrade pip

# Install dependencies
RUN pip3 install --no-cache-dir \
    wyoming>=1.2.0 \
    onnxruntime>=1.15.0 \
    numpy \
    librosa \
    kaldi-native-fbank \
    scipy \
    soundfile

# Copy application files
COPY rootfs /

# Make run script executable
RUN chmod a+x /run.sh

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 10305

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD nc -z localhost 10305 || exit 1

# Run
CMD ["/run.sh"]