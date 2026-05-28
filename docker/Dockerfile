FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL maintainer="VisoMaster" \
      description="VisoMaster headless VNC container with OBS, filebrowser, noVNC" \
      io.openshift.expose-services="6901:http,5901:xvnc,8000:http,5173:http,8585:http,9091:http,10000:webrtc"

# ── Ports ─────────────────────────────────────────────────────────────────────
# 5901  TigerVNC
# 6901  noVNC (browser VNC)
# 8000  VisoMaster FastAPI server
# 5173  Vite dev server (React web UI)
# 8585  filebrowser
# 9091  streamrelay WebRTC signaling + web client
# 10000 WebRTC media channel (TCP + UDP)
EXPOSE 5901 6901 8000 5173 8585 9091 10000/tcp 10000/udp

# ── Base env ──────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    DISPLAY=:1 \
    VNC_PORT=5901 \
    NO_VNC_PORT=6901 \
    NO_VNC_HOME=/opt/noVNC \
    VNC_COL_DEPTH=24 \
    VNC_RESOLUTION=1280x800 \
    VNC_PW=visomaster \
    VNC_PASSWORDLESS=false \
    HOME=/workspace \
    STARTUPDIR=/dockerstartup \
    CONDA_DIR=/opt/conda \
    CONDA_ENV=visomaster \
    VISOMASTER_HOME=/workspace/VisoMaster \
    FILEBROWSER_PORT=8585 \
    API_PORT=8000 \
    VITE_PORT=5173 \
    STREAMRELAY_PORT=9091 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

WORKDIR /workspace

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core tools
    wget curl git unzip bzip2 ca-certificates gnupg2 software-properties-common \
    apt-transport-https locales tzdata \
    # Build tools
    build-essential cmake pkg-config \
    # VNC / display
    tigervnc-standalone-server tigervnc-common \
    xfce4 xfce4-terminal xfce4-taskmanager \
    dbus-x11 x11-utils x11-xserver-utils xdg-utils \
    # Python / OpenCV runtime libs
    python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libxkbcommon0 libdbus-1-3 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
    # Audio (pyvirtualcam / OBS)
    libpulse0 pulseaudio \
    # v4l2loopback (virtual camera)
    v4l2loopback-dkms v4l2loopback-utils \
    # FFmpeg
    ffmpeg \
    # Misc
    jq nano vim htop procps net-tools curl \
    && locale-gen en_US.UTF-8 \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# ── OBS Studio ────────────────────────────────────────────────────────────────
RUN apt-get update \
    && add-apt-repository -y ppa:obsproject/obs-studio \
    && apt-get update \
    && apt-get install -y --no-install-recommends obs-studio \
    && rm -rf /var/lib/apt/lists/*

# ── Tailscale ─────────────────────────────────────────────────────────────────
# Installed at build time so startup is instant; daemon is started at runtime.
# Kernel mode (full UDP/WebRTC) requires the container to have NET_ADMIN +
# /dev/net/tun — both are available when running --privileged (default in
# docker-compose.yml). Without them it falls back to userspace mode
# automatically (TCP/DERP only — WebRTC media will not work in that case).
RUN curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg \
        | tee /usr/share/keyrings/tailscale-archive-keyring.gpg > /dev/null \
    && curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list \
        | tee /etc/apt/sources.list.d/tailscale.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends tailscale \
    && rm -rf /var/lib/apt/lists/*

# ── noVNC ─────────────────────────────────────────────────────────────────────
RUN git clone --depth 1 --branch v1.5.0 \
        https://github.com/novnc/noVNC.git $NO_VNC_HOME \
    && git clone --depth 1 \
        https://github.com/novnc/websockify.git $NO_VNC_HOME/utils/websockify \
    && ln -s $NO_VNC_HOME/vnc.html $NO_VNC_HOME/index.html

# ── Miniconda ─────────────────────────────────────────────────────────────────
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda clean --all -y

ENV PATH=$CONDA_DIR/bin:$PATH

# ── Conda environment ─────────────────────────────────────────────────────────
RUN conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true \
    && conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true \
    && conda create -n $CONDA_ENV python=3.10.13 -y \
    && conda clean --all -y

ENV PATH=$CONDA_DIR/envs/$CONDA_ENV/bin:$PATH \
    CONDA_DEFAULT_ENV=$CONDA_ENV

# ── CUDA runtime libs inside conda env ───────────────────────────────────────
RUN conda install -n $CONDA_ENV -c nvidia/label/cuda-12.4.1 cuda-runtime -y \
    && conda install -n $CONDA_ENV -c conda-forge cudnn -y \
    && conda clean --all -y

# ── VisoMaster Python dependencies ───────────────────────────────────────────
# Copy only requirements first so this layer is cached unless deps change
COPY requirements_cu124.txt /tmp/requirements_cu124.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements_cu124.txt \
    && rm /tmp/requirements_cu124.txt

# ── filebrowser ───────────────────────────────────────────────────────────────
RUN wget -q -O - https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

# ── VisoMaster source ─────────────────────────────────────────────────────────
# Copy project (respects .dockerignore)
COPY . $VISOMASTER_HOME/
WORKDIR $VISOMASTER_HOME

# Initialise git submodules (streamrelay)
RUN git submodule update --init --recursive 2>/dev/null || true

# ── Startup infrastructure ────────────────────────────────────────────────────
RUN mkdir -p $STARTUPDIR /workspace/logs

COPY docker/scripts/vnc_startup.sh    $STARTUPDIR/vnc_startup.sh
COPY docker/scripts/start_services.sh $STARTUPDIR/start_services.sh
COPY docker/config/xfce4/             /root/.config/xfce4/
COPY docker/desktop/                  /root/Desktop/

RUN chmod +x $STARTUPDIR/vnc_startup.sh $STARTUPDIR/start_services.sh \
    && chmod +x /root/Desktop/*.desktop 2>/dev/null || true

# ── Volume mount points ───────────────────────────────────────────────────────
# /workspace/data  — models, outputs, workspace state (mount a network volume here)
VOLUME ["/workspace/data"]

ENTRYPOINT ["/dockerstartup/vnc_startup.sh"]
CMD ["--wait"]
