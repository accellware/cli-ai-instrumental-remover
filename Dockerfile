# syntax=docker/dockerfile:1.6
#
# Multi-stage build for the music-separator CLI.
#
# FFmpeg 8.1.1 is built from source so the container matches the Windows
# host (which uses ffmpeg-8.1.1-full_build-shared). Debian trixie's apt
# only ships FFmpeg 7.1.
#
# Stages:
#   ffmpeg-build  debian:trixie-slim + build-essential → /opt/ffmpeg
#   builder       rust:1-trixie → cargo build --features ffmpeg
#   runtime       debian:trixie-slim + shared FFmpeg libs from /opt/ffmpeg
#
# Build context = repo root, so impl/ and models/ are reachable.
#   docker build -t music-separator .

ARG FFMPEG_VERSION=8.1.1
ARG FFMPEG_PREFIX=/opt/ffmpeg

# ---------- ffmpeg-build ----------
FROM debian:trixie-slim AS ffmpeg-build

ARG FFMPEG_VERSION
ARG FFMPEG_PREFIX
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        nasm \
        yasm \
        pkg-config \
        ca-certificates \
        curl \
        xz-utils \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz \
    && tar -xJf ffmpeg.tar.xz \
    && cd "ffmpeg-${FFMPEG_VERSION}" \
    && ./configure \
        --prefix="${FFMPEG_PREFIX}" \
        --enable-shared \
        --disable-static \
        --disable-doc \
        --disable-debug \
        --enable-pic \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf "/tmp/ffmpeg-${FFMPEG_VERSION}" /tmp/ffmpeg.tar.xz

# ---------- builder ----------
FROM rust:1-trixie AS builder

ARG FFMPEG_PREFIX
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        clang \
        libclang-dev \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg-build ${FFMPEG_PREFIX} ${FFMPEG_PREFIX}

ENV PKG_CONFIG_PATH=${FFMPEG_PREFIX}/lib/pkgconfig
ENV LD_LIBRARY_PATH=${FFMPEG_PREFIX}/lib

WORKDIR /build

# Cache deps: copy manifests first, fetch, then copy sources.
COPY impl/Cargo.toml impl/Cargo.lock ./
RUN mkdir -p src && echo "fn main() {}" > src/main.rs \
    && cargo fetch --locked \
    && rm -rf src

COPY impl/src ./src
COPY impl/tests ./tests

RUN cargo build --release --features ffmpeg --locked \
    && strip target/release/music-separator

# ---------- runtime ----------
FROM debian:trixie-slim AS runtime

ARG FFMPEG_PREFIX
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg-build ${FFMPEG_PREFIX} ${FFMPEG_PREFIX}

ENV LD_LIBRARY_PATH=${FFMPEG_PREFIX}/lib
ENV PATH=${FFMPEG_PREFIX}/bin:${PATH}

WORKDIR /app

COPY --from=builder /build/target/release/music-separator /usr/local/bin/music-separator
COPY models/mdxnet /app/models/mdxnet
COPY docker/config.json /app/config.json

# /inputs is mounted by the user; /out captures results.
RUN mkdir -p /inputs /out
VOLUME ["/inputs", "/out"]

ENTRYPOINT ["music-separator"]
CMD ["--help"]
