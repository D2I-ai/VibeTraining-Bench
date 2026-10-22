# syntax=docker/dockerfile:1
#
# trainer -- the container a researcher agent works inside.
#
# It has to satisfy one requirement: an agent given a shell in here can edit an
# SFT script and actually launch slime training on the target model. Everything
# else is a means to that end.
#
# The trainer *is* the upstream slime training image, plus the agent's tools, so
# the image this repo documents and the image that produces the checkpoints are
# the same thing.
#
# Base image. slime brings Megatron-LM, sglang for rollout, flash-attn, ray and
# the torch build all three are compiled against; reproducing that stack by hand
# means owning every version choice in it. Pinned to a nightly tag rather than
# :latest so a rebuild in six months is the same image:
#
#   slimerl/slime:nightly-dev-20260810a-cu129@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f
#
# The trainer base derives from
# lmsysorg/sglang:v0.5.15.post1-cu129@sha256:c527318558f8c9d74a104c3a1f2e404b92da3e3c6a58c5aec2df16e05b56e63d.
# sglang.Dockerfile pins the matching server image so training and serving
# use compatible kernels.
#
# Override for a local experiment with --build-arg SLIME_IMAGE=..., but a change
# committed here means re-running check_stack.py against every other image.
ARG SLIME_IMAGE=slimerl/slime:nightly-dev-20260810a-cu129@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f
FROM ${SLIME_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Node and the agent CLIs -- the only thing this image adds to slime.
#
# Install claude-code and codex for the corresponding research-agent adapters.
# Additional adapters must declare and install any required CLI here.
#
# Pinned, because a scaffold version is part of what produced a run: its prompts,
# its tool loop and its step accounting all move scores with no model changing.
#
# Node itself comes from the nodejs.org tarball rather than nodesource + apt, and
# is pinned to a full version rather than a major. The base image carries Ubuntu
# 24.04's stock archive.ubuntu.com sources and apt is the only thing in this file
# that needs them; where they are slow or blocked the nodesource setup script
# fails to register its repo *and still exits 0*, so `apt-get install nodejs`
# quietly resolves to Ubuntu's own nodejs 18 -- the pin above turning into a
# different major without a word. A tarball plus its published checksum reaches
# one host, cannot half-succeed, and gives every rebuild the same interpreter.
# NODE_SHA256 is the linux-x64 line of
# https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt; move both together.
ARG NODE_VERSION=22.23.2
ARG NODE_SHA256=d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307
ARG CLAUDE_CODE_VERSION=2.1.34
ARG CODEX_VERSION=0.156.1
RUN curl -fsSL -o /tmp/node.tar.xz \
        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
    && echo "${NODE_SHA256}  /tmp/node.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 --no-same-owner \
        --exclude CHANGELOG.md --exclude LICENSE --exclude README.md \
    && rm /tmp/node.tar.xz \
    && npm install -g \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
        "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force

# The cross-image version contract. This is the *source* side of it: the trainer
# writes the checkpoints, so its versions are the ones the servers must match.
# Asserting here catches a base-image bump that moves the trainer out from under
# a contract the servers are still built against.
COPY .python-version containers/model-stack.txt containers/check_stack.py /opt/vtb/
RUN python3 /opt/vtb/check_stack.py

# Agents talk to a grading service and a model endpoint on the host loopback;
# a proxy in between turns both into a confusing failure.
ENV NO_PROXY=localhost,127.0.0.1 \
    no_proxy=localhost,127.0.0.1

# Where the training code is, named so a script does not have to hardcode it.
# Both trees belong to the base image and are left where slime put them: its
# launchers, its setup.py develop install and its docs all assume these paths.
#
# For the current target model the starting points inside the image are
# scripts/models/qwen3.5-9B.sh (the architecture arguments) and
# scripts/run-qwen3-4B-base-sft.sh (the base-model SFT pattern to adapt).
ENV VTB_SLIME_DIR=/root/slime \
    VTB_MEGATRON_DIR=/root/Megatron-LM

WORKDIR /root

LABEL org.opencontainers.image.title="vibetrainbench trainer" \
      org.opencontainers.image.description="slime training image plus the agent CLIs, for a researcher agent to run SFT in"
