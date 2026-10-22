# syntax=docker/dockerfile:1
#
# sglang -- one of the two model servers behind a harbor evaluation.
#
# Upstream ships a coherent image, so this adds almost nothing to it: sglang's own
# release brings the torch and torchvision builds its compiled operators were built
# against, which is the part that is painful to reproduce by hand.
#
# Pinned to the sglang release the trainer's slime image is itself built FROM, so
# a served model runs the same kernels that training rolled out with. That is the
# whole reason to name a version here rather than track :latest -- see
# trainer.Dockerfile for the other end of the pairing.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.15.post1-cu129@sha256:c527318558f8c9d74a104c3a1f2e404b92da3e3c6a58c5aec2df16e05b56e63d
FROM ${SGLANG_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Same contract the trainer asserts, checked from the serving side: this is the
# image that reads a checkpoint the trainer wrote, and a transformers major
# disagreement across that boundary is silent rather than loud.
COPY .python-version containers/model-stack.txt containers/check_stack.py /opt/vtb/
RUN python3 /opt/vtb/check_stack.py

# The server binds 0.0.0.0 and is reached from a task container; a proxy in
# between is never what is wanted for a loopback or LAN endpoint.
ENV NO_PROXY=localhost,127.0.0.1 \
    no_proxy=localhost,127.0.0.1

# No ENTRYPOINT of our own. eval/runner/model_server.py builds the whole command
# line -- model path, served name, tensor parallelism, api key, port -- because
# the same payload has to work for both servers and for the host runtime. An
# entrypoint here would only be a second place that decides serving arguments.
LABEL org.opencontainers.image.title="vibetrainbench sglang server" \
      org.opencontainers.image.description="sglang OpenAI-compatible server, pinned to the sglang release the trainer image is built on"
