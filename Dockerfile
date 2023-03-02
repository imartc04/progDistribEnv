# Use the latest Ubuntu LTS as the base image
FROM ubuntu:22.04

# Update the package index and install sudo
RUN apt update && \
    apt install -y sudo && \
    apt install -y curl && \
    apt install -y git

# Set up an alias for "ll" in the .bashrc file
RUN echo "alias ll='ls -alF'" >> /root/.bashrc

# Install auto completion tools for the bash console
RUN apt install -y bash-completion



# Install byobu
RUN apt install -y byobu

# Install conda
RUN apt install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to the PATH
ENV PATH /root/miniconda3/bin:$PATH

# Update conda
RUN conda update conda

# Set up conda channels
RUN conda config --add channels conda-forge && \
    conda config --add channels bioconda
