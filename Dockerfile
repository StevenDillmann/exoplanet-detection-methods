# Use a base image that includes Conda
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /sd2022

# Copy your application code into the container
COPY . /sd2022

# Create the environment: gal4_sd2022_env
RUN conda env create --file environment.yml

# Activate the environment and install JupyterLab
SHELL ["conda", "run", "-n", "exo", "/bin/bash", "-c"]
RUN conda install -y jupyterlab

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Set the default command to launch JupyterLab upon container start
CMD ["conda", "run", "--no-capture-output", "-n", "exo", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
