# Exoplanets - Coursework Submission (sd2022)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description
This project is associated with the submission of the coursework for the Exoplanets Module as part of the MPhil in Data Intensive Science at the University of Cambridge. The coursework assignment can be found here: [Exoplanets - Coursework Assignment](Exoplanets.pdf). The associated report can be found here: [Exoplanets - Coursework Report](report/exo_sd2022_report.pdf).

## Table of Contents
- [Installation and Usage](#installation-and-usage)
- [Support](#support)
- [License](#license)
- [Project Status](#project-status)
- [Authors and Acknowledgment](#authors-and-acknowledgment)

## Installation and Usage

To get started with the code associated with the coursework submission, follow these steps:

### Requirements

- Python 3.9 or higher installed on your system.
- Conda installed (for managing the Python environment).
- Docker (if using containerisation for deployment).

### Data 

The data used for this project is found in the `sd2022/data` directory. The datasets are the following:

- [ex1\_stars\_image.png](data/ex1_stars_image.png) (relevant for Exercise 1)
- [ex1\_tess\_lc.txt](data/ex1_tess_lc.txt) (relevant for Exercise 1)
- [ex2\_RVs.txt](data/ex2_RVs.txt) (relevant for Exercise 2)

Please make sure to include these data files in the `sd2022/data` directory if you want to run the main notebooks [ex1\_transit.ipynb](src/ex1_transit.ipynb), [ex2\_rv.ipynb](src/ex2_rv.ipynb), [ex2\_rv2.ipynb](src/ex2_rv2.ipynb) and [ex2\_rv3.ipynb](src/ex2_rv3.ipynb).

### Steps

You can either run the code locally using a `conda` environment or with a container using Docker. The Jupyter Notebooks associated with the different parts are located in the `sd2022/src` directory:

- TESS Lightcurve Planet Search: [ex1\_lc.ipynb](src/ex1_lc.ipynb)
- Doppler Radial Velocity Planet Search: 
[ex2\_rv\_search.ipynb](src/ex2_rv_search.ipynb) (0 planets, stellar activity GP only), [ex2\_rv2.ipynb](src/ex2_rv2.ipynb) (1-2 planets, circular orbit) and [ex2\_rv3.ipynb](src/ex2_rv3.ipynb) (1-2 planets, eccentric orbit). 

The Jupyter Notebooks will run faster locally on a high-spec computer (recommended).

#### Local Setup (Using Conda) [RECOMMENDED]

1. **Clone the Repository:**

    Clone the repository to your local machine with the following command:

    ```
    $ git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/a5_exo_assessment/sd2022
    ```

    or simply download it from [Exoplanets - Coursework Repository (sd2022)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/a5_exo_assessment/sd2022).

2. **Navigate to the Project Directory:**

    On your local machine, navigate to the project directory with the following command:

    ```
    $ cd /full/path/to/sd2022
    ```

    and replace `/full/path/to/` with the directory on your local machine where the repository lives in.

3. **Setting up the Environment:**

    Set up and activate the `conda` environment with the following command:

    ```
    $ conda env create -f environment.yml
    $ conda activate exo
    ```

4. **Install ipykernel:**

    To run the notebook cells with `exo`, install the ipykernel package with the following command:

    ```
    python -m ipykernel install --user --name exo --display-name "Python (exo)"
    ```


4. **Open and Run the Notebook:**

    Open the `sd2022` directory with an integrated development environment (IDE), e.g. VSCode or PyCharm, select the kernel associated with the `exo` environment and run the Jupyter Notebooks (located in the `sd2022/src` directory).


#### Containerised Setup (Using Docker)

1. **Clone the Repository:**

    Clone the repository to your local machine with the following command:

    ```
    $ git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/a5_exo_assessment/sd2022
    ```

    or simply download it from [Exoplanets - Coursework Repository (sd2022)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/a5_exo_assessment/sd2022).

2. **Navigate to the Project Directory:**

    On your local machine, navigate to the project directory with the following command:

    ```
    $ cd /full/path/to/sd2022
    ```

    and replace `/full/path/to/` with the directory on your local machine where the repository lives in.

3. **Install and Run Docker:**

    You can install Docker from the official webpage under [Docker Download](https://www.docker.com/).
    Once installed, make sure to run the Docker application.

4. **Build the Docker Image:**

    You can build a Docker image with the following command:

    ```
    $ docker build -t [image] .
    ```

    and replace `[image]` with the name of the image you want to build.

4. **Run a Container from the Image:**

    Once the image is built, you can run a container based on this image:

    ```
    $ docker run -p 8888:8888 [image]
    ```

    This command starts a container from the `[image]` image and maps port `8888` of the container to port `8888` on your local machine. The Jupyter Notebook server within the container will be accessible on JupyterLab at [http://localhost:8888](http://localhost:8888). 

5. **Access and Run the Notebook:**

    After running the container, you'll see logs in the terminal containing a URL with a token. It will look similar to this:

    ```
     http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    ```
    
    Navigate to [http://localhost:8888](http://localhost:8888) and enter the token `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`. Once you accessed JupyterLab, run the Jupyter Notebooks(located in the `sd2022/src` directory) with an `ipykernel` (Python 3).

    **Note:** Make sure that no other Jupter Notebook Servers are running. Otherwise, you might encounter 'Invalid credentials' issues when entering the token. Close any running Jupter Notebook Servers. To stop a running server, use `Ctrl + C` in the terminal where you launched JupyterLab. Also make sure port `8888` is not occupied.


## Support
For any questions, feedback, or assistance, please feel free to reach out via email at [sd2022@cam.ac.uk](sd2022@cam.ac.uk).

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the [LICENSE](LICENSE) file for details.

## Project Status
The project is in a state ready for submission. All essential features have been implemented, and the codebase is stable. Future updates may focus on minor improvements, bug fixes, or optimisations.

## Use of auto-generation tools
GitHub Co-Pilot assisted the author in producing all function docstrings present in the project repository. No specific commands have been given, instead auto-completion suggestions have occasionally been accepted. Other uses of auto-generation tools:

#### ChatGPT Prompt 1: How do I create a custom colour map in Python?

- ChatGPT output: 

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the colors for your colormap
colors = ["red", "green", "blue"]

# Create the colormap
cmap = mcolors.LinearSegmentedColormap.from_list("my_custom_cmap", colors)

# Use your colormap
plt.imshow([[0,1],[2,3]], cmap=cmap)
plt.colorbar()
plt.show()
```

- Modification of the output:

```python
sd2022_gold = '#cfbd62' 
sd2022_silver = '#C0C0C0'
sd2022_bronze =  '#b08d57' 
cmap_gold = mcolors.LinearSegmentedColormap.from_list('my_colormap', ['black',sd2022_gold, 'white'], N=256)
cmap_silver = mcolors.LinearSegmentedColormap.from_list('my_colormap', ['black',sd2022_silver, 'white'], N=256)
cmap_bronze = mcolors.LinearSegmentedColormap.from_list('my_colormap', ['black',sd2022_bronze, 'white'], N=256)
cmap_bronze_r = mcolors.LinearSegmentedColormap.from_list('my_colormap', ['white',sd2022_bronze, 'black'], N=256)
```

## Authors and Acknowledgment
This project is maintained by [Steven Dillmann](https://www.linkedin.com/in/stevendillmann/) at the University of Cambridge.

21st June 2024
