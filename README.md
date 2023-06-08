# napari-mcsim
Napari plugin automating structured illumination microscopy (SIM) reconstruction and diagnostics 
from the [mcSIM](https://github.com/QI2lab/mcSIM) package

# Installation
Create a python environment using favorite package manager (conda, mamba, etc...) with Python >= 3.9.
For example, with conda
```
conda create -n napari-mcsim python=3.9
conda activate napari-mcsim
```

Once you have activated that environment, the best way to use this python package is to install it with pip
```
git clone https://github.com/QI2lab/napari-mcsim.git
cd napari-mcsim
pip install .
```

If you would like to edit the code, then install using the `-e` option,
```
git clone https://github.com/QI2lab/napari-mcsim.git
cd napari-mcsim
pip install -e .
```

For more information on GPU support through CUDA, please see the installation instructions in the [mcSIM repo](https://github.com/QI2lab/mcSIM).

# Usage
Activate the conda environment (if you not already) and start napari
```
conda activate napari-mcsim
napari
```

The napari-mcsim plugin is now available in the "plugins" menu. A detailed video walkthrough is in the works.
