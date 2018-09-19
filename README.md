# Deep Neural Networks for Accurate Predictions of Crystal Stability

This repo implements a web application utilizing a deep neural network to
predict the formation energies and stability of garnet and perovskite crystals, as described in Ye, W.; Chen, C.; Wang, Z.; Chu, I.-H.; Ong, S. P. Deep neural networks for accurate predictions of crystal stability. Nat. Commun. 2018, 9 (1), 3800 DOI: [10.1038/s41467-018-06322-x](http://dx.doi.org/10.1038/s41467-018-06322-x).

A live version of the web application powered by Heroku is running at
http://crystals.ai.

The web application allows the user to enter the compositions of targeting garnets or perovskites, and outputs the predicted formation energy (Ef) and energy above hull (Ehull).

## Local setup

To run the web application locally, simply type:

```bash
python app.py
```

and go to http://127.0.0.1:5000 on your web browser.

## How to cite
This repos contains data and codes for the work:

      Title: Deep Neural Networks for Accurate Predictions of Crystal Stability

      Authors: Weike Ye , Chi Chen, Zhenbin Wang, Iek-Heng Chu, Shyue Ping Ong

Correspondence and requests for materials should be addressed to Dr. Shyue Ping Ong (ongsp@eng.ucsd.edu)

##  Prerequisites
All the structures and data analysis were carried out using
  -[Pymatgen](http://pymatgen.org).

All the deep learning model manipulation were carried out using

  -[Sklearn](http://scikit-learn.org/stable/)

  -[Keras](http://keras.io)

  -[Tensorflow](http://www.tensorflow.org/ )

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html)

## The structure of the repo
```
├──root_dir

  ├──app.py
    The main code for running the flask website.

  ├──garnetdnn

    ├──formation_energy.py
      Contains code to calculate the formation energy (Ef)

    ├──ehull.py
      Contains code to calculate the energy above hull (Ehull)

    ├──util.py
      Contains utility functions such as loading models and standardizing the format of the species of the input structure.

  ├──data

    ├──binary_oxide_entries.json
      Contains the entries(Pymatgen.ComputedStructureEntry) of the binary oxides Nb2O3 (ICSD 25750), Eu2O3 (ICSD 40472) and Yb2O3 (ICSD 33658).
      They are calculated because they are not corrected included in Materials Project database yet.
      All the entries for the rest of binary oxides are from Materials Project.

    ├──garnet_oxi_table.json
      Dictionary of the bianry oxides information for each species used in generating garnet compositions.
      Each entry contains the
      Key: Species, eg "Al3+"
      Values:
          {
           "#"(int): The index, eg. 14,
           "binary oxide"(str): The reduced formula of the binary oxides, eg. "Al2O3",
          "icsd"(int): The ICSD index for the structure, eg. 43732,
          "mpid"(str): The Materials Project ID for the structure, eg.  "mp-1143"
          "oxidation_state"(int): oxidation state in the oxide, eg. 3 (for Al3+)
          "specie"(str): The element symbol of the species, eg. "Al" (for Al3+)
        }

    ├──perovskite_oxi_table.json
        Same as garnet_oxi_table.json, excepts it contains species for generating perovskite compositions.

    ├──garnet
          ├──garnet_calc_entries.json
          ├──garnet_calc_structure_entries.json.gz
            The former contains the calculated entries(Pymatgen.ComputedEntry) for each calculated garnet compositions.
            The latter contains the calculated entries with structures.

    ├──perovskite
          ├──perov_calc_entries.json
          ├──perov_calc_structure_entries.json.gz
            The former contains the calculated entries(Pymatgen.ComputedEntry) for each calculated perovskite compositions.
            The latter contains the calculated entries with structures.
  ├──models

    ├──garnet
          ├──model_unmix.h5
          ├──model_average.h5
          ├──model_c.h5
          ├──model_a.h5
          ├──model_d.h5
          ├──scaler_unmix.pkl
          ├──scaler_average.pkl
          ├──scaler_c.pkl
          ├──scaler_a.pkl
          ├──scaler_d.pkl
            The name of "unmix", "gen", "c", "a" and "d" refer to the unmixed model, the averaged model and the three ordered models for C-mixed, A-mixed and D-mixed garnets. The .h5 files are the model files whereas the scaler.pkl files are the scaler files used to normalize inputs.

    ├──perovskite
          ├──model_unmix.h5
          ├──model_a.h5
          ├──model_b.h5
          ├──scaler_unmix.pkl
          ├──scaler_a.pkl
          ├──scaler_b.pkl
            The name of "unmix", "gen", "a" and "b" refer to the unmixed model, the averaged model and the two ordered models for A-mixed and B-mixed perovskites. The .h5 files are the model files whereas the scaler.pkl files are the scaler files used to normalize inputs.

```  

## Data and Models loading

All the data is stored in .json format

- Example to load binary_oxide_entries.json

```python
import json
with open("data/binary_oxide_entries.json", "r") as f:
  data = json.load(f)
```
```python
from monty.serialization import loadfn
data = loadfn("data/binary_oxide_entries.json")
```

All the model and scalers are in .h5 and .pkl format respectively.
They can be loaded using keras or pickle, or they can be loaded together using garnetdnn.util.load_model_and_scaler

- Example to load garnet unmix model and scaler

```python
from kears.models import load_model
import pickle
model = load_model("models/garnet/model_unmix.h5")
with open("models/garnet/scaler_unmix.pkl", "rb") as f:
  scaler = pickle.load(f)
```

Or

```python
from garnetdnn.util import load_model_and_scaler
structure_type = 'garnet'
model_type = 'unmix'
model, scaler, graph = load_model_and_scaler(structure_type, model_type)
```
