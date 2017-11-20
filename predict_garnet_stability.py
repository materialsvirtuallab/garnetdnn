# coding: utf-8

"""
This python script is designed for predicting the formation energy
referenced to stable binary oxides (Ef) and energy above convex
hull (Ehull) of unknown garnet materials of the form C3A2D3O12.
C, A and D represents 24c, 16a and 24d Wyckoff sites in
prototypical cubic garnet structure Y3Al5O12(ICSD 170157,
spacegroup Ia-3d). Mixed species supported.The oxidation state
is pre-assigned to be common oxidation state of the elements
(see Table 1 in paper).

How to use :
Parameters:
    -c Species on C site. Mixed species supported
        E.g., "Ca2+ 2 Y3+ 1" specifies a disordered site with Ca2+ and Y3+
              in the ratio of 2:1
    -a Species on A site. Mixed species supported
        E.g., "Ca2+ 2 Y3+ 1" specifies a disordered site with Ca2+ and Y3+
              in the ratio of 2:1
    -d Species on D site. Mixed species supported
        E.g., "Ca2+ 2 Y3+ 1" specifies a disordered site with Ca2+ and Y3+
              in the ratio of 2:1
    -m Path to the model(.h5) file. Default the general model reported in
        the paper

    -s Path to the scaler(.pkl) file. Default the scaler associated with
        the general model.
Return:
    Ef: formation energy referenced to stable binary oxides.(eV/f.u.)
    Ehull: (eV/atom)



For example:
    1) python predict_garnet_stability.py -c Y3+ 3 -a Al3+ 2 -d Al3+ 3
    returns Ef (eV/f.u.),Ehull (eV/atom) of Y3Al5O12 garnet material
    2) python predict_garnet_stability.py -c Ba2+ 3 -a Sn4+ 2 -d Ga3+ 2 Ge4+ 1
    returns Ef (eV/f.u.): -2.626 eV/f.u.,Ehull (eV/atom):0.098 eV/atom
     of Ba3Sn2Ga2GeO12 D-mixed garnet material
"""

# TODO from sp:
# 1. Input checking - what are allowed species on C, A and D? Program needs
#   to quit with useful error if bad species are entered.  (DONE)
#    Check 1:  the right oxidation state
#    Check 2:  the allowed species on the site
#    Check 3:  the total charge being neutral
#    Check 4:  the total site occupancy to be 1
# 2. The entire things needs to be self-contained. No database access,
#   except via MPRester to MP.                              (DONE)
#    Option 1: get entries from MP
#    Option 2: get entries from downloaded entries (GARNET_ENTRIES_UNIQUE)
#    Uncomment line 348-349 & line 393-394 to switch to Option2
#    Both OK
# 3. Change the model.h5 and scaler.pkl to .json            (DONE)
#    For user non-default input, should take h5 and pkl

import os
import json
import pickle
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from keras.models import load_model, model_from_json

from monty.serialization import loadfn

from pymatgen import MPRester, Specie, Composition
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.io.vasp.sets import _load_yaml_config



__author__ = "Shyue Ping Ong, Weike Ye"
__version__ = "1.0"
__maintainer__ = "Weike Ye"
__email__ = "w6ye@ucsd.edu"
__date__ = "Oct 16 2017"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


CONFIG = _load_yaml_config("MPRelaxSet")
LDAUU = CONFIG["INCAR"]['LDAUU']['O']

ELS_PATH = os.path.join(MODULE_DIR, "elements.json")
ENTRIES_PATH = os.path.join(MODULE_DIR, "garnet_entries_unique.json")
BINARY_OXIDES_PATH = os.path.join(MODULE_DIR, "binary_oxide_entries.json")

GARNET_ELEMENTS= loadfn(ELS_PATH)
GARNET_ENTRIES_UNIQUE = loadfn(ENTRIES_PATH)
BINARY_OXDIES_ENTRIES = loadfn(BINARY_OXIDES_PATH)

GARNET_ELS = {
    'c': {get_el_sp(i).element: get_el_sp(i).oxi_state for i in
          ['Bi3+', 'Hf4+', 'Zr4+', 'La3+', 'Pr3+', 'Nd3+', 'Sm3+', 'Eu3+',
           'Gd3+', 'Tb3+', 'Dy3+', 'Ho3+', 'Er3+', 'Tm3+', 'Yb3+', 'Lu3+',
           'Y3+', 'Cd2+', 'Zn2+', 'Ba2+', 'Sr2+', 'Ca2+', 'Mg2+', 'Na+']},

    'a': {get_el_sp(i).element: get_el_sp(i).oxi_state for i in
          ['Rh3+', 'Ru4+', 'Cr3+', 'Sb5+', 'Ta5+',
           'Nb5+', 'Sn4+', 'Ge4+', 'Hf4+', 'Zr4+', 'Ti4+',
           'In3+', 'Ga3+', 'Al3+', 'Lu3+', 'Yb3+', 'Tm3+',
           'Er3+', 'Ho3+', 'Dy3+', 'Y3+', 'Sc3+', 'Zn2+',
           'Mg2+', 'Li+']},

    'd': {get_el_sp(i).element: get_el_sp(i).oxi_state for i in
          ['As5+', 'P5+', 'Sn4+', 'Ge4+', 'Si4+',
           'Ti4+', 'Ga3+', 'Al3+', 'Li+']}
}
SITE_OCCU = {'c': 3, 'a': 2, 'd': 3}

m = MPRester()




def binary_encode(config, mix_site):
    """
    This transferes the config index into binary
    coding
    C-mix: max_len = 5
    A-mix: max_len = 3
    D-mix: max_len = 5
    :param config: int, config index
    :param mix_site:str, the site which contains 2 species
    :return: list of binary encoding of config
    number
    eg., config = 19, return [1, 0, 0, 1, 1]
    """
    ENCODING_LEN = {
        'c':  5,
        'a':  3,
        'd':  5
    }
    max_len = ENCODING_LEN[mix_site]
    get_bin = lambda x: format(x, 'b')
    vb = get_bin(config)
    letter = [int(char) for char in vb]
    letter = [0] * (max_len-len(letter)) + letter
    return letter

def binary_decode(letter):
    letter = [str(i) for i in letter]
    return int(''.join(letter), 2)

def get_decomposed_entries(species, model, scaler):
    """
    Get decomposed entries for mix types
    Args:
        species (dict): species in dictionary.
        model (keras.model): keras model
        scaler(keras.StandardScaler):
                scaler accociated with model

    Returns:
        decompose entries(list):
            list of entries prepared from unmix
            garnets decomposed from input mix
            garnet
    """
    def decomposed(specie_complex):
        """Decompose those have sub-dict to individual dict objects."""
        for site, specie in specie_complex.items():
            spe_copy = specie_complex.copy()
            if len(specie) > 1:
                for spe, amt in specie.items():
                    spe_copy[site] = {spe: 1}
                    yield spe_copy

    decompose_entries = []
    for unmix in decomposed(species):
        descriptors = get_descriptor_ext(unmix)
        form_e = get_form_e_ext(descriptors, model, scaler)
        tot_e = get_tote(form_e, unmix)
        entry = prepare_entry(tot_e, unmix)
        compat = MaterialsProjectCompatibility()
        entry = compat.process_entry(entry)
        decompose_entries.append(entry)

    return decompose_entries


def spe2form(species):
    """
    Transfer from a given species dict to the
    standard garnet formula. (C3A2D3O12)

    Args:
        species (dict): species in dictionary.
            e.g. for Y3Al5O12,
                species = { "c": {"Y": 3},
                            "a": {"Al": 2},
                            "d": {"Al": 3}}
            e.g. for BiBa2Hf2Ga3O12:
                species = {"c": {"Bi":1, "Ba": 2},
                           "a": {"Hf": 2}, "d": {"Ga": 3}}
    Returns:
        formula (str)
    """
    sites = ['c', 'a', 'd']

    spe_list = [spe.name + str(int(SITE_OCCU[site] * amt))
                for site in sites for
                spe, amt in species[site].items()]
    formula = "".join(spe_list)
    formula = formula.replace("1", "") + 'O12'
    return formula


def model_load(model_type):
    """
    Load model and scaler for Ef prediction.
    Models are saved in the GarnetModels.json
    for each model type:
    {
    "model": {parameters:model(keras.model).to_json(),
              "weights":model(keras.model).get_weights()}
    "scaler": serialize_class(scaler(StandardScaler))
    }
    Serialized in Serialization.ipynb
    Args:
        model_type (str): type of mdoels
            ext_c : Extended model trained on unmix+cmix
            ext_a : Extended model trained on unmix+amix
            ext_d : Extended model trained on unmix+dmix
    Returns:
        model (keras.model)
        scaler(keras.StandardScaler)

    """
    MODELS = loadfn("GarnetModels.json")


    model_json = MODELS[model_type]['model']
    model = model_from_json(model_json["parameters"])
    model.set_weights(model_json["weights"])


    from sklearn.preprocessing import StandardScaler
    import serialize_sk as sk
    import sys
    def deserialize_class(cls_repr):
        cls_repr = sk.decode(cls_repr)
        cls_ = getattr(sys.modules[cls_repr['mod']], cls_repr['name'])
        cls_init = cls_()
        for k, v in cls_repr['attr'].items():
            setattr(cls_init, k, v)
        return cls_init
    scaler_json = MODELS[model_type]['scaler']
    scaler = deserialize_class(scaler_json)


    return model, scaler

def get_descriptor_ext(species):
    """
    Prepare the inputs for model prediction.
    i.e. extract the
    [rC', XC', rC'', XC'',rA, XA, rD, XD, b1, b2, b3, b4, b5]
    inputs array for given species.

    Args:
        species (dict): species in dictionary.

    Returns:
        inputs (dict): standard input array.
            e.g. {"X_a": float, "r_a": float,
                  "X_c": float, "r_c": float,
                  "X_d": float, "r_d": float}
    """
    ORDERINGS = {
        'c': 20,
        'a': 7,
        'd': 18
    }

    # To sort the c site based on occupancy
    sites = ['c', 'a', 'd']
    mix_site = [site for site in sites if \
                len(species[site]) == 2]
    if not mix_site:
        #unmixed type
        mix_site = 'c'
        input_spe = [el for site in ['c','c','a','d'] for el in species[site]]
    else:
        #mixed type
        mix_site = mix_site[0]
        sites.remove(mix_site)
        spes = species.copy()
        spes['%s_sorted' % mix_site] = sorted(spes[mix_site],
                                             key=lambda k: \
                                                 (spes[mix_site][k], k))
        input_spe = [el for site in ['%s_sorted' % mix_site] + sites \
                     for el in spes[site]]

    descriptors = [(get_el_sp(spe).ionic_radius, get_el_sp(spe).X)
                   for spe in input_spe]
    descriptors = list(sum(descriptors, ()))
    descriptors_config = [descriptors + binary_encode(config, mix_site) \
                          for config in range(0, ORDERINGS[mix_site])]

    return descriptors_config


def get_form_e_ext(descriptors_ext, model, scaler):
    """
    Get formation energy from the given inputs.

    Args:
        descriptors (dict): input descriptors dict.
        model (keras.model): keras model object.
        scaler (keras.StandardScaler): keras StandardScaler object
    Returns:
        predicted_ef (float): the predicted formation Energy.
    """

    inputs_ext_scaled = scaler.transform(descriptors_ext)
    form_e = min(model.predict(inputs_ext_scaled))[0]

    return form_e

def get_tote(form_e, species):
    """
    Get total energy with respect to given Ef
    and species.

    Args:
        form_e (float): Ef
            the form_e is in eV/f.u.
            where the formula should agree with the species
        species (dict): species in dictionary.
             e.g. {"c": {"Zr4+": 1, "Zn2+": 2},
                   "a": {"Mg2+": 2}, "d": {"Si4+": 3}}

    Returns:
        tot_e (float): the total energy in the same
                        unit as form_e.
    """
    formula = spe2form(species)
    composition = Composition(formula)
    tote = form_e

    for el, amt in composition.items():
        stable_bio_entry = None
        if el.symbol == 'O':
            continue
        if el.symbol in BINARY_OXDIES_ENTRIES:
            stable_bio_entry = BINARY_OXDIES_ENTRIES[el.symbol]

        if not stable_bio_entry:
            stable_bio_df = pd.DataFrame.from_csv(os.path.join(
                MODULE_DIR, 'stable_binary_oxides_garnet.csv'))
            stable_bio_id = stable_bio_df.loc[
                lambda df: df.specie == el.symbol]['mpid'].tolist()[0]
            stable_bio_entry = m.get_entry_by_material_id(
                stable_bio_id, property_data=['e_above_hull',
                                              'formation_energy_per_atom'])
        min_e = stable_bio_entry.uncorrected_energy
        amt_bio = stable_bio_entry.composition[el.name]
        tote += (amt / amt_bio) * min_e

    return tote


def prepare_entry(tot_e, species):
    """
    Prepare entries from total energy and species.

    Args:
        tot_e (float): total energy in eV/f.u.
        species (dict): species in dictionary.

    Returns:
        ce (ComputedEntry)
    """

    formula = spe2form(species)
    composition = Composition(formula)
    elements = [el.name for el in composition]
    potcars = set()

    #all_entries = m.get_entries_in_chemsys(elements=elements)

    all_entries = [e for e in GARNET_ENTRIES_UNIQUE \
                    if set(e.composition).issubset(set(composition))]

    for e in all_entries:
        if len(e.composition) == 1 \
                and e.composition.reduced_formula in elements:
            potcars.update(e.parameters["potcar_symbols"])

    potcars.update({"pbe O"})

    parameters = {"potcar_symbols": list(potcars),
                  "oxide_type": 'oxide'}

    for el in elements:
        if el in LDAUU:
            parameters.update({"hubbards": {el: LDAUU[el]}})

    ce = ComputedEntry(composition=composition, energy=0, parameters=parameters)
    ce.uncorrected_energy = tot_e
    compat = MaterialsProjectCompatibility()
    ce = compat.process_entry(ce)

    return ce


def get_ehull(tot_e, species, unmix_entries=None):
    """
    Get Ehull predicted under given total energy and species. The composition
    can be either given by the species dict(for garnet only) or a formula.

    Args:
        tot_e (float): total energy, the unit is in accordance with given
            composition.
        species (dict): species in dictionary.
        unmix_entries (list): additional list of unmix entries.

    Returns:
        ehull (float): energy above hull.
    """
    formula = spe2form(species)
    composition = Composition(formula)
    unmix_entries = [] if unmix_entries is None else unmix_entries
    elements = [el.name for el in composition]
    #all_entries = m.get_entries_in_chemsys(elements=elements)
    all_entries = [e for e in GARNET_ENTRIES_UNIQUE \
                    if set(e.composition).issubset(set(composition))]
    if not all_entries:
        raise ValueError("Incomplete")
    entry = prepare_entry(tot_e, species)

    phase_diagram = PhaseDiagram(all_entries+[entry]+unmix_entries)

    return phase_diagram.get_e_above_hull(entry)


def main(args):
    args = vars(args)
    species = {}
    data_type = 'unmix'
    for k in ["a", "c", "d"]:
        if len(args[k]) == 1:
            species[k] = {get_el_sp(args[k][0]): 1.0}
        else:
            data_type = 'mix'
            toks = args[k]
            species[k] = {get_el_sp(toks[2 * i]): float(toks[2 * i + 1]) for i
                          in range(int(len(toks) / 2))}
            species[k] = {k2: v2 / sum(species[k].values()) for k2, v2 in
                          species[k].items()}
    # Input Check
    charge = 0
    for site_label, spe_set in species.items():
        # check the occupancy reasonable
        occu = sum([amt for spe, amt in spe_set.items()])
        assert np.isclose(occu, 1), \
            "Total occupancy on site {} is not 1".format(site_label)

        # check specie in my element set
        for k, v in spe_set.items():
            # warning for Yb:
            if k.__str__() == 'Yb3+':
                warnings.warn("The Ehull for Yb-contained compound "
                             "is not accurate due to incomplete entries "
                              "in database")
            if k.element not in GARNET_ELS[site_label]:
                raise ValueError(
                    "{} not allowed on site {}!".format(k.element, site_label))

            # check the oxidation state reasonable
            if k.oxi_state != GARNET_ELS[site_label][k.element]:
                raise ValueError(
                    "Oxi state for {} is wrong!".format(k.element))
            charge += int(k.oxi_state) * SITE_OCCU[site_label] * v
    # Check charge neutrality
    if not np.isclose(charge, 24.0):
        raise ValueError("{} Charge not neutral".format(charge))

    mix_site = [site for site in ['a','c','d'] if \
                len(species[site]) > 1]
    if len(mix_site)>1 :
        raise ValueError("Do not support more than one mixed site")
    elif len(mix_site) == 1:
        mix_site = mix_site[0]
    elif len(mix_site)==0 :
        #unmix sample use ext-c model
        mix_site = 'c'

    # model, scaler = model_load(args["model"], args["scaler"])
    model, scaler = model_load("ext_%s"%mix_site)
    inputs = get_descriptor_ext(species)
    form_e = get_form_e_ext(inputs, model, scaler)
    tot_e = get_tote(form_e, species)
    if data_type == 'mix':
        # For mixed samples, include two decomposed unmixed garnets in the pd
        model_c, scaler_c = model_load('ext_c')
        decompose_entries = get_decomposed_entries(species, model_c, scaler_c)
        ehull_pred = get_ehull(tot_e=tot_e, species=species,
                               unmix_entries=decompose_entries)
    else:
        ehull_pred = get_ehull(tot_e=tot_e, species=species)

    print("Ef: %0.03f eV/f.u. \n"
          "Ehull: %0.03f eV/atom \n" % (form_e, ehull_pred))


if __name__ == "__main__":
    eg1 = "python predict_garnet_stability.py -c Ca2+ 1" \
         " Y3+ 2 -a Al3+ 2 -d Al3+ 3 -m \"path to model.h5\"" \
         " -s \"path to scaler.pkl\""
    eg2 = "python predict_garnet_stability.py -c Mg2+ 3 "\
            "-a Ta5+ 1 Ge4+ 1 -d Al3+ 3 "
    parser = argparse.ArgumentParser(
        description="""
        This python codes is designed for predicting Ef and
        Energy above convex hull(Ehull) of unknown garnet
        materials. Parameters c, a and d represents 24c, 16a
        and 24d Wyckoff site in prototypical cubic garnet
        structure Y3Al5O12(ICSD 170157,spacegroup Ia-3d(230)).
        The inputs takes the formula on each site as well as
        the path to the model.h5 and StandardScaler.pkl file.

        Examples:
            unmixed: {}
            mixed:   {}
        returns the Ef and Ehull of input garnet material."""\
            .format(eg1,eg2))

    parser.add_argument('-c', nargs="+",
                        help='Species on C site. Mixed species supported. '
                             'E.g., "Ca2+ 2 Y3+ 1" specifies a disordered '
                             'site with Ca2+ and Y3+ in the ratio of 2:1.')
    parser.add_argument('-a', nargs="+",
                        help='Species on A site. Mixed species supported. '
                             'E.g., "Ca2+ 2 Y3+ 1" specifies a disordered '
                             'site with Ca2+ and Y3+ in the ratio of 2:1.')
    parser.add_argument('-d', nargs="+",
                        help='Species on D site. Mixed species supported. '
                             'E.g., "Ca2+ 2 Y3+ 1" specifies a disordered '
                             'site with Ca2+ and Y3+ in the ratio of 2:1.')
    # Change the files below to JSON if possible.
    # H5 and pickle is extremely difficult to use.
    # parser.add_argument('-m', '--model',
    #                     default=sorted(glob.glob("garnet_model*.json"))[-1],
    #                     help="H5 version of the model")
    # parser.add_argument('-s', '--scaler',
    #                     default=sorted(glob.glob("garnet_scaler*.json"))[-1],
    #                     help="Pickled scaler file.")
    args = parser.parse_args()
    main(args)

# python predict_garnet_stability.py -c Ca2+ 1 Y3+ 2 -a Al3+ 2 -d Al3+ 3
