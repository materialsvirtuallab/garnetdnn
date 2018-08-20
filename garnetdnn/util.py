import os
import pickle
import re

import tensorflow as tf
from keras.models import load_model
from pymatgen import MPRester, Composition
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.io.vasp.sets import _load_yaml_config

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
CONFIG = _load_yaml_config("MPRelaxSet")
LDAUU = CONFIG["INCAR"]['LDAUU']['O']

# ENTRIES_PATH = os.path.join(DATA_DIR, "garnet_entries_unique.json")
# GARNET_CALC_ENTRIES_PATH = os.path.join(DATA_DIR, "garnet_calc_entries.json")
# BINARY_OXIDES_PATH = os.path.join(DATA_DIR, "binary_oxide_entries.json")

# GARNET_ENTRIES_UNIQUE = loadfn(ENTRIES_PATH)
# GARNET_CALC_ENTRIES = loadfn(GARNET_CALC_ENTRIES_PATH)
# BINARY_OXDIES_ENTRIES = loadfn(BINARY_OXIDES_PATH)

ELS = {'garnet': {
    'C': [get_el_sp(i) for i in
          ['Bi3+', 'Hf4+', 'Zr4+', 'La3+', 'Pr3+', 'Nd3+', 'Sm3+', 'Eu3+',
           'Gd3+', 'Tb3+', 'Dy3+', 'Ho3+', 'Er3+', 'Tm3+', 'Yb3+', 'Lu3+',
           'Y3+', 'Cd2+', 'Zn2+', 'Ba2+', 'Sr2+', 'Ca2+', 'Mg2+', 'Na+']],

    'A': [get_el_sp(i) for i in
          ['Rh3+', 'Ru4+', 'Cr3+', 'Sb5+', 'Ta5+', 'Nb5+', 'Sn4+', 'Ge4+',
           'Hf4+', 'Zr4+', 'Ti4+', 'In3+', 'Ga3+', 'Al3+', 'Lu3+', 'Yb3+',
           'Tm3+', 'Er3+', 'Ho3+', 'Dy3+', 'Y3+', 'Sc3+', 'Zn2+', 'Mg2+',
           'Li+']],

    'D': [get_el_sp(i) for i in
          ['As5+', 'P5+', 'Sn4+', 'Ge4+', 'Si4+', 'Ti4+', 'Ga3+', 'Al3+',
           'Li+']]},
    'perovskite': {
        'A': [get_el_sp(i) for i in
              ['Al3+', 'Ba2+', 'Bi3+', 'Ca2+', 'Cd2+', 'Ce3+', 'Ce4+', 'Dy3+',
               'Er3+', 'Gd3+', 'Ho3+', 'La3+', 'Mg2+', 'Mn2+', 'Nd3+', 'Ni2+',
               'Pb2+', 'Pd2+', 'Pr3+', 'Pt2+', 'Rh3+', 'Sc3+', 'Sm3+', 'Sn4+',
               'Sr2+', 'Tb3+', 'Tl3+', 'Tm3+', 'Y3+', 'Zn2+']],

        'B': [get_el_sp(i) for i in
              ["Al3+", "Au3+", "Bi3+", "Ce3+", "Ce4+", "Co2+", "Co3+", "Cr3+",
               "Cu2+", "Dy3+", "Er3+", "Eu3+", "Fe2+", "Fe3+", "Ga3+", "Gd3+",
               "Ge4+", "Hf4+", "Ho3+", "In3+", "Ir4+", "La3+", "Lu3+", "Mg2+",
               "Mn2+", "Mn4+", "Mo4+", "Nd3+", "Ni2+", "Os4+", "Pb4+", "Pd4+",
               "Pr3+", "Pt4+", "Re4+", "Rh3+", "Ru4+", "Sc3+", "Si4+", "Sm3+",
               "Sn4+", "Ta5+", "Tb3+", "Tc4+", "Ti4+", "Tl3+", "Tm3+", "V5+",
               "W4+", "Y3+", "Zn2+", "Zr4+"]]}

}

STD_FORMULA = {'garnet': Composition("C3A2D3O12"),
               "perovskite": Composition("A2B2O6")}
SITES = {'garnet': ['c', 'a', 'd'],
         'perovskite': ['a', 'b']}  # use list to preserve order
SITE_INFO = {'garnet': {'c': {"num_atoms": 3, "max_ordering": 20, "cn": "VIII"},
                        'a': {"num_atoms": 2, "max_ordering": 7, "cn": "VI"},
                        'd': {"num_atoms": 3, "max_ordering": 18, "cn": "IV"}},
             'perovskite': {'a': {"num_atoms": 2, "max_ordering": 10, 'cn': "XII"},
                            'b': {"num_atoms": 2, "max_ordering": 10, 'cn': "VI"}}}


m = MPRester("xNebFpxTfLhTnnIH")

MODELS = {}


def load_model_and_scaler(structure_type, model_type):
    """
    Load model and scaler for Ef prediction.

    Args:
        structure_type (str): garnet or perovskite
        model_type (str): type of models
            ext_c : Extended model trained on unmix+cmix
            ext_a : Extended model trained on unmix+amix
            ext_d : Extended model trained on unmix+dmix

    Returns:
        model (keras.model)
        scaler(keras.StandardScaler)
        graph(tf.graph)
    """
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../models/%s" % structure_type)
    model = load_model(os.path.join(MODEL_DIR,
                                    "model_%s.h5" % model_type))
    graph = tf.get_default_graph()
    with open(os.path.join(MODEL_DIR,
                           "scaler_%s.pkl" % model_type), "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, graph


def html_formula(f):
    return re.sub(r"([\d.]+)", r"<sub>\1</sub>", f)


def spe2form(structure_type, species):
    """
    Transfer from a given species dict to the
    standard perovskite formula. (A2B2O6)

    Args:
        structure_type (str): garnet or perovskite
        species (dict): species in dictionary.
            e.g. for Ca2Ti2O6,
                species = {
                            "a": {"Ca2+": 1},
                            "b": {"Ti4+": 1}}
            e.g. for CaSrTi2O6:
                species = {"a": {"Ca2+":0.5,
                                "Sr2+": 0.5},
                           "b": {"Ti4+": 1}}
    Returns:
        formula (str)
    """
    sites = SITES[structure_type]

    spe_list = [spe.name + str(round(SITE_INFO[structure_type][site]['num_atoms'] \
                                     * species[site][spe]))
                for site in sites for
                spe in sorted(species[site], key=lambda x: species[site][x])]
    formula = "".join(spe_list)
    num_oxy = int(STD_FORMULA[structure_type]['O'])
    formula = formula.replace("1", "") + 'O%s' % num_oxy
    return formula

def norm_species(structure_type, species):
    # sites = SITES[structure_type]
    # try:
    #     norm_species = {site:{spe.__str__(): SITE_INFO[structure_type][site]['num_atoms'] \
    #                           / species[site][spe]}
    #                     for site in sites for
    #                     spe in species[site]}
    # except:
    #     norm_species = {site: {get_el_sp(spe).__str__(): SITE_INFO[structure_type][site]['num_atoms'] \
    #                                           / species[site][spe]}
    #                     for site in sites for
    #                     spe in species[site]}
    norm_spec = {}
    sites = SITES[structure_type]
    for site in sites:
        norm_spec[site] = {}
        for spe, amt in species[site].items():
            if isinstance(spe, str):
                spe = get_el_sp(spe)
            norm_spec[site].update({spe: amt/SITE_INFO[structure_type][site]['num_atoms']})
    return norm_spec

def parse_composition(structure_type, s, ctype):
    toks = s.strip().split()
    if len(toks) == 1:
        c = Composition({toks[0].split(":")[0]: 1})
    else:
        c = Composition({t.split(":")[0]: float(t.split(":")[1])
                         for t in toks})
        c = Composition({k2: v2 / sum(c.values()) for k2, v2 in c.items()})
        if len(c) != 2:
            raise ValueError("Bad composition on %s." % ctype)
        frac = [c.get_atomic_fraction(k) for k in c.keys()]

        if structure_type == 'garnet':
            if ctype == "A":
                if abs(frac[0] - 0.5) > 0.01:
                    raise ValueError("Bad composition on %s. "
                                     "Only 1:1 mixing allowed!" % ctype)
            elif ctype in ["C", "D"]:
                if not (abs(frac[0] - 1.0 / 3) < 0.01 or abs(
                        frac[1] - 1.0 / 3) < 0.01):
                    raise ValueError("Bad composition on %s. "
                                     "Only 2:1 mixing allowed!" % ctype)
        elif structure_type == 'perovskite':
            if abs(frac[0] - 0.5) > 0.01:
                raise ValueError("Bad composition on %s. "
                                 "Only 1:1 mixing allowed!" % ctype)
    try:
        for k in c.keys():
            k.oxi_state
            if k not in ELS[structure_type][ctype]:
                raise ValueError("%s is not a valid species for %s site."
                                 % (k, ctype))
    except AttributeError:
        raise ValueError("Oxidation states must be specified for all species!")

    return c
