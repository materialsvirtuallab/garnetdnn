import os
import pandas as pd
import pickle

from keras.models import load_model
import keras
from keras import backend as K
import tensorflow as tf
from monty.serialization import loadfn

from pymatgen import MPRester, Composition
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.io.vasp.sets import _load_yaml_config

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/garnet")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/garnet")
CONFIG = _load_yaml_config("MPRelaxSet")
LDAUU = CONFIG["INCAR"]['LDAUU']['O']

ENTRIES_PATH = os.path.join(DATA_DIR, "garnet_entries_unique.json")
GARNET_CALC_ENTRIES_PATH = os.path.join(DATA_DIR, "garnet_calc_entries.json")
BINARY_OXIDES_PATH = os.path.join(DATA_DIR, "binary_oxide_entries.json")

GARNET_ENTRIES_UNIQUE = loadfn(ENTRIES_PATH)
GARNET_CALC_ENTRIES = loadfn(GARNET_CALC_ENTRIES_PATH)
BINARY_OXDIES_ENTRIES = loadfn(BINARY_OXIDES_PATH)

GARNET_ELS = {
    'C': [get_el_sp(i) for i in
          ['Bi3+', 'Hf4+', 'Zr4+', 'La3+', 'Pr3+', 'Nd3+', 'Sm3+', 'Eu3+',
           'Gd3+', 'Tb3+', 'Dy3+', 'Ho3+', 'Er3+', 'Tm3+', 'Yb3+', 'Lu3+',
           'Y3+', 'Cd2+', 'Zn2+', 'Ba2+', 'Sr2+', 'Ca2+', 'Mg2+', 'Na+']],

    'A': [get_el_sp(i) for i in
          ['Rh3+', 'Ru4+', 'Cr3+', 'Sb5+', 'Ta5+',
           'Nb5+', 'Sn4+', 'Ge4+', 'Hf4+', 'Zr4+', 'Ti4+',
           'In3+', 'Ga3+', 'Al3+', 'Lu3+', 'Yb3+', 'Tm3+',
           'Er3+', 'Ho3+', 'Dy3+', 'Y3+', 'Sc3+', 'Zn2+',
           'Mg2+', 'Li+']],

    'D': [get_el_sp(i) for i in
          ['As5+', 'P5+', 'Sn4+', 'Ge4+', 'Si4+',
           'Ti4+', 'Ga3+', 'Al3+', 'Li+']]
}

SITE_OCCU = {'c': 3, 'a': 2, 'd': 3}

m = MPRester("xNebFpxTfLhTnnIH")

MODELS = {}



def lazy_load_model_and_scaler(model_type):
    """
    Load model and scaler for Ef prediction.

    Args:
        model_type (str): type of models
            ext_c : Extended model trained on unmix+cmix
            ext_a : Extended model trained on unmix+amix
            ext_d : Extended model trained on unmix+dmix

    Returns:
        model (keras.model)
        scaler(keras.StandardScaler)
    """
    if model_type not in MODELS:
        model = load_model(os.path.join(MODEL_DIR,
                                        "model_ext_%s.h5" % model_type))
        graph = tf.get_default_graph()
        with open(os.path.join(MODEL_DIR,
                               "scaler_ext_%s.pkl" % model_type), "rb") as f:
            scaler = pickle.load(f)
        MODELS[model_type] = model, scaler, graph
        return model, scaler, graph
    return MODELS[model_type]


def load_model_and_scaler(model_type):
    """
    Load model and scaler for Ef prediction.

    Args:
        model_type (str): type of models
            ext_c : Extended model trained on unmix+cmix
            ext_a : Extended model trained on unmix+amix
            ext_d : Extended model trained on unmix+dmix

    Returns:
        model (keras.model)
        scaler(keras.StandardScaler)
    """

    model = load_model(os.path.join(MODEL_DIR,
                                    "model_ext_%s.h5" % model_type))
    graph = tf.get_default_graph()
    with open(os.path.join(MODEL_DIR,
                           "scaler_ext_%s.pkl" % model_type), "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, graph


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
        'c': 5,
        'a': 3,
        'd': 5
    }
    max_len = ENCODING_LEN[mix_site]
    get_bin = lambda x: format(x, 'b')
    vb = get_bin(config)
    letter = [int(char) for char in vb]
    letter = [0] * (max_len - len(letter)) + letter
    return letter


def get_decomposed_entries(species):
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
    model, scaler, graph = load_model_and_scaler("c")
    for unmix_species in decomposed(species):
        charge = sum([spe.oxi_state * amt * SITE_OCCU[site]
                      for site in ['a', 'c', 'd']
                      for spe, amt in unmix_species[site].items()])
        if not abs(charge - 2 * 12) < 0.1:
            continue
        descriptors = get_descriptor_ext(unmix_species)
        with graph.as_default():
            form_e = get_form_e_ext(descriptors, model, scaler)
        tot_e = get_tote(form_e, unmix_species)
        entry = prepare_entry(tot_e, unmix_species)
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

    spe_list = [spe.name + str(round(SITE_OCCU[site] * species[site][spe]))
                for site in sites for
                spe in sorted(species[site])]
    formula = "".join(spe_list)
    formula = formula.replace("1", "") + 'O12'
    return formula


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
    CN = {
        'c': 'VIII',
        'a': 'VI',
        'd': 'IV'
    }

    sites = ['c', 'a', 'd']
    mix_site = [site for site in sites if len(species[site]) == 2]
    if not mix_site:
        # use ext-c model for unmixed type
        mix_site = 'c'
        input_spe = [(el, site) for site in ['c', 'c', 'a', 'd']
                     for el in species[site]]

    else:
        # mixed type
        mix_site = mix_site[0]
        sites.remove(mix_site)
        spes = species.copy()
        # sort the mix site on the order of increasing occupancy
        spes['%s_sorted' % mix_site] = sorted(
            spes[mix_site],
            key=lambda k: (spes[mix_site][k], k),
            reverse=True if mix_site == 'a' else False)
        input_spe = [(el, site) for site in ['%s_sorted' % mix_site] + sites
                     for el in spes[site]]

    descriptors = [(get_el_sp(el).ionic_radius, get_el_sp(el).X)
                   for el, site in input_spe]
    descriptors = list(sum(descriptors, ()))
    descriptors_config = [descriptors + binary_encode(config, mix_site)
                          for config in range(0, ORDERINGS[mix_site])]
    if mix_site == 'a':
        descriptors_r = descriptors[2:4] + descriptors[:2] + descriptors[4:]
        descriptors_config_r = [descriptors_r + binary_encode(config, mix_site)
                                for config in range(0, ORDERINGS[mix_site])]
        descriptors_config += descriptors_config_r

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
    Ef = Etot - sum(Etot,ox)
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
                DATA_DIR, 'stable_binary_oxides_garnet.csv'))
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

    potcars = ["pbe %s" % CONFIG['POTCAR'][el] for el in elements]

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
    all_entries = [e for e in GARNET_ENTRIES_UNIQUE + GARNET_CALC_ENTRIES
                   if set(e.composition).issubset(set(composition))]
    if not all_entries:
        raise ValueError("Incomplete")
    entry = prepare_entry(tot_e, species)

    phase_diagram = PhaseDiagram(all_entries + [entry] + unmix_entries)

    return phase_diagram.get_decomp_and_e_above_hull(entry)


def parse_composition(s, ctype):
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
        if ctype == "A":
            if abs(frac[0] - 0.5) > 0.01:
                raise ValueError("Bad composition on %s. "
                                 "Only 1:1 mixing allowed!" % ctype)
        elif ctype in ["C", "D"]:
            if not (abs(frac[0] - 1.0 / 3) < 0.01 or abs(
                    frac[1] - 1.0 / 3) < 0.01):
                raise ValueError("Bad composition on %s. "
                                 "Only 2:1 mixing allowed!" % ctype)
    try:
        for k in c.keys():
            k.oxi_state
            if k not in GARNET_ELS[ctype]:
                raise ValueError("%s is not a valid species for %s site."
                                 % (k, ctype))
    except AttributeError:
        raise ValueError("Oxidation states must be specified for all species!")

    return c
