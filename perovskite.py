import os
import pandas as pd
import re
import pickle
import keras
import tensorflow as tf
from keras.models import load_model

from monty.serialization import loadfn

from pymatgen import MPRester, Composition
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.entries.computed_entries import ComputedStructureEntry, ComputedEntry
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.io.vasp.sets import _load_yaml_config

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/perovskite")
CONFIG = _load_yaml_config("MPRelaxSet")
LDAUU = CONFIG["INCAR"]['LDAUU']['O']

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/perovskite")
BINARY_OXIDES_PATH = os.path.join(DATA_DIR, "binary_oxide_entries.json")
BINARY_OXDIES_ENTRIES = loadfn(BINARY_OXIDES_PATH)

PEROVSKITE_CALC_ENTRIES_PATH = os.path.join(DATA_DIR, "perov_calc_entries.json")
PEROVSKITE_CALC_ENTRIES = loadfn(PEROVSKITE_CALC_ENTRIES_PATH)

PEROVSKITE_ELS = {
    'A': [get_el_sp(i) for i in
          ['Al3+', 'Ba2+', 'Bi3+', 'Ca2+', 'Cd2+', 'Ce3+', 'Ce4+', 'Dy3+', 'Er3+',
           'Gd3+', 'Ho3+', 'La3+', 'Mg2+', 'Mn2+', 'Nd3+', 'Ni2+', 'Pb2+', 'Pd2+',
           'Pr3+', 'Pt2+', 'Rh3+', 'Sc3+', 'Sm3+', 'Sn4+', 'Sr2+', 'Tb3+', 'Tl3+',
           'Tm3+', 'Y3+', 'Zn2+'
           ]],

    'B': [get_el_sp(i) for i in
          ["Al3+", "Au3+", "Bi3+", "Ce3+", "Ce4+", "Co2+", "Co3+", "Cr3+", "Cu2+",
           "Dy3+", "Er3+", "Eu3+", "Fe2+", "Fe3+", "Ga3+", "Gd3+", "Ge4+", "Hf4+",
           "Ho3+", "In3+", "Ir4+", "La3+", "Lu3+", "Mg2+", "Mn2+", "Mn4+", "Mo4+",
           "Nd3+", "Ni2+", "Os4+", "Pb4+", "Pd4+", "Pr3+", "Pt4+", "Re4+", "Rh3+",
           "Ru4+", "Sc3+", "Si4+", "Sm3+", "Sn4+", "Ta5+", "Tb3+", "Tc4+", "Ti4+",
           "Tl3+", "Tm3+", "V5+", "W4+", "Y3+", "Zn2+", "Zr4+"]]
}

MODELS = {}
SITE_OCCU = {'a': 2, 'b': 2}
m = MPRester("xNebFpxTfLhTnnIH")


def lazy_load_model_and_scaler(model_type):
    """
    Load model and scaler for Ef prediction.

    Args:
        model_type (str): type of models
            ext_a : Extended model trained on unmix+amix
            ext_b : Extended model trained on unmix+bmix

    Returns:
        model (keras.model)
        scaler(keras.StandardScaler)
    """
    keras.backend.clear_session()
    if model_type not in MODELS:
        model = load_model(os.path.join(MODEL_DIR,
                                        "model_ext_%s.h5" % model_type))
        with open(os.path.join(MODEL_DIR,
                               "scaler_ext_%s.pkl" % model_type), "rb") as f:
            scaler = pickle.load(f)
        MODELS[model_type] = model, scaler
        return model, scaler
    return MODELS[model_type]


def load_model_and_scaler(model_type):
    """
    Load model and scaler for Ef prediction.

    Args:
        model_type (str): type of models
            ext_a : Extended model trained on unmix+amix
            ext_b : Extended model trained on unmix+bmix

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


def html_formula(f):
    return re.sub(r"([\d.]+)", r"<sub>\1</sub>", f)


def binary_encode(config):
    """
    This transferes the config index into binary
    coding
    A-mix: max_len = 4
    B-mix: max_len = 4
    :param config: int, config index

    :return: list of binary encoding of config
    number
    eg., config = 1, return [0, 0, 0, 1]
    """
    max_len = 4
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
    model, scaler, graph = load_model_and_scaler("a")
    for unmix_species in decomposed(species):
        charge = sum([spe.oxi_state * amt * SITE_OCCU[site]
                      for site in ['a', 'b']
                      for spe, amt in unmix_species[site].items()])
        if not abs(charge - 2 * 6) < 0.1:
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
    standard perovskite formula. (A2B2O6)

    Args:
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
    sites = ['a', 'b']

    spe_list = [spe.name + str(round(SITE_OCCU[site] * species[site][spe]))
                for site in sites for
                spe in sorted(species[site])]
    formula = "".join(spe_list)
    formula = formula.replace("1", "") + 'O6'
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
        'a': 10,
        'b': 10
    }

    sites = ['a', 'b']
    mix_site = [site for site in sites if len(species[site]) == 2]
    if not mix_site:
        # use ext-a model for unmixed type
        mix_site = 'a'
        input_spe = [k for site in ['a', 'a', 'b'] \
                     for k in species[site]]
    else:
        # mixed type
        mix_site = mix_site[0]
        sites.remove(mix_site)
        spes = species.copy()
        # sort the mix site on the order of increasing occupancy
        spes['%s_sorted' % mix_site] = sorted(
            spes[mix_site],
            key=lambda k: (spes[mix_site][k], k))
        input_spe = [spe for site in ['a', 'b']
                     for spe in spes[site]]

    descriptors = [(get_el_sp(spe).ionic_radius, get_el_sp(spe).X) \
                   for spe in input_spe]
    des1 = list(sum(descriptors, ()))
    if mix_site == 'a':
        des2 = des1[2:4] + des1[:2] + des1[4:]
    elif mix_site == 'b':
        des2 = des1[:2] + des1[4:] + des1[2:4]

    descriptors_config = [des1 + binary_encode(config)
                          for config in range(0, ORDERINGS[mix_site])]
    descriptors_config_r = [des2 + binary_encode(config)
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
    calculate formation energy for single perovskites
    :param entry_dict: dict
        vasprun.get_computed_entry().as_dict()
    :param factor: str 'per_fu' or 'per_atom'
        unit of formation energy
    :param energy: float
        total energy of the entry, if not given ,use entry.uncorrected_energy
    :param spes: list
        list of speces ['Ba2+','Ti4+']

    :return: float
        formation energy from binary oxides
    """

    formula = spe2form(species)
    composition = Composition(formula)
    tote = form_e
    species_list = [spe for site in ['a', 'b'] for spe in species[site]]
    for el in species_list:
        # print("finding stable bio entry for %s"%el)
        stable_bio_entry = None
        if el == 'O2-':
            continue
        el_name = el.element.name
        amt = composition[el_name]

        if el.__str__() in BINARY_OXDIES_ENTRIES:
            stable_bio_entry = BINARY_OXDIES_ENTRIES[el.__str__()]

        if not stable_bio_entry:
            stable_bio_df = pd.DataFrame.from_csv(os.path.join(
                DATA_DIR, 'stable_binary_oxides_perov.csv'), index_col=None)
            stable_bio_id = stable_bio_df.loc[
                lambda df: df.Ion == el.__str__()]['mpid'].tolist()[0]
            stable_bio_entry = m.get_entry_by_material_id(
                stable_bio_id, property_data=['e_above_hull',
                                              'formation_energy_per_atom'])

        min_e = stable_bio_entry.uncorrected_energy
        amt_bio = stable_bio_entry.composition[el_name]
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
    all_entries = m.get_entries_in_chemsys([el.name for el in composition.elements])
    all_calc_entries = [e for e in PEROVSKITE_CALC_ENTRIES
                        if set(e.composition).issubset(set(composition))]
    if all_calc_entries:
        all_entries = all_entries + all_calc_entries
    compat = MaterialsProjectCompatibility()
    all_entries = compat.process_entries(all_entries)
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

        if abs(frac[0] - 0.5) > 0.01:
            raise ValueError("Bad composition on %s. "
                             "Only 1:1 mixing allowed!" % ctype)
    try:
        for k in c.keys():
            k.oxi_state
            if k not in PEROVSKITE_ELS[ctype]:
                raise ValueError("%s is not a valid species for %s site."
                                 % (k, ctype))
    except AttributeError:
        raise ValueError("Oxidation states must be specified for all species!")

    return c
