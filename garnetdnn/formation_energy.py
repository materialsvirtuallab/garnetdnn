import os
from collections import Counter

import numpy as np
from monty.serialization import loadfn
from pymatgen import Composition
from pymatgen.core.periodic_table import get_el_sp

from garnetdnn.util import spe2form
from pymatgen import MPRester

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
BINARY_OXIDES_PATH = os.path.join(DATA_DIR, "binary_oxide_entries.json")
BINARY_OXDIES_ENTRIES = loadfn(BINARY_OXIDES_PATH)
STD_FORMULA = {'garnet': Composition("C3A2D3O12"),
               "perovskite": Composition("A2B2O6")}
SITES = {'garnet': ['c', 'a', 'd'],
         'perovskite': ['a', 'b']}  # use list to preserve order
SITE_INFO = {'garnet': {'c': {"num_atoms": 3, "max_ordering": 20, "cn": "VIII"},
                        'a': {"num_atoms": 2, "max_ordering": 7, "cn": "VI"},
                        'd': {"num_atoms": 3, "max_ordering": 18, "cn": "IV"}},
             'perovskite': {'a': {"num_atoms": 2, "max_ordering": 10, 'cn': "XII"},
                            'b': {"num_atoms": 2, "max_ordering": 10, 'cn': "VI"}}}

m = MPRester("VIqD4QUxH6wNpyc5")


def binary_encode(config, tot_configs):
    """
    Args:
        config(int): the index of the configuration
        tot_configs(int): total number of configurations
    """
    get_bin = lambda x: format(x, 'b')
    vb = get_bin(config)
    max_digit = len([i for i in (bin(tot_configs))[2:] if i.isdigit()])
    letter = [int(char) for char in vb]
    letter = [0] * (max_digit - len(letter)) + letter
    return letter


def _raw_input(input_spe, cn_specific, site_info):
    """
    inputs w/o configuration encoded
    """
    if cn_specific:
        descriptors = [(get_el_sp(spe).get_shannon_radius(cn=site_info[site]['cn']),
                        get_el_sp(spe).X)
                       for spe, site in input_spe]
    else:
        descriptors = [(get_el_sp(spe).ionic_radius, get_el_sp(spe).X) \
                       for spe, site in input_spe]
    return list(sum(descriptors, ()))

def get_descriptor(structure_type, species, cn_specific=True, unmix_expansion=None, config=None):
    """
    Prepare the inputs for model prediction.
    i.e. extract the
    [rC', XC', rC'', XC'',rA, XA, rD, XD, b1, b2, b3, b4, b5]
    inputs array for given species.

    Args:
        structure_type (str): 'garnet' or 'perovskite'
        species (dict): species in dictionary.
        cn_specific(bool): True if use cn specific ionic radius
        unmix_expansion(list): list the order of sites that reorgonize unmix data
            eg. for an unmix garnet Ca3Al2Ga3O12, to obtain the corresponding
            descriptors for extended c-mix/a-mix/d-mix model
            the unmix_expansion can be specified as
            ['c', 'c', 'a', 'd'],
            ['c', 'a', 'a', 'd'],
            ['c', 'a', 'd', 'd'],
            respectively

    Returns:
        inputs (list): numerical inputs of the input structure
    """

    site_info = SITE_INFO[structure_type]
    sites = SITES[structure_type]
    input_spe = [(spe, site) for site in sites
                 for spe in sorted(species[site],
                                   key=lambda x: species[site][x])]

    mix_site = [site for site in sites if len(species[site]) == 2]
    if not mix_site:
        if not unmix_expansion:
            return _raw_input(input_spe, cn_specific, site_info)
        else:
            sites = unmix_expansion
            mix_site = Counter(unmix_expansion).most_common(1)[0][0]
            input_spe = [(spe, site) for site in sites
                         for spe in sorted(species[site],
                                           key=lambda x: species[site][x])]
            max_ordering = site_info[mix_site]['max_ordering']
            descriptors = _raw_input(input_spe, cn_specific, site_info)
            descriptors_config = [descriptors + binary_encode(config, max_ordering)
                                  for config in range(0, max_ordering)]
            return descriptors_config


    else:
        mix_site = mix_site[0]
        if len(set(species[mix_site].values())) > 1:
            descriptors = _raw_input(input_spe, cn_specific, site_info)
            max_ordering = site_info[mix_site]['max_ordering']
            if config != None:
                descriptors_config = descriptors + binary_encode(config, max_ordering)
            else:
                descriptors_config = [descriptors + binary_encode(config, max_ordering)
                                      for config in range(0, max_ordering)]
            return descriptors_config

        else:
            mix_site = mix_site[0]
            max_ordering = site_info[mix_site]['max_ordering']
            input_spe = [(spe, site) for site in sites
                         for spe in sorted(species[site],
                                           key=lambda x: x.__str__())]
            descriptors = _raw_input(input_spe, cn_specific, site_info)
            input_spe_r = [(spe, site) for site in sites
                           for spe in sorted(species[site],
                                             key=lambda x: x.__str__(),
                                             reverse=True)]
            descriptors_r = _raw_input(input_spe_r, cn_specific, site_info)
            if config != None:
                descriptors_config = [descriptors + binary_encode(config, max_ordering)]
                descriptors_config_r = [descriptors_r + binary_encode(config, max_ordering)]
            else:
                descriptors_config = [descriptors + binary_encode(config, max_ordering)
                                      for config in range(0, max_ordering)]
                descriptors_config_r = [descriptors_r + binary_encode(config, max_ordering)
                                        for config in range(0, max_ordering)]
            return descriptors_config + descriptors_config_r


def get_form_e(descriptors, model, scaler, return_full=False):
    """
    Get formation energy from the given inputs.

    Args:
        descriptors (dict): input descriptors dict.
        model (keras.model): keras model object.
        scaler (keras.StandardScaler): keras StandardScaler object
        return_full (bool): if True return the full list of energies
                            instead of the lowest
    Returns:
        predicted_ef (float): the predicted formation Energy.
    """
    if len(np.array(descriptors).shape) == 1:
        descriptors = np.array(descriptors).reshape(1, -1)
    inputs_ext_scaled = scaler.transform(descriptors)
    if return_full:
        return model.predict(inputs_ext_scaled)
    else:
        form_e = min(model.predict(inputs_ext_scaled))[0]
        return form_e


def get_tote(structure_type, form_e, species, oxides_table_path, debug=False):
    # formula = spe2form(structure_type, species)
    # composition = Composition(formula)
    spe_dict = Counter({})
    for site in SITE_INFO[structure_type]:
        spe_dict += Counter({spe.__str__(): round(SITE_INFO[structure_type][site]['num_atoms'] \
                                                  * species[site][spe]) \
                             for spe in sorted(species[site], key=lambda x: species[site][x])})
    composition = Composition(spe_dict)
    tote = form_e

    for el, amt in composition.items():
        if debug:
            print(el)
        stable_ox_entry = None
        if el.symbol == 'O':
            continue
        if el.symbol in BINARY_OXDIES_ENTRIES:
            stable_ox_entry = BINARY_OXDIES_ENTRIES[el.symbol]

        if not stable_ox_entry:
            stable_ox_table = loadfn(oxides_table_path)
            stable_ox_id = stable_ox_table[el.__str__()]['mpid']
            stable_ox_entry = m.get_entry_by_material_id(
                stable_ox_id,
                property_data=['e_above_hull',
                               'formation_energy_per_atom'])
        min_e = stable_ox_entry.uncorrected_energy
        amt_ox = stable_ox_entry.composition[el.name]
        tote += (amt / amt_ox) * min_e
    return tote
