import os
import pandas as pd
import sys

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import serialize_sk as sk

from monty.serialization import loadfn

from pymatgen import MPRester, Specie, Composition
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.io.vasp.sets import _load_yaml_config

from flask import render_template, make_response, request, Response, Flask


app = Flask(__name__)

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

m = MPRester("xNebFpxTfLhTnnIH")


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
    mix_site = [site for site in sites if len(species[site]) == 2]
    if not mix_site:
        # unmixed type
        mix_site = 'c'
        input_spe = [el for site in ['c','c','a','d'] for el in species[site]]
    else:
        # mixed type
        mix_site = mix_site[0]
        sites.remove(mix_site)
        spes = species.copy()
        spes['%s_sorted' % mix_site] = sorted(
            spes[mix_site], key=lambda k: (spes[mix_site][k], k))
        input_spe = [el for site in ['%s_sorted' % mix_site] + sites
                     for el in spes[site]]

    descriptors = [(get_el_sp(spe).ionic_radius, get_el_sp(spe).X)
                   for spe in input_spe]
    descriptors = list(sum(descriptors, ()))
    descriptors_config = [descriptors + binary_encode(config, mix_site)
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

    # all_entries = m.get_entries_in_chemsys(elements=elements)

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
    all_entries = [e for e in GARNET_ENTRIES_UNIQUE \
                    if set(e.composition).issubset(set(composition))]
    if not all_entries:
        raise ValueError("Incomplete")
    entry = prepare_entry(tot_e, species)

    phase_diagram = PhaseDiagram(all_entries+[entry]+unmix_entries)

    return phase_diagram.get_e_above_hull(entry)


def model_load_single(model_type):
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


@app.route('/', methods=['GET'])
def index():
    return make_response(render_template('index.html'))


def parse_composition(s, ctype):
    toks = s.strip().split()
    if len(toks) == 1:
        c = Composition({toks[0].split(":")[0]: 1})
    else:
        c = Composition({t.split(":")[0]: float(t.split(":")[1])
                         for t in toks})
        if len(c) != 2:
            raise ValueError("Bad composition on %s." % ctype)
        frac = [c.get_atomic_fraction(k) for k in c.keys()]
        if ctype == "A" and abs(frac[0] - 0.5) > 0.01:
            raise ValueError("Bad composition on %s. Only 1:1 mixing allowed!" % ctype)
        elif not (abs(frac[0] - 1.0/3) < 0.01 or abs(frac[1] - 1.0/3) < 0.01):
            raise ValueError("Bad composition on %s. Only 2:1 mixing allowed!" % ctype)
    try:
        for k in c.keys():
            k.oxi_state
    except AttributeError:
        raise ValueError("Oxidation states must be specified for all species!")
    return c


@app.route('/query', methods=['GET'])
def query():
    try:
        c_string = request.args.get("c_string")
        a_string = request.args.get("a_string")
        d_string = request.args.get("d_string")

        c_composition = parse_composition(c_string, "C")
        a_composition = parse_composition(a_string, "A")
        d_composition = parse_composition(d_string, "D")

        charge = -2.0 * 12

        for k in c_composition.keys():
            charge += 3 * k.oxi_state * c_composition.get_atomic_fraction(k)
        for k in a_composition.keys():
            charge += 2 * k.oxi_state * a_composition.get_atomic_fraction(k)
        for k in d_composition.keys():
            charge += 3 * k.oxi_state * d_composition.get_atomic_fraction(k)

        if len(c_composition) > 1:
            mix_site = "c"
        elif len(a_composition) > 1:
            mix_site = "a"
        elif len(d_composition) > 1:
            mix_site = "d"
        else:
            mix_site = None

        species = {"a": a_composition, "d": d_composition, "c": c_composition}

        if abs(charge) < 0.1:
            model, scaler = model_load_single("ext_%s" % mix_site) \
                if mix_site else model_load_single("ext_c")
            inputs = get_descriptor_ext(species)
            form_e = get_form_e_ext(inputs, model, scaler)
            tot_e = get_tote(form_e, species)
            if mix_site:
                decompose_entries = get_decomposed_entries(species, model,
                                                           scaler)
                ehull_pred = get_ehull(tot_e=tot_e, species=species,
                                       unmix_entries=decompose_entries)
            else:
                ehull_pred = get_ehull(tot_e=tot_e, species=species)
            message = "<i>E<sub>f</sub></i> = %.3f eV/fu<br><i>E<sub>hull</sub></i> = %.0f meV/atom" % (form_e, ehull_pred * 100)
        else:
            message = "Not charge neutral! Total charge = %.0f" % charge
    except Exception as ex:
        message = str(ex)

    return make_response(render_template(
        'index.html', c_string=c_string, a_string=a_string, d_string=d_string,
        message=message)
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="""Basic web app for garnet deep neural network.""",
        epilog="Author: Shyue Ping Ong")

    parser.add_argument(
        "-d", "--debug", dest="debug", action="store_true",
        help="Whether to run in debug mode.")
    parser.add_argument(
        "-hh", "--host", dest="host", type=str, nargs="?",
        default='0.0.0.0',
        help="Host in which to run the server. Defaults to 0.0.0.0.")
    parser.add_argument(
        "-p", "--port", dest="port", type=int, nargs="?",
        default=5000,
        help="Port in which to run the server. Defaults to 5000.")

    args = parser.parse_args()
    app.run(debug=args.debug, host=args.host, port=args.port)
