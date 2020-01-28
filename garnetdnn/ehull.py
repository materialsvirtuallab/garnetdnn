import os
from pymatgen import Structure
from monty.serialization import loadfn
from pymatgen import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher

from pymatgen.analysis.structure_matcher import ElementComparator
from pymatgen.core import Composition
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.entries.computed_entries import ComputedEntry

from pymatgen.io.vasp.sets import _load_yaml_config

from garnetdnn.formation_energy import get_descriptor, get_form_e, get_tote
from garnetdnn.util import load_model_and_scaler, spe2form
import itertools

CONFIG = _load_yaml_config("MPRelaxSet")
LDAUU = CONFIG["INCAR"]['LDAUU']['O']
m = MPRester("xNebFpxTfLhTnnIH")

STD_FORMULA = {'garnet': Composition("C3A2D3O12"),
               "perovskite": Composition("A2B2O6")}
SITES = {'garnet': ['c', 'a', 'd'],
         'perovskite': ['a', 'b']}  # use list to preserve order
SITE_INFO = {'garnet': {'c': {"num_atoms": 3, "max_ordering": 20, "cn": "VIII"},
                        'a': {"num_atoms": 2, "max_ordering": 7, "cn": "VI"},
                        'd': {"num_atoms": 3, "max_ordering": 18, "cn": "IV"}},
             'perovskite': {'a': {"num_atoms": 2, "max_ordering": 10, 'cn': "XII"},
                            'b': {"num_atoms": 2, "max_ordering": 10, 'cn': "VI"}}}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
GARNET_CALC_ENTRIES_PATH = os.path.join(DATA_DIR,
                                        "garnet/garnet_calc_entries_dict.json")
GARNET_CALC_ENTRIES = loadfn(GARNET_CALC_ENTRIES_PATH)
PEROVSKITE_CALC_ENTRIES_PATH = os.path.join(DATA_DIR,
                                            "perovskite/perov_calc_entries_dict.json")
PEROVSKITE_CALC_ENTRIES = loadfn(PEROVSKITE_CALC_ENTRIES_PATH)
CALC_ENTRIES = {'garnet': GARNET_CALC_ENTRIES,
                'perovskite': PEROVSKITE_CALC_ENTRIES}
GARNET_EHULL_ENTRIES_PATH = os.path.join(DATA_DIR,
                                         "garnet/garnet_ehull_entries_dict.json")
GARNET_EHULL_ENTRIES = loadfn(GARNET_EHULL_ENTRIES_PATH)
PEROVSKITE_EHULL_ENTRIES_PATH = os.path.join(DATA_DIR,
                                             "perovskite/perov_ehull_entries_dict.json")
PEROVSKITE_EHULL_ENTRIES = loadfn(PEROVSKITE_EHULL_ENTRIES_PATH)
EHULL_ENTRIES = {'garnet': GARNET_EHULL_ENTRIES,
                 'perovskite': PEROVSKITE_EHULL_ENTRIES}
MATCHER = None
PROTO = None


def get_entries_in_chemsy(entries_dict, elements):
    """
    Args:
        entries_dict(dict): hashtable of entries pool
                    The key of the dict should be
                    "-".join(sorted(elements))
        elements(list): list of elements included in
                    the chemical space
                    eg. ['Ca', 'O']
    Returns:
        entries(list):
                    list of entries in the chemical
                    space from provided entries

    """
    entries = []
    for i in range(len(elements)):
        for els in itertools.combinations(elements, i + 1):
            key = '-'.join(sorted(els))
            if key not in entries_dict:
                continue
            entries.extend(entries_dict[key])
    return entries


def get_decomposed_entries(structure_type, species):
    """
    Get decomposed entries for mix types
    Args:
        structure_type(str): "garnet" or "perovskite"
        species (dict): species in dictionary.
        structure_type(str): garnet or perovskite

    Returns:
        decompose entries(list):
            list of entries prepared from unmix
            garnets/perovskite decomposed from input mix
            garnet/perovskite
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
    model, scaler = load_model_and_scaler(structure_type, "unmix")
    std_formula = STD_FORMULA[structure_type]
    for unmix_species in decomposed(species):
        charge = sum([spe.oxi_state * amt * SITE_INFO[structure_type][site]["num_atoms"]
                      for site in SITE_INFO[structure_type].keys()
                      for spe, amt in unmix_species[site].items()])
        if not abs(charge - 2 * std_formula['O']) < 0.1:
            continue

        formula = spe2form(structure_type, unmix_species)
        composition = Composition(formula)
        elements = [el.name for el in composition]
        chemsy = '-'.join(sorted(elements))
        calc_entries = []
        if CALC_ENTRIES[structure_type].get(chemsy):
            calc_entries = [entry for entry in CALC_ENTRIES[structure_type][chemsy] if \
                            entry.name == Composition(formula).reduced_formula]
        else:
            pass
        if calc_entries:
            decompose_entries.extend(calc_entries)

        else:
            cn_specific = True if structure_type == 'garnet' else False
            descriptors = get_descriptor(structure_type, unmix_species,
                                         cn_specific=cn_specific)
            form_e = get_form_e(descriptors, model, scaler)
            # tot_e = get_tote(form_e * std_formula.num_atoms, unmix_species)
            tot_e = get_tote(structure_type, form_e * std_formula.num_atoms, unmix_species)
            entry = prepare_entry(structure_type, tot_e, unmix_species)
            compat = MaterialsProjectCompatibility()
            entry = compat.process_entry(entry)
            decompose_entries.append(entry)

    return decompose_entries


def prepare_entry(structure_type, tot_e, species):
    """
    Prepare entries from total energy and species.

    Args:
        structure_type(str): "garnet" or "perovskite"
        tot_e (float): total energy in eV/f.u.
        species (dict): species in dictionary.

    Returns:
        ce (ComputedEntry)
    """

    formula = spe2form(structure_type, species)
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
    ce = compat.process_entry(ce)  # Correction added

    return ce


def filter_entries(structure_type, all_entries, species, return_removed=False):
    """
    Filter out the entry with exact same structure as queried entry among the
    entries in queried chemical space obtained from Materials Project.

    Used Pymatgen.analysis.structure_matcher.fit_anonymous to match the prototype
    of give structures.

    Args:
        structure_type(str): "garnet" or "perovskite"
         all_entries (list): entries in queried chemical space obtained from Materials Project
         composition (Composition): composition of queried entry
         return_removed (bool): If True, return the filtered entries

    Returns:
        filtered_entries (list)
    """
    global MATCHER, PROTO
    if not MATCHER:
        MATCHER = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True,
                                   comparator=ElementComparator())

    if not PROTO:
        garnet_proto_path = os.path.join(DATA_DIR, "garnet/proto_mp-3050.cif")
        perov_proto_path = os.path.join(DATA_DIR, "perovskite/proto_mp-4019.cif")
        PROTO = {"garnet": Structure.from_file(garnet_proto_path).get_primitive_structure(),
                 "perovskite": Structure.from_file(perov_proto_path).get_primitive_structure()}
        a_sites = [60, 61, 62, 63, 64, 65, 66, 67]
        for site_ind in a_sites:
            PROTO['garnet'].replace(site_ind, {"Ga": 1})

    P = PROTO[structure_type].copy()
    if structure_type == 'garnet:':
        P.replace_species({"Y": species['c'], "Ga": species['a'], "Al": species['d']})
    elif structure_type == 'perovskite':
        P.replace_species({"Ca": species['a'], "Ti": species['b']})
    composition = Composition(spe2form(structure_type, species))
    if not return_removed:
        return [e for e in all_entries \
                if e.name != composition.reduced_formula \
                and not MATCHER.fit(e.structure, P)]
    else:
        removed = [e for e in all_entries \
                   if e.name == composition.reduced_formula \
                   and MATCHER.fit(e.structure, P)]
        return removed, [e for e in all_entries if e not in removed]


def get_ehull(structure_type, tot_e, species,
              unmix_entries=None, all_entries=None,
              debug=False, from_mp=False):
    """
    Get Ehull predicted under given total energy and species. The composition
    can be either given by the species dict(for garnet only) or a formula.

    Args:
        structure_type(str): "garnet" or "perovskite"
        tot_e (float): total energy, the unit is in accordance with given
            composition.
        species (dict): species in dictionary.
        unmix_entries (list): additional list of unmix entries.
        all_entries(list): Manually supply the entries whithin the chemical space
        debug(bool): Whether or not to run it in debug mode. (For test only)
        from_mp(bool): Whether or not to query entries from MP (would take long)

    Returns:
        ehull (float): energy above hull.
    """

    formula = spe2form(structure_type, species)
    composition = Composition(formula)
    elements = [i.name for i in composition.elements]
    unmix_entries = [] if unmix_entries is None else unmix_entries
    if not all_entries:
        if from_mp:
            all_entries = m.get_entries_in_chemsys([el.name for el in composition],
                                                   inc_structure=True)
        else:
            entries_dict = EHULL_ENTRIES[structure_type]
            all_entries = get_entries_in_chemsy(entries_dict, elements)

    all_entries = filter_entries(structure_type, all_entries, species)

    # For unmix: no need to find calc entries, for mix,
    # calc entries were provided through unmix_entries
    # calc_entries_dict = CALC_ENTRIES[structure_type]
    # all_calc_entries = get_entries_in_chemsy(calc_entries_dict, elements)
    # compat = MaterialsProjectCompatibility()
    # all_calc_entries = compat.process_entries(all_calc_entries)
    # if all_calc_entries:
    #     all_entries = all_entries + all_calc_entries

    if not all_entries:
        raise ValueError("Incomplete")
    entry = prepare_entry(structure_type, tot_e, species)
    if debug:
        return entry, all_entries

    phase_diagram = PhaseDiagram(all_entries + [entry] + unmix_entries)

    return phase_diagram.get_decomp_and_e_above_hull(entry)
