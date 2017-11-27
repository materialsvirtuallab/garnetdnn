from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
from pymatgen.core.periodic_table import get_el_sp
from app import *



class GarnetEhullPredTest(unittest.TestCase):
    def setUp(self):
        # self.unmix_species = {"c":{"Y3+":1},"a":{"Al3+":1},"d":{"Al3+":1}}

        self.amix_species = {'a': {get_el_sp('Ga3+'): 0.5, get_el_sp('In3+'): 0.5},
                             'c': {get_el_sp('Sm3+'): 1},
                             'd': {get_el_sp('Ga3+'): 1}}
        self.amix_des = [[0.76, 1.81, 0.94, 1.78, 1.0979999999999999, 1.17, 0.76, 1.81, 0, 1, 1],
                         [0.76, 1.81, 0.94, 1.78, 1.0979999999999999, 1.17, 0.76, 1.81, 1, 0, 0],
                         [0.76, 1.81, 0.94, 1.78, 1.0979999999999999, 1.17, 0.76, 1.81, 0, 0, 1],
                         [0.76, 1.81, 0.94, 1.78, 1.0979999999999999, 1.17, 0.76, 1.81, 0, 0, 0],
                         [0.76, 1.81, 0.94, 1.78, 1.0979999999999999, 1.17, 0.76, 1.81, 0, 1, 0],
                         [0.76, 1.81, 0.94, 1.78, 1.0979999999999999, 1.17, 0.76, 1.81, 1, 0, 1],
                         [0.76, 1.81, 0.94, 1.78, 1.0979999999999999, 1.17, 0.76, 1.81, 1, 1, 0]]
        self.amix_des.sort()
        self.amix_dft_form_e = -1.3097931912500016
        self.amix_dft_tote = -138.1004917975
        self.amix_dft_ehull = 0.0005

        self.cmix_species = {"a":{get_el_sp("Sc3+"):1},
                             "c":{get_el_sp("Mg2+"):1/3, get_el_sp("Zn2+"):2/3},
                             "d":{get_el_sp("Si4+"):1}}
        self.cmix_des = [[0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 0, 0, 0, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 0, 0, 0, 1],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 0, 0, 1, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 0, 0, 1, 1],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 0, 1, 0, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 0, 1, 0, 1],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 0, 1, 1, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 0, 1, 1, 1],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 1, 0, 0, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 1, 0, 0, 1],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 1, 0, 1, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 1, 0, 1, 1],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 1, 1, 0, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 1, 1, 0, 1],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 1, 1, 1, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 0, 1, 1, 1, 1],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 1, 0, 0, 0, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 1, 0, 0, 0, 1],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 1, 0, 0, 1, 0],
                         [0.86, 1.31, 0.88, 1.65, 0.885, 1.36, 0.54, 1.9, 1, 0, 0, 1, 1]]
        self.cmix_des.sort()
        self.cmix_dft_form_e = 1.9102350587499934
        self.cmix_dft_tote = -144.692096835
        self.cmix_dft_ehull = 0.130


        self.dmix_species = {"a":{get_el_sp("Al3+"):1},
                             "c":{get_el_sp("Sr2+"):1},
                             "d":{get_el_sp("Si4+"):2/3, get_el_sp("Sn4+"):1/3}}

        self.dmix_des = [[0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 0, 1, 0, 0],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 0, 0, 1, 0],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 0, 0, 0, 1],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 1, 1, 1, 1],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 1, 0, 1, 0],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 1, 0, 0, 0, 0],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 0, 1, 1, 0],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 1, 0, 0, 0],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 0, 1, 0, 1],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 0, 1, 1, 1],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 0, 0, 1, 1],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 1, 0, 1, 1],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 0, 0, 0, 0],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 1, 1, 0, 1],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 1, 1, 0, 0],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 1, 0, 0, 0, 1],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 1, 0, 0, 1],
                         [0.83, 1.96, 0.54, 1.9, 1.32, 0.95, 0.675, 1.61, 0, 1, 1, 1, 0]]
        self.dmix_des.sort()
        self.dmix_dft_form_e = -1.4759222533333514
        self.dmix_dft_tote = -141.47510287
        self.dmix_dft_ehull = 0.101

    def test_amix(self):
        model, scaler = _load_model_and_scaler('ext_a')
        des = get_descriptor_ext(self.amix_species)
        des.sort()
        self.assertEqual(des, self.amix_des)

        form_e = get_form_e_ext(des, model, scaler)
        self.assertTrue(abs(form_e - self.amix_dft_form_e) < 0.25)

        tot_e = get_tote(form_e, self.amix_species)
        self.assertTrue(abs((tot_e - self.amix_dft_tote) \
                             - (form_e - self.amix_dft_form_e))\
                                < 1e-2)

        decompose_entries = get_decomposed_entries(self.amix_species)
        decomp, ehull = get_ehull(tot_e=tot_e,
                          species=self.amix_species,
                          unmix_entries=decompose_entries)
        self.assertTrue(abs(ehull-self.amix_dft_ehull) < 1e-2 )

    def test_cmix(self):
        model, scaler = _load_model_and_scaler('ext_c')
        des = get_descriptor_ext(self.cmix_species)
        des.sort()
        self.assertEqual(des, self.cmix_des)

        form_e = get_form_e_ext(des, model, scaler)
        print(form_e)
        self.assertTrue(abs(form_e - self.cmix_dft_form_e) < 0.25)

        tot_e = get_tote(form_e, self.cmix_species)
        self.assertAlmostEqual(tot_e, self.cmix_dft_tote + (form_e - self.cmix_dft_form_e))

        decompose_entries = get_decomposed_entries(self.cmix_species)
        decomp, ehull = get_ehull(tot_e=tot_e,
                          species=self.cmix_species,
                          unmix_entries=decompose_entries)
        self.assertTrue(abs(ehull-self.cmix_dft_ehull) < 1e-2 )

    def test_dmix(self):
        model, scaler = _load_model_and_scaler('ext_d')
        des = get_descriptor_ext(self.dmix_species)
        des.sort()
        self.assertEqual(des, self.dmix_des)

        form_e = get_form_e_ext(des, model, scaler)
        self.assertTrue(abs(form_e-self.dmix_dft_form_e) < 0.25)

        tot_e = get_tote(form_e, self.dmix_species)
        self.assertAlmostEqual(tot_e, self.dmix_dft_tote + (form_e - self.dmix_dft_form_e))


        decompose_entries = get_decomposed_entries(self.dmix_species)
        decomp, ehull = get_ehull(tot_e=tot_e,
                          species=self.dmix_species,
                          unmix_entries=decompose_entries)
        self.assertTrue(abs(ehull-self.dmix_dft_ehull) < 1e-2 )


if __name__ == "__main__":
    unittest.main()
