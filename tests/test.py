from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
from pymatgen.core.periodic_table import get_el_sp
from app import *

class GarnetEhullPredTest(unittest.TestCase):
    def setUp(self):
        # self.unmix_species = {"c":{"Y3+":1},"a":{"Al3+":1},"d":{"Al3+":1}}

        self.amix_species = {'a': {get_el_sp('Mg2+'): 0.5, get_el_sp('Zr4+'): 0.5},
                             'c': {get_el_sp('Sr2+'): 1},
                             'd': {get_el_sp('Ge4+'): 1}}
        self.amix_des = [[0.86, 1.33, 0.86, 1.31, 1.32, 0.95, 0.67, 2.01, 0, 1, 0],
                         [0.86, 1.31, 0.86, 1.33, 1.32, 0.95, 0.67, 2.01, 0, 1, 0],
                         [0.86, 1.33, 0.86, 1.31, 1.32, 0.95, 0.67, 2.01, 1, 0, 1],
                         [0.86, 1.31, 0.86, 1.33, 1.32, 0.95, 0.67, 2.01, 1, 0, 1],
                         [0.86, 1.33, 0.86, 1.31, 1.32, 0.95, 0.67, 2.01, 0, 0, 0],
                         [0.86, 1.31, 0.86, 1.33, 1.32, 0.95, 0.67, 2.01, 0, 0, 0],
                         [0.86, 1.33, 0.86, 1.31, 1.32, 0.95, 0.67, 2.01, 0, 1, 1],
                         [0.86, 1.31, 0.86, 1.33, 1.32, 0.95, 0.67, 2.01, 0, 1, 1],
                         [0.86, 1.33, 0.86, 1.31, 1.32, 0.95, 0.67, 2.01, 0, 0, 1],
                         [0.86, 1.31, 0.86, 1.33, 1.32, 0.95, 0.67, 2.01, 0, 0, 1],
                         [0.86, 1.33, 0.86, 1.31, 1.32, 0.95, 0.67, 2.01, 1, 0, 0],
                         [0.86, 1.31, 0.86, 1.33, 1.32, 0.95, 0.67, 2.01, 1, 0, 0],
                         [0.86, 1.33, 0.86, 1.31, 1.32, 0.95, 0.67, 2.01, 1, 1, 0],
                         [0.86, 1.31, 0.86, 1.33, 1.32, 0.95, 0.67, 2.01, 1, 1, 0]]
        self.amix_des.sort()
        self.amix_dft_form_e = -4.0575368925
        self.amix_dft_tote = -138.9334240975
        self.amix_dft_ehull = 0.010064063041652638
        self.amix_dnn_form_e = round(-4.12900925, 7)

        self.cmix_species = {"a":{get_el_sp("Ge4+"):1},
                             "c":{get_el_sp("Lu3+"):1/3, get_el_sp("Mg2+"):2/3},
                             "d":{get_el_sp("Al3+"):1}}
        self.cmix_des = [[1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 1, 0, 0, 1, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 0, 0, 0, 0],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 0, 0, 0, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 1, 0, 0, 0],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 1, 1, 1, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 1, 1, 0, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 1, 1, 0, 0],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 0, 0, 1, 0],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 0, 1, 1, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 0, 1, 0, 0],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 0, 1, 1, 0],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 0, 0, 1, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 1, 0, 0, 0, 0],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 1, 0, 0, 1, 0],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 1, 1, 1, 0],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 1, 0, 0, 0, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 0, 1, 0, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 1, 0, 1, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 1, 0, 0, 1],
                         [1.001, 1.27, 0.86, 1.31, 0.67, 2.01, 0.675, 1.61, 0, 1, 0, 1, 0]]
        self.cmix_des.sort()
        self.cmix_dft_form_e = -0.08208461562500347
        self.cmix_dft_tote = -140.0701468475
        self.cmix_dft_ehull = 0.062090770312499544
        self.cmix_dnn_form_e = round(-0.06310445, 7)


        self.dmix_species = {"a":{get_el_sp("Sb5+"):1},
                             "c":{get_el_sp("Na+"):1},
                             "d":{get_el_sp("Ge4+"):2/3, get_el_sp("Ga3+"):1/3}}

        self.dmix_des = [[0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 1, 1, 0, 0],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 1, 0, 0, 0, 1],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 0, 1, 0, 0],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 0, 0, 0, 0],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 1, 1, 0, 1],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 0, 0, 1, 0],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 1, 0, 0, 0, 0],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 1, 1, 1, 0],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 0, 1, 1, 0],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 0, 1, 1, 1],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 0, 0, 1, 1],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 1, 1, 1, 1],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 0, 0, 0, 1],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 1, 0, 0, 0],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 1, 0, 1, 1],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 0, 1, 0, 1],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 1, 0, 0, 1],
                         [0.76, 1.81, 0.67, 2.01, 1.16, 0.93, 0.76, 2.05, 0, 1, 0, 1, 0]]
        self.dmix_des.sort()
        self.dmix_dft_form_e = -5.380956327499993
        self.dmix_dft_tote = -117.8037579025
        self.dmix_dft_ehull = 0.011935839152778627
        self.dmix_dnn_form_e = round(-4.88453293, 7)


    def test_amix(self):
        model, scaler = load_model_and_scaler('a')
        des = get_descriptor_ext(self.amix_species)
        des.sort()
        self.assertEqual(des, self.amix_des)

        form_e = get_form_e_ext(des, model, scaler)
        self.assertTrue(abs(form_e-self.amix_dnn_form_e)<1e-5)
        # self.assertEqual(form_e, self.amix_dnn_form_e)

        tot_e = get_tote(form_e, self.amix_species)
        self.assertTrue(abs((tot_e - self.amix_dft_tote) \
                             - (form_e - self.amix_dft_form_e))\
                                < 1e-2)

        decompose_entries = get_decomposed_entries(self.amix_species)
        decomp, ehull = get_ehull(tot_e=tot_e,
                          species=self.amix_species,
                          unmix_entries=decompose_entries)
        self.assertTrue(abs(ehull - self.amix_dft_ehull) < abs(form_e - self.amix_dft_form_e))

    def test_cmix(self):
        model, scaler = load_model_and_scaler('c')
        des = get_descriptor_ext(self.cmix_species)
        des.sort()
        self.assertEqual(des, self.cmix_des)

        form_e = get_form_e_ext(des, model, scaler)
        print(form_e)
        self.assertTrue(abs(form_e - self.cmix_dnn_form_e) < 1e-5)

        tot_e = get_tote(form_e, self.cmix_species)
        self.assertAlmostEqual(tot_e, self.cmix_dft_tote + (form_e - self.cmix_dft_form_e))

        decompose_entries = get_decomposed_entries(self.cmix_species)
        decomp, ehull = get_ehull(tot_e=tot_e,
                          species=self.cmix_species,
                          unmix_entries=decompose_entries)
        self.assertTrue(abs(ehull - self.cmix_dft_ehull) < abs(form_e - self.cmix_dft_form_e))

    def test_dmix(self):
        model, scaler = load_model_and_scaler('d')
        des = get_descriptor_ext(self.dmix_species)
        des.sort()
        self.assertEqual(des, self.dmix_des)

        form_e = get_form_e_ext(des, model, scaler)
        self.assertTrue(abs(form_e - self.dmix_dnn_form_e) < 1e-5)

        tot_e = get_tote(form_e, self.dmix_species)
        self.assertAlmostEqual(tot_e, self.dmix_dft_tote + (form_e - self.dmix_dft_form_e))


        decompose_entries = get_decomposed_entries(self.dmix_species)
        decomp, ehull = get_ehull(tot_e=tot_e,
                          species=self.dmix_species,
                          unmix_entries=decompose_entries)
        self.assertTrue(abs(ehull-self.dmix_dft_ehull) < abs(form_e-self.dmix_dft_form_e))


if __name__ == "__main__":
    unittest.main()
