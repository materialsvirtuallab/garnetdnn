from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import pickle
from predict_garnet_stability import *
from pymatgen.core import Specie
import shlex
import subprocess


class GarnetEhullPredTest(unittest.TestCase):
    def setUp(self):
        self.yag_inputs = "-c Y3+ 3 -a Al3+ 2 -d Al3+ 3"
        self.nszs_inputs = "-c Na+ 3 -a Sb5+ 1 Zr4+ 1 -d Si4+ 3"

        self.nszs_spe = {'d': {Specie("Si", 4): 1.0},
               'a': {Specie("Sb", 5): 1/2,
                     Specie("Zr", 4): 1/2},
               'c': {Specie("Na", 1): 1.0}}

        self.zzms_spe = {'d': {Specie("Si", 4): 1.0},
                    'a': {Specie("Mg", 2): 1.0},
                    'c': {Specie("Zn", 2): 2/3,
                        Specie("Zr", 4): 1/3}}

        self.yag_spe = {'d': {Specie("Al", 3): 1.0},
               'a': {Specie("Al", 3): 1.0},
               'c': {Specie("Y", 3): 1.0}}

        self.model = load_model(sorted(glob.glob("garnet_model*.h5"))[-1])
        with open(sorted(glob.glob("garnet_scaler*.pkl"))[-1],'rb') as f:
            self.scaler = pickle.load(f)




    def test_spe2form(self):
        self.assertEqual(spe2form(self.yag_spe), "Y3Al2Al3O12")
        self.assertTrue(spe2form(self.zzms_spe) in ["ZrZn2Mg2Si3O12","Zn2ZrMg2Si3O12"] )

    def test_yag(self):
        keys = ["X_a", "r_a", "X_c", "r_c", "X_d", "r_d"]
        form_e_pred = get_form_e(get_descriptor(self.yag_spe), self.model, self.scaler)
        inputs = np.array([get_descriptor(self.yag_spe)[k] for k in keys])
        form_e_model = self.model.predict(self.scaler.transform(inputs.reshape(1,-1)))
        self.assertEqual(form_e_pred, form_e_model)

        yag_form_e = -1.02695807749998 #eV/fu
        yag_tot_e = -162.8925220375 #eV/fu
        tote_pred = get_tote(form_e_pred, self.yag_spe)
        self.assertEqual(yag_form_e-yag_tot_e, form_e_pred-tote_pred)


        command = "python predict_garnet_stability.py %s"%self.yag_inputs
        args = shlex.split(command)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE)
        output = proc.stdout.read().decode("utf-8")

        output = output.split(" ")
        form_e_out, ehull, tote_out = output[1], output[4], output[7]
        self.assertEqual([form_e_out,ehull,tote_out],
                         ["%0.03f"%i for i in [form_e_pred, 0.006491071227600287, tote_pred]])

    def test_nszs(self):
        keys = ["X_a", "r_a", "X_c", "r_c", "X_d", "r_d"]

        # Test the get_form_e function
        form_e_pred = get_form_e(get_descriptor(self.nszs_spe), self.model, self.scaler)
        inputs = np.array([get_descriptor(self.nszs_spe)[k] for k in keys])
        form_e_model = self.model.predict(self.scaler.transform(inputs.reshape(1,-1)))
        self.assertEqual(form_e_pred, form_e_model)
        self.assertEqual(form_e_pred, form_e_model)

        nszs_form_e = -3.9525266399999985 # eV/fu
        nszs_tot_e = -141.33169747 # eV/fu
        # Test the get_tote funciton
        tote_pred = get_tote(form_e_pred, self.nszs_spe)
        self.assertEqual(nszs_form_e - nszs_tot_e, form_e_model - tote_pred)

        command = "python predict_garnet_stability.py %s"%self.nszs_inputs
        args = shlex.split(command)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE)
        output = proc.stdout.read().decode("utf-8")

        output = output.split(" ")
        form_e_out, ehull, tote_out = output[1], output[4], output[7]
        # test the output of the script
        self.assertEqual([form_e_out,ehull,tote_out],
                         ["%0.03f"%i for i in [form_e_pred, 0.076, tote_pred]])




if __name__ == "__main__":
    unittest.main()