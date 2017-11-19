import json
import re
import os

from monty.serialization import loadfn
from monty.json import jsanitize
from pymatgen import Specie, Composition
from flask import render_template, make_response
from flask.json import jsonify

from flask import request, Response

module_path = os.path.dirname(os.path.abspath(__file__))

from flask import Flask
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return make_response(render_template('index.html'))


def parse_composition(s):
    toks = s.strip().split()
    if len(toks) == 1:
        return Composition({toks[0].split(":")[0]: 1})
    else:
        return Composition({t.split(":")[0]: float(t.split(":")[1])
                            for t in toks})


@app.route('/query', methods=['GET'])
def query():
    error_message = None

    try:
        c_string = request.args.get("c_string")
        a_string = request.args.get("a_string")
        d_string = request.args.get("d_string")

        c_composition = parse_composition(c_string)
        a_composition = parse_composition(a_string)
        d_composition = parse_composition(d_string)

        charge = -2.0 * 12
        for k in c_composition.keys():
            charge += 3 * k.oxi_state * c_composition.get_atomic_fraction(k)
        for k in a_composition.keys():
            charge += 2 * k.oxi_state * a_composition.get_atomic_fraction(k)
        for k in d_composition.keys():
            charge += 3 * k.oxi_state * d_composition.get_atomic_fraction(k)

        if abs(charge) < 0.1:
            error_message = "Charge neutral!"
        else:
            error_message = "Not charge neutral! Total charge = %.0f" % charge
    except Exception as ex:
        error_message = str(ex)

    return make_response(render_template(
        'index.html', c_string=c_string, a_string=a_string, d_string=d_string,
        error_message=error_message)
    )


if __name__ == "__main__":
    app.run(debug=True)
