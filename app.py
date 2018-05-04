import re
from flask import render_template, make_response, request, Flask
import garnet
import perovskite
import tensorflow as tf

app = Flask(__name__)


def html_formula(f):
    return re.sub(r"([\d.]+)", r"<sub>\1</sub>", f)


@app.route('/', methods=['GET'])
def index():
    return make_response(render_template('index.html'))


@app.route('/query', methods=['GET'])
def query():
    try:
        c_string = request.args.get("c_string")
        a_string = request.args.get("a_string")
        d_string = request.args.get("d_string")
        formula = ""

        c_composition = garnet.parse_composition(c_string, "C")
        a_composition = garnet.parse_composition(a_string, "A")
        d_composition = garnet.parse_composition(d_string, "D")

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
            with tf.Session() as sess:
                model, scaler, graph = garnet.load_model_and_scaler(mix_site) if mix_site \
                    else garnet.load_model_and_scaler("c")
                inputs = garnet.get_descriptor_ext(species)

                with graph.as_default():
                    form_e = garnet.get_form_e_ext(inputs, model, scaler)
                tot_e = garnet.get_tote(form_e, species)
                if mix_site:
                    decompose_entries = garnet.get_decomposed_entries(species)
                    decomp, ehull = garnet.get_ehull(tot_e=tot_e, species=species,
                                                     unmix_entries=decompose_entries)
                else:
                    decomp, ehull = garnet.get_ehull(tot_e=tot_e, species=species)
                formula = garnet.spe2form(species)
                message = ["<i>E<sub>f</sub></i> = %.3f eV/fu" % form_e,
                           "<i>E<sub>hull</sub></i> = %.0f meV/atom" %
                           (ehull * 1000)]
                if ehull > 0:
                    reaction = []
                    for k, v in decomp.items():
                        comp = k.composition
                        comp, f = comp.get_reduced_composition_and_factor()
                        reaction.append(
                            '%.3f <a href="https://www.materialsproject.org/materials/%s">%s</a>'
                            % (v * f / comp.num_atoms * 20, k.entry_id,
                               html_formula(comp.reduced_formula)))
                    message.append("Decomposition: " + " + ".join(reaction))

                message = "<br>".join(message)
        else:
            message = "Not charge neutral! Total charge = %.0f" % charge
    except Exception as ex:
        message = str(ex)

    return make_response(render_template(
        'index.html',
        c_string=c_string, a_string=a_string, d_string=d_string,
        formula=html_formula(formula),
        message=message
    )
    )


@app.route('/perovskite', methods=['GET'])
def perovskite_index():
    return make_response(render_template('index_perov.html'))


@app.route('/perovskite_query')
def perovskite_query():
    try:
        a_string = request.args.get("a_string")
        b_string = request.args.get("b_string")
        formula = ""

        a_composition = perovskite.parse_composition(a_string, "A")
        b_composition = perovskite.parse_composition(b_string, "B")

        charge = -2.0 * 6

        for k in a_composition.keys():
            charge += 2 * k.oxi_state * a_composition.get_atomic_fraction(k)
        for k in b_composition.keys():
            charge += 2 * k.oxi_state * b_composition.get_atomic_fraction(k)

        if len(a_composition) > 1:
            mix_site = "a"
        elif len(b_composition) > 1:
            mix_site = "b"
        else:
            mix_site = None

        species = {"a": a_composition, "b": b_composition}

        if abs(charge) < 0.1:
            with tf.Session() as sess:
                model, scaler, graph = perovskite.load_model_and_scaler(mix_site) if mix_site \
                    else perovskite.load_model_and_scaler("a")
                inputs = perovskite.get_descriptor_ext(species)
                with graph.as_default():
                    form_e = perovskite.get_form_e_ext(inputs, model, scaler)
                # form_e predicted from model is always in /atom
                # the get_tote func always returns the tote with in /standard fu
                # which is A2B2O6, 10 atoms
                tot_e = perovskite.get_tote(form_e * 10, species)
                if mix_site:
                    decompose_entries = perovskite.get_decomposed_entries(species)
                    decomp, ehull = perovskite.get_ehull(tot_e=tot_e, species=species,
                                                         unmix_entries=decompose_entries)
                else:
                    decomp, ehull = perovskite.get_ehull(tot_e=tot_e, species=species)
                formula = perovskite.spe2form(species)
                message = ["<i>E<sub>f</sub></i> = %.3f eV/fu" % (form_e * 10),
                           "<i>E<sub>hull</sub></i> = %.0f meV/atom" %
                           (ehull * 1000)]
                if ehull > 0:
                    reaction = []
                    for k, v in decomp.items():
                        comp = k.composition
                        rcomp, f = comp.get_reduced_composition_and_factor()
                        reaction.append(
                            '%.3f <a href="https://www.materialsproject.org/materials/%s">%s</a>'
                            % (v * f / comp.num_atoms * 10, k.entry_id,
                               html_formula(comp.reduced_formula)))
                    message.append("Decomposition: " + " + ".join(reaction))

                message = "<br>".join(message)
        else:
            message = "Not charge neutral! Total charge = %.0f" % charge
    except Exception as ex:
        message = str(ex)

    return make_response(render_template(
        'index_perov.html',
        a_string=a_string, b_string=b_string,
        formula=html_formula(formula),
        message=message
    )
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""Basic web app for garnet deep neural network.""",
        epilog="Authors: Weike Ye, Chi Chen, Zhenbin Wang, Iek-Heng Chu, Shyue Ping Ong")

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


if __name__ == "__main__":
    main()

    #
    # import cProfile, pstats, os
    #
    # cProfile.run('profile()', 'stats')
    # p = pstats.Stats('stats')
    # p.sort_stats('cumulative').print_stats(30)
    # os.remove('stats')
