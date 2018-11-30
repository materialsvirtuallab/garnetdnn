import dash
import dash_core_components as dcc
import dash_html_components as html
import re

import os
from flask import render_template, make_response, request, Flask
import tensorflow as tf
from garnetdnn.ehull import get_decomposed_entries, get_ehull
from garnetdnn.formation_energy import get_descriptor, get_form_e, get_tote
from garnetdnn.util import load_model_and_scaler, spe2form, html_formula, parse_composition
from collections import OrderedDict
import time

MAX_CACHE = 500
app = dash.Dash()
app.title = "garnet.crystals.ai"
# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='page-content')
# ])


app.layout = html.Div([
    html.Div(id='cache', style={'display': 'none'}),
    html.H1("Garnet Neural Network", id="title", style={"text-align": "center"}),
    html.Div([
        html.Label('C', style={"width": "5em", "float": "left"}),
        dcc.Input(
            id="C-site",
            placeholder='Ca2+',
            type='text',
            value='',
            style={"width": "5em", "float": "left"}
        ),
        html.Label('A', style={"width": "5em", "float": "left"}),
        dcc.Input(
            id="A-site",
            placeholder='Al3+',
            type='text',
            value='',
            style={"width": "5em", "float": "left"}
        ),
        html.Label('D', style={"width": "7em", "float": "left"}),
        dcc.Input(
            id="D-site",
            placeholder='Si4+',
            type='text',
            value='',
            style={"width": "5em", "float": "left"}
        )
    ]
    ),
    html.Br(),
    html.Button('Compute', id='button'),
    html.Div(id='results-div')
])


# perovskite_layout = html.Div([
#     html.Div(id='cache', style={'display': 'none'}),
#     html.H1("Perovskite Neural Network", id="perovskite-title", style={"text-align": "center"}),
#     html.Div([
#         html.Label('A', style={"width": "5em", "float": "left"}),
#         dcc.Input(
#             id="perovskite-A-site",
#             placeholder='Ca2+',
#             type='text',
#             value='',
#             style={"width": "5em", "float": "left"}
#         ),
#         html.Label('B', style={"width": "5em", "float": "left"}),
#         dcc.Input(
#             id="perovskite-B-site",
#             placeholder='Al3+',
#             type='text',
#             value='',
#             style={"width": "5em", "float": "left"}
#         )
#         ]
#     ),
#     html.Br(),
#     html.Button('Compute', id='button'),
#     html.Div(id='perovskite-results-div')
# ])
#
#
#
#
# @app.callback(dash.dependencies.Output('page-content', 'children'),
#               [dash.dependencies.Input('url', 'pathname')])
# def display_page(pathname):
#     if pathname == '/garnet':
#         return garnet_layout
#     elif pathname == '/perovskite':
#         return perovskite_layout
#     else:
#         return garnet_layout


def html_formula(f):
    return re.sub(r"([\d.]+)", r"<sub>\1</sub>", f)


# @app.route('/', methods=['GET'])
def index():
    return make_response(render_template('index.html'))


ResponseCache = OrderedDict()


# @app.route('/query', methods=['GET'])
@app.callback(
    dash.dependencies.Output('results-div', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('C-site', 'value'),
     dash.dependencies.State('A-site', 'value'),
     dash.dependencies.State('D-site', 'value')]
)
def query(nclicks, c_string, a_string, d_string):
    if nclicks > 0:
        try:
            t0 = time.time()
            structure_type = 'garnet'
            formula = ""

            c_composition = parse_composition(structure_type, c_string, "C")
            a_composition = parse_composition(structure_type, a_string, "A")
            d_composition = parse_composition(structure_type, d_string, "D")

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
                formula = spe2form(structure_type, species)
                if ResponseCache.get(formula):
                    response = ResponseCache[formula]
                    form_e = response['form_e']
                    decomp = response['decomp']
                    ehull = response['ehull']
                    print("Read from Cache")
                else:  # Cache miss
                    with tf.Session() as sess:

                        oxide_table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "data/garnet_oxi_table.json")
                        model, scaler, graph = load_model_and_scaler(structure_type, mix_site) if mix_site \
                            else load_model_and_scaler(structure_type, "unmix")
                        inputs = get_descriptor(structure_type, species)

                        with graph.as_default():
                            form_e = get_form_e(inputs, model, scaler) * 20
                        tot_e = get_tote(structure_type, form_e, species,
                                         oxides_table_path=oxide_table_path)
                        if mix_site:
                            decompose_entries = get_decomposed_entries(structure_type,
                                                                       species,
                                                                       oxide_table_path)
                            decomp, ehull = get_ehull(structure_type, tot_e, species,
                                                      unmix_entries=decompose_entries)
                        else:
                            decomp, ehull = get_ehull(structure_type, tot_e, species)
                            response = {"form_e": form_e, "decomp": decomp, "ehull": ehull}
                        if len(ResponseCache) > MAX_CACHE:
                            ResponseCache.popitem(last=False)
                        ResponseCache.update({formula: response})

            message = [html.P("Ef = %.3f eV/fu" % form_e),
                       html.P("Ehull = %.0f meV/atom" % (ehull * 1000))]
            if ehull > 0:
                message.append(html.Span("Decomposition: "))
                i = 0
                for k, v in decomp.items():
                    comp = k.composition
                    comp, f = comp.get_reduced_composition_and_factor()
                    if i != 0:
                        message.append(html.Span(" + "))
                    message.append(html.Span("%.3f" % (v * f / comp.num_atoms * 20)))
                    message.append(dcc.Link(comp.reduced_formula,
                                            href='https://www.materialsproject.org/materials/%s' % k.entry_id)
                                   )
                    i += 1

            else:
                message = [html.P("Not charge neutral! Total charge = %.0f" % charge)]
        except Exception as ex:
            message = html.P(str(ex))
        print(message)
        print("Time of this query: %s" % (time.time()))
        return message

    # return make_response(render_template(
    #     'index.html',
    #     c_string=c_string, a_string=a_string, d_string=d_string,
    #     formula=html_formula(formula),
    #     message=message
    # )
    # )


# @app.route('/perovskite', methods=['GET'])
def perovskite_index():
    return make_response(render_template('index_perov.html'))


# @app.route('/perovskite_query')
def perovskite_query():
    try:
        t0 = time.time()
        structure_type = 'perovskite'
        a_string = request.args.get("a_string")
        b_string = request.args.get("b_string")
        formula = ""

        a_composition = parse_composition(structure_type, a_string, "A")
        b_composition = parse_composition(structure_type, b_string, "B")

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
            formula = spe2form(structure_type, species)
            if ResponseCache.get(formula):
                response = ResponseCache[formula]
                form_e = response['form_e']
                decomp = response['decomp']
                ehull = response['ehull']
                print("Read from Cache")
            else:  # Cache Miss
                with tf.Session() as sess:
                    oxide_table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "data/perovskite_oxi_table.json")
                    model, scaler, graph = load_model_and_scaler(structure_type, mix_site) if mix_site \
                        else load_model_and_scaler(structure_type, "unmix")
                    inputs = get_descriptor(structure_type, species, cn_specific=False)
                    with graph.as_default():
                        form_e = get_form_e(inputs, model, scaler) * 10
                    # form_e predicted from model is always in /atom
                    # the get_tote func always returns the tote with in /standard fu
                    # which is A2B2O6, 10 atoms
                    tot_e = get_tote(structure_type, form_e, species,
                                     oxides_table_path=oxide_table_path)
                    if mix_site:
                        decompose_entries = get_decomposed_entries(structure_type,
                                                                   species,
                                                                   oxide_table_path)
                        decomp, ehull = get_ehull(structure_type, tot_e, species,
                                                  unmix_entries=decompose_entries)
                    else:
                        decomp, ehull = get_ehull(structure_type, tot_e, species)

                    response = {"form_e": form_e, "decomp": decomp, "ehull": ehull}
                    if len(ResponseCache) > MAX_CACHE:
                        ResponseCache.popitem(last=False)
                    ResponseCache.update({formula: response})

            message = ["<i>E<sub>f</sub></i> = %.3f eV/fu" % (form_e),
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
    print("Time of this query: %s" % (time.time()))
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
    app.run_server(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

    #
    # import cProfile, pstats, os
    #
    # cProfile.run('profile()', 'stats')
    # p = pstats.Stats('stats')
    # p.sort_stats('cumulative').print_stats(30)
    # os.remove('stats')
