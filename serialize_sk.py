"""
Python and numpy serialization module.
Edit by Chris Emmery, for https://cmry.github.io/notes/_serialize.
All credits go to:
Copyright (c) 2013, Christopher R. Wagner
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# pylint:        disable=W0612,W0122,R0204

from collections import namedtuple, OrderedDict
from types import GeneratorType
import json
import sys
import numpy as np


class Dummy(object):
    """Dummy class to reinitialize."""

    def __init__(self):
        """Empty init."""
        pass


def mod_load(mod, name):
    """Module loader."""
    try:
        getattr(sys.modules[mod], name)
    except KeyError:
        exec("from " + mod + " import " + name)
    return getattr(sys.modules[mod], name)


def isnamedtuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return isinstance(obj, tuple) \
        and hasattr(obj, "_fields") \
        and hasattr(obj, "_asdict") \
        and callable(obj._asdict)


def _serialize(data):
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [_serialize(val) for val in data]
    if isinstance(data, OrderedDict):
        return {"py/collections.OrderedDict":
                [[_serialize(k), _serialize(v)] for k, v in data.items()]}
    if isnamedtuple(data):
        return {"py/collections.namedtuple": {
            "type":   type(data).__name__,
            "fields": list(data._fields),
            "values": [_serialize(getattr(data, f)) for f in data._fields]}}
    # --- custom ---
    if isinstance(data, type):
        return {"py/numpy.type": data.__name__}
    if isinstance(data, np.integer):
        return {"py/numpy.int": int(data)}
    if isinstance(data, np.float):
        return {"py/numpy.float": data.hex()}
    # -------------
    if isinstance(data, dict):
        if all(isinstance(k, str) for k in data):
            return {k: _serialize(v) for k, v in data.items()}
        return {"py/dict": [[_serialize(k), _serialize(v)]
                            for k, v in data.items()]}
    if isinstance(data, tuple):
        return {"py/tuple": [_serialize(val) for val in data]}
    if isinstance(data, set):
        return {"py/set": [_serialize(val) for val in data]}
    if isinstance(data, np.ndarray):
        return {"py/numpy.ndarray": {
            "values": data.tolist(),
            "dtype":  str(data.dtype)}}
    # --- custom ---
    if isinstance(data, GeneratorType):
        return {'py/generator': str(data)}
    if not isinstance(data, type) and hasattr(data, '__module__'):
        return {'py/class': {'name': data.__class__.__name__,
                             'mod': data.__module__,
                             'attr': _serialize(data.__dict__)}}
    if '_csv.reader' in str(type(data)):
        return ''
    if not isinstance(data, type):
        try:
            hook = str(type(data)).split("'")[1].split('.')
            name, mod = hook.pop(-1), '.'.join(hook)
            return {'py/class': {'name': name,
                                 'mod': mod,
                                 'attr': {}}}
        except Exception:
            pass
    raise TypeError("Type %s not data-serializable" % type(data))


def _restore(dct):
    # --- custom ---
    if "py/numpy.type" in dct:
        return np.dtype(dct["py/numpy.type"]).type
    elif "py/numpy.int" in dct:
        return np.int32(dct["py/numpy.int"])
    elif "py/numpy.float" in dct:
        return np.float64.fromhex(dct["py/numpy.float"])
    # -------------
    elif "py/dict" in dct:
        return dict(dct["py/dict"])
    elif "py/tuple" in dct:
        return tuple(dct["py/tuple"])
    elif "py/set" in dct:
        return set(dct["py/set"])
    elif "py/collections.namedtuple" in dct:
        data = dct["py/collections.namedtuple"]
        return namedtuple(data["type"], data["fields"])(*data["values"])
    elif "py/numpy.ndarray" in dct:
        data = dct["py/numpy.ndarray"]
        return np.array(data["values"], dtype=data["dtype"])
    elif "py/collections.OrderedDict" in dct:
        return OrderedDict(dct["py/collections.OrderedDict"])
    # --- custom ---
    elif "py/generator" in dct:
        return []
    elif "py/class" in dct:
        obj = dct["py/class"]
        cls_ = mod_load(obj['mod'], obj['name'])
        class_init = Dummy()
        class_init.__class__ = cls_
        for k, v in _restore(obj['attr']).items():
            setattr(class_init, k, v)
        return class_init
    return dct


def encode(data, fp=False):
    """Python object to file or string."""
    if fp:
        return json.dump(_serialize(data), fp)
    else:
        return json.dumps(_serialize(data))


def decode(hook):
    """File, String, or Dict to python object."""
    try:
        return json.load(hook, object_hook=_restore)
    except (AttributeError, ValueError):
        pass
    try:
        return json.loads(hook, object_hook=_restore)
    except (TypeError, ValueError):
        pass
    return json.loads(json.dumps(hook), object_hook=_restore)
