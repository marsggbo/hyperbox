import sys
import pathlib
import importlib
import collections


class LazyModule():
    def __init__(self, module):
        self.module = module

    def __repr__(self):
        return repr(self.module)

    def __getattr__(self, name):
        return getattr(self.module, name)


def add_module_properties(module_name, properties):
    module = sys.modules[module_name]
    replace = False
    if isinstance(module, LazyModule):
        lazy_type = type(module)
    else:
        lazy_type = type('LazyModule({})'.format(module_name), (LazyModule,), {})
        replace = True

    for name, prop in properties.items():
        setattr(lazy_type, name, prop)

    if replace:
        sys.modules[module_name] = lazy_type(module)


class staticproperty(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        if fget is not None and not isinstance(fget, staticmethod):
            raise ValueError('fget should be a staticmethod')
        if fset is not None and not isinstance(fset, staticmethod):
            raise ValueError('fset should be a staticmethod')
        if fdel is not None and not isinstance(fdel, staticmethod):
            raise ValueError('fdel should be a staticmethod')
        super().__init__(fget, fset, fdel, doc)

    def __get__(self, inst, cls=None):
        if inst is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget.__get__(inst, cls)() # pylint: disable=no-member

    def __set__(self, inst, val):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        return self.fset.__get__(inst)(val) # pylint: disable=no-member

    def __delete__(self, inst):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        return self.fdel.__get__(inst)() # pylint: disable=no-member


# utils to work with nested collections
def recursive_iter(seq):
    ''' Iterate over elements in seq recursively (returns only non-sequences)
    '''
    if isinstance(seq, collections.abc.Sequence):
        for e in seq:
            for v in recursive_iter(e):
                yield v
    else:
        yield seq


def flatten(seq):
    ''' Flatten all nested sequences, returned type is type of ``seq``
    '''
    return list(recursive_iter(seq))


def copy_structure(data, shape):
    ''' Put data from ``data`` into nested containers like in ``shape``.
        This can be seen as "unflatten" operation, i.e.:
            seq == copy_structure(flatten(seq), seq)
    '''
    d_it = recursive_iter(data)

    def copy_level(s):
        if isinstance(s, collections.abc.Sequence):
            return type(s)(copy_level(ss) for ss in s)
        else:
            return next(d_it)
    return copy_level(shape)


def count(seq):
    ''' Count elements in ``seq`` in a streaming manner.
    '''
    ret = 0
    for _ in seq:
        ret += 1
    return ret


def get_first_n(seq, n):
    ''' Get first ``n`` elements of ``seq`` in a streaming manner.
    '''
    c = 0
    i = iter(seq)
    while c < n:
        yield next(i)
        c += 1


class BackendsAccessor():
    def __init__(self, parent_module_init, parent_module_name):
        self.parent_module_path = pathlib.Path(parent_module_init).parent
        self.parent_module_name = parent_module_name
        self.backends = {}
        self.available_backends = [d.name for d in self.parent_module_path.iterdir() if d.is_dir()]

    def _check_backend(self, backend):
        if backend == 'tf':
            try:
                from nasbench_asr.quiet_tensorflow import tensorflow as _
            except ImportError as e:
                raise ImportError('Tensorflow backend not available') from e
        elif backend == 'torch':
            try:
                import torch as _
            except ImportError as e:
                raise ImportError('PyTorch backend not available') from e
        else:
            raise ValueError(f'Unknown backend: {backend}')

    def _deduce_backend(self):
        try:
            self._check_backend('tf')
            return 'tf'
        except ImportError:
            pass

        try:
            self._check_backend('torch')
            return 'torch'
        except ImportError:
            pass

        raise ImportError('Neither tensorflow nor torch package could not be imported - at least one should be available to train/create models')

    def get_backend(self, backend, set_default=False):
        if backend in self.backends:
            return self.backends[backend]

        is_none = False
        if backend is None:
            backend = self._deduce_backend()
            is_none = True
        else:
            self._check_backend(backend)

        backend_impl = importlib.import_module(f'.{backend}', self.parent_module_name)
        self.backends[backend] = backend_impl
        if is_none or set_default:
            self.backends[None] = backend_impl
        return backend_impl


def make_nice_number(num):
    n = str(num)
    parts = (len(n)-1)//3 + 1
    if parts == 1:
        return n
    offset = len(n)%3 or 3
    breaks = [0] + [offset + i*3 for i in range(parts)] + [len(n)]
    return ','.join(n[breaks[i]:breaks[i+1]] for i in range(parts))
