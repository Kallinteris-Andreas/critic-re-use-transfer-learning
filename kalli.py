import copy


def m_deepcopy(self, excluded_keys: list[str]):
    """similar to `copy.deepcopy`, but excludes copying the member variables in `excluded_keys`."""
    dct = self.__dict__.copy()
    for key in excluded_keys:
        del dct[key]
    # we avoid the normal init. I *think* unpickling does something like this too?
    other = type(self).__new__(type(self))
    other.__dict__ = copy.deepcopy(dct)
    return other