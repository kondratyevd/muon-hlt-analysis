import awkward as ak
import numpy as np
import scipy.stats

class MuCollection:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __iadd__(self, other):
        attrs = [a for a in dir(other) if not a.startswith('__') and not callable(getattr(other, a))]
        for a in attrs:
            if hasattr(self, a):
                attr = getattr(self, a)
                if ak.count(attr, axis=None) == 0:
                    setattr(self, a, getattr(other, a))
                else:
                    setattr(self, a, ak.concatenate([attr, getattr(other, a)]))
            else:
                setattr(self, a, getattr(other, a))
        return self


def clopper_pearson(total, passed, level):
    alpha = (1.0 - level) / 2
    if total == passed:
        hi = 1.0
    else:
        hi = scipy.stats.beta.ppf(1 - alpha, passed + 1, total - passed)
    if passed == 0:
        lo = 0.0
    else:
        lo = scipy.stats.beta.ppf(alpha, passed, total - passed + 1.0)
    return lo, hi 


def regular(nbins, xmin, xmax):
    bins = [xmin]
    bin_width = (xmax-xmin)/float(nbins)
    for i in range(nbins):
        bins.append(xmin + (i+1)*bin_width)
    return bins


def match(first, second, **kwargs):
    if 'dR_cutoff' not in kwargs:
        raise Exception("Please specify dR cutoff for matching!")
    dR_cutoff = kwargs.pop('dR_cutoff', 0.3)
    return_match_properties = kwargs.pop('return_match_properties', False)
    etas = ak.cartesian(
        {'first': first.eta, 'second': second.eta},
        axis=1,
        nested=True
    )
    phis = ak.cartesian(
        {'first': first.phi, 'second': second.phi},
        axis=1,
        nested=True
    )
    dR, deta, dphi = delta_r(etas['first'], etas['second'], phis['first'], phis['second'])
    min_idx = ak.argmin(dR, axis=2)
    match_properties = {
        'pt': second.pt[min_idx],
        'eta': second.eta[min_idx],
        'phi': second.phi[min_idx],
    }
    if return_match_properties:
        return ak.any(dR < dR_cutoff, axis=2), match_properties
    else:
        return ak.any(dR < dR_cutoff, axis=2)


def delta_r(eta1, eta2, phi1, phi2):
    deta = abs(eta1 - eta2)
    dphi = abs(np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi)
    dr = np.sqrt(deta**2 + dphi**2)
    return dr, deta, dphi