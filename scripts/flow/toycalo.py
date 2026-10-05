#!/usr/bin/env python
"""Jamie Nagle's toy calorimeter, reimplemented (FLOW_NOTES.md, "Jamie's toy
exercises with OT flow matching").

A reimplementation, not his file: toycalo.py itself was not available. The
jet, UE and quenching throws follow the code listings of
nagle_cyclegan_explained.pdf (printed slides 142-144, PDF pages 155-157)
line by line, including the order of the random draws within each function.
What the listings do not show is filled in here and documented:
- the grid: 24 x 64 towers, eta in [-1.1, 1.1) in 24 equal bins, phi in
  [0, 2 pi) in 64 bins (the deck's displays: eta axis -1.1 to 1.1, phi 0 to
  2 pi; the sPHENIX geometry of our other studies);
- deposit(): a particle's energy goes to the tower containing (eta, phi);
  phi wraps around; particles outside |eta| < 1.1 are lost. Here deposit()
  also returns the lost energy, for the acceptance bookkeeping;
- f_from_girth(): g = sum(e r) / sum(e) over all of the vacuum jet's
  particles, r their distance to the jet axis (eta0, phi0), then
  f = clip(0.25 + 0.14 (g - 0.096) / 0.038, 0.02, 0.60) (slide 161);
- a non-positive event multiplicity m (probability ~1e-23 for N(1, 0.1)) is
  redrawn, so that the gamma scale is positive; the number of redraws is
  counted (rng.gamma refuses a negative scale);
- every parent has its own random stream (numpy SeedSequence of the global
  seed, the pool and the parent index), so pools, splits and replicas are
  reproducible and independent.

Quenching acts on the particles of the vacuum jet: each keeps (1 - f) of its
energy and moves away from the jet axis by (1 + f); f E is redistributed into
max(1, Poisson(15)) recoil particles at r ~ U(0.3, 1.0) around the axis
(Dirichlet(1, ..., 1) shares). The same particle list with fresh (f, recoil)
draws gives the repeated quenching of one parent.
"""

import numpy as np

NETA, NPHI = 24, 64
ETA_MIN, ETA_MAX = -1.1, 1.1
DETA = (ETA_MAX - ETA_MIN) / NETA
DPHI = 2 * np.pi / NPHI
ETA_C = ETA_MIN + (np.arange(NETA) + 0.5) * DETA
PHI_C = (np.arange(NPHI) + 0.5) * DPHI

SEED = 20261005        # the study's global seed

def rng_of(pool, parent, role = 0):
    """The random stream of one parent: role 0 jet, 1 UE, 2 + k quench
    replica k."""
    return np.random.default_rng([ SEED, int(pool), int(parent), int(role) ])

def deposit(img, eta, phi, e):
    """Adds each particle's energy to its tower; returns the energy that falls
    outside the acceptance."""
    eta = np.asarray(eta, float)
    ieta = np.floor((eta - ETA_MIN) / DETA).astype(int)
    iphi = np.floor(np.mod(phi, 2 * np.pi) / DPHI).astype(int) % NPHI
    ok = (ieta >= 0) & (ieta < NETA)
    np.add.at(img, (ieta[ok], iphi[ok]), e[ok])
    return float(np.sum(e[~ok]))

def jet_particles(rng, E, eta0, phi0):
    """A toy jet: one or two prongs, each a spray of particles with hard-ish
    fragmentation (slide 142, verbatim)."""
    if rng.random() < 0.4:
        z = rng.uniform(0.1, 0.5); dr = rng.uniform(0.1, 0.3); ang = rng.uniform(0, 2 * np.pi)
        prongs = [ (1 - z, -z * dr * np.cos(ang), -z * dr * np.sin(ang)),
                   (z, (1 - z) * dr * np.cos(ang), (1 - z) * dr * np.sin(ang)) ]
    else:
        prongs = [ (1.0, 0.0, 0.0) ]
    etas, phis, es = [], [], []
    for (frac, de, dp) in prongs:
        n = max(2, rng.poisson(4 + 2 * np.log(frac * E)))
        w = rng.dirichlet(np.full(n, 0.5))
        sig = 0.04 + 0.4 / np.sqrt(frac * E)                        # softer prongs are wider
        r = np.abs(rng.normal(0, sig, n)) / np.sqrt(np.maximum(w * n, 0.05))  # soft particles wider
        r = np.minimum(r, 0.6); a = rng.uniform(0, 2 * np.pi, n)
        etas.append(eta0 + de + r * np.cos(a)); phis.append(phi0 + dp + r * np.sin(a))
        es.append(frac * E * w)
    return np.concatenate(etas), np.concatenate(phis), np.concatenate(es)

def sample_jet_energy(rng, n = None, emin = 20.0, emax = 80.0, power = 4.0):
    """Steeply falling spectrum dN/dE ~ E^-power between emin and emax
    (power 0: flat)."""
    u = rng.random(n)
    if power == 0:
        return emin + u * (emax - emin)
    a = 1 - power
    return (emin ** a + u * (emax ** a - emin ** a)) ** (1 / a)

def background_image(rng, stats = None):
    """Toy central Au+Au underlying event (slide 143): ~0.6 GeV per tower on
    average, gamma-distributed tower energies, an event-level multiplicity
    fluctuation, elliptic flow with a random event plane, and a few soft
    minijets that make correlated clumps."""
    m = rng.normal(1.0, 0.10)
    while m <= 0:                     # never seen; keeps the gamma scale positive
        if stats is not None:
            stats['m_redraws'] = stats.get('m_redraws', 0) + 1
        m = rng.normal(1.0, 0.10)
    v2 = rng.uniform(0.0, 0.08); psi = rng.uniform(0, np.pi)
    if stats is not None:
        stats.update(m = m, v2 = v2, psi = psi)
    mu = 0.6 * m * (1 + 2 * v2 * np.cos(2 * (PHI_C[None, :] - psi))) \
        * (1 - 0.1 * ETA_C[:, None] ** 2)
    k = 1.5
    img = rng.gamma(k, mu / k * np.ones((NETA, NPHI)))
    nmj = rng.poisson(3)
    for _ in range(nmj):
        eta, phi, e = jet_particles(rng, rng.uniform(2, 5), rng.uniform(-1.1, 1.1),
                                    rng.uniform(0, 2 * np.pi))
        deposit(img, eta, phi, e)
    if stats is not None:
        stats['minijets'] = nmj
    return img

def girth_particles(eta, phi, e, eta0, phi0):
    """Particle-level girth of a jet around its axis."""
    r = np.sqrt((eta - eta0) ** 2 + (phi - phi0) ** 2)
    return float(np.sum(e * r) / np.sum(e))

def f_from_girth(eta, phi, e, eta0, phi0):
    """The deterministic quenching strength of slide 161."""
    g = girth_particles(eta, phi, e, eta0, phi0)
    return float(np.clip(0.25 + 0.14 * (g - 0.096) / 0.038, 0.02, 0.60))

class VacuumJet:
    """One parent's vacuum jet: its particles, axis and generated energy."""

    def __init__(self, rng, eta_max, emin = 20.0, emax = 80.0, power = 4.0):
        # pylint: disable=too-many-arguments
        self.E = float(sample_jet_energy(rng, emin = emin, emax = emax, power = power))
        self.eta0 = rng.uniform(-eta_max, eta_max)
        self.phi0 = rng.uniform(0, 2 * np.pi)
        (self.eta, self.phi, self.e) = jet_particles(rng, self.E, self.eta0, self.phi0)
        self.g = girth_particles(self.eta, self.phi, self.e, self.eta0, self.phi0)

    def image(self):
        img = np.zeros((NETA, NPHI))
        lost = deposit(img, self.eta, self.phi, self.e)
        return (img, lost)

    def quench(self, rng, mode = 'beta'):
        """One quenched version (slide 144): (image, f, lost energy). mode:
        beta, f ~ Beta(2, 6) (the toy's); girth, f from the particle girth;
        uniform, f ~ U(0, 0.5) (the broad prior's quenched half)."""
        f = rng.beta(2, 6)
        if mode == 'girth':
            f = f_from_girth(self.eta, self.phi, self.e, self.eta0, self.phi0)
        elif mode == 'uniform':
            f = rng.uniform(0.0, 0.5)
        else:
            assert mode == 'beta', mode
        q = np.zeros((NETA, NPHI))
        lost = deposit(q, self.eta0 + (self.eta - self.eta0) * (1 + f),
                       self.phi0 + (self.phi - self.phi0) * (1 + f), self.e * (1 - f))
        n = max(1, rng.poisson(15)); r = rng.uniform(0.3, 1.0, n)
        a = rng.uniform(0, 2 * np.pi, n)
        lost += deposit(q, self.eta0 + r * np.cos(a), self.phi0 + r * np.sin(a),
                        f * self.E * rng.dirichlet(np.ones(n)))
        return (q, float(f), lost)

def delta_r(eta0, phi0):
    """(24, 64) distance of every tower centre to an axis (phi periodic)."""
    dphi = np.angle(np.exp(1j * (PHI_C[None, :] - phi0)))
    return np.sqrt((ETA_C[:, None] - eta0) ** 2 + dphi ** 2)
