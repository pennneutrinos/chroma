import numpy as np
from particle import Particle

# Photon history bits (see photon.h for source)
NO_HIT           = 0x1 << 0
BULK_ABSORB      = 0x1 << 1
SURFACE_DETECT   = 0x1 << 2
SURFACE_ABSORB   = 0x1 << 3
RAYLEIGH_SCATTER = 0x1 << 4
REFLECT_DIFFUSE  = 0x1 << 5
REFLECT_SPECULAR = 0x1 << 6
SURFACE_REEMIT   = 0x1 << 7
SURFACE_TRANSMIT = 0x1 << 8
BULK_REEMIT      = 0x1 << 9
CHERENKOV        = 0x1 << 10
SCINTILLATION    = 0x1 << 11
NAN_ABORT        = 0x1 << 31

class Steps(object):
    def __init__(self,x,y,z,t,dx,dy,dz,ke,edep,qedep):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.ke = ke
        self.edep = edep
        self.qedep = qedep
    

class Vertex(object):
    def __init__(self, particle_name, pos, dir, ke, t0=0.0, pol=None, steps=None, children=None, trackid=-1, pdgcode=None):
        '''Create a particle vertex.

           particle_name: string
               Name of particle, following the GEANT4 convention.  
               Examples: e-, e+, gamma, mu-, mu+, pi0

           pos: array-like object, length 3
               Position of particle vertex (mm)

           dir: array-like object, length 3
               Normalized direction vector

           ke: float
               Kinetic energy (MeV)

           t0: float
               Initial time of particle (ns)
               
           pol: array-like object, length 3
               Normalized polarization vector.  By default, set to None,
               and the particle is treated as having a random polarization.
        '''
        self.particle_name = particle_name
        self.pos = pos
        self.dir = dir
        self.pol = pol
        self.ke = ke
        self.t0 = t0
        self.steps = steps
        self.children = children
        self.trackid = trackid
        self.pdgcode = Particle.from_evtgen_name(self.particle_name).pdgid if pdgcode == None else pdgcode
        
    def __str__(self):
        return 'Vertex('+self.particle_name+',ke='+str(self.ke)+',steps='+str(True if self.steps else False)+')'
    
    __repr__ = __str__

class Photons(object):
    def __init__(self, pos=np.empty((0,3)), dir=np.empty((0,3)), pol=np.empty((0,3)), wavelengths=np.empty((0)), t=None, last_hit_triangles=None, flags=None, weights=None, evidx=None, channel=None):
        '''Create a new list of n photons.

            pos: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Position 3-vectors (mm)

            dir: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Direction 3-vectors (normalized)

            pol: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Polarization direction 3-vectors (normalized)

            wavelengths: numpy.ndarray(dtype=numpy.float32, shape=n)
               Photon wavelengths (nm)

            t: numpy.ndarray(dtype=numpy.float32, shape=n)
               Photon times (ns)

            last_hit_triangles: numpy.ndarray(dtype=numpy.int32, shape=n)
               ID number of last intersected triangle.  -1 if no triangle hit in last step
               If set to None, a default array filled with -1 is created

            flags: numpy.ndarray(dtype=numpy.uint32, shape=n)
               Bit-field indicating the physics interaction history of the photon.  See 
               history bit constants in chroma.event for definition.

            weights: numpy.ndarray(dtype=numpy.float32, shape=n)
               Survival probability for each photon.  Used by 
               photon propagation code when computing likelihood functions.
               
           evidx: numpy.ndarray(dtype=numpy.uint32, shape=n)
               Index of the event in a GPU batch
        '''
        self.pos = np.asarray(pos, dtype=np.float32)
        self.dir = np.asarray(dir, dtype=np.float32)
        self.pol = np.asarray(pol, dtype=np.float32)
        self.wavelengths = np.asarray(wavelengths, dtype=np.float32)

        if t is None:
            self.t = np.zeros(len(pos), dtype=np.float32)
        else:
            self.t = np.asarray(t, dtype=np.float32)

        if last_hit_triangles is None:
            self.last_hit_triangles = np.empty(len(pos), dtype=np.int32)
            self.last_hit_triangles.fill(-1)
        else:
            self.last_hit_triangles = np.asarray(last_hit_triangles,
                                                 dtype=np.int32)

        if flags is None:
            self.flags = np.zeros(len(pos), dtype=np.uint32)
        else:
            self.flags = np.asarray(flags, dtype=np.uint32)

        if weights is None:
            self.weights = np.ones(len(pos), dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32)
            
        if evidx is None:
            self.evidx = np.zeros(len(pos), dtype=np.uint32)
        else:
            self.evidx = np.asarray(evidx, dtype=np.uint32)
            
        if channel is None:
            self.channel = np.zeros(len(pos), dtype=np.uint32)
        else:
            self.channel = np.asarray(channel, dtype=np.uint32)

    @staticmethod 
    def join(photon_list,concatenate=True):
        '''Concatenates many photon objects together efficiently'''
        if concatenate: #internally lists
            pos = np.concatenate([p.pos for p in photon_list])
            dir = np.concatenate([p.dir for p in photon_list])
            pol = np.concatenate([p.pol for p in photon_list])
            wavelengths = np.concatenate([p.wavelengths for p in photon_list])
            t = np.concatenate([p.t for p in photon_list])
            last_hit_triangles = np.concatenate([p.last_hit_triangles for p in photon_list])
            flags = np.concatenate([p.flags for p in photon_list])
            weights = np.concatenate([p.weights for p in photon_list])
            evidx = np.concatenate([p.evidx for p in photon_list])
            channel = np.concatenate([p.channel for p in photon_list])
            return Photons(pos, dir, pol, wavelengths, t,
                           last_hit_triangles, flags, weights, evidx,channel)
        else: #internally scalars
            pos = np.asarray([p.pos for p in photon_list])
            dir = np.asarray([p.dir for p in photon_list])
            pol = np.asarray([p.pol for p in photon_list])
            wavelengths = np.asarray([p.wavelengths for p in photon_list])
            t = np.asarray([p.t for p in photon_list])
            last_hit_triangles = np.asarray([p.last_hit_triangles for p in photon_list])
            flags = np.asarray([p.flags for p in photon_list])
            weights = np.asarray([p.weights for p in photon_list])
            evidx = np.asarray([p.evidx for p in photon_list])
            channel = np.asarray([p.channel for p in photon_list])
            return Photons(pos, dir, pol, wavelengths, t,
                           last_hit_triangles, flags, weights, evidx, channel)

    def __add__(self, other):
        '''Concatenate two Photons objects into one list of photons.

           other: chroma.event.Photons
              List of photons to add to self.

           Returns: new instance of chroma.event.Photons containing the photons in self and other.
        '''
        pos = np.concatenate((self.pos, other.pos))
        dir = np.concatenate((self.dir, other.dir))
        pol = np.concatenate((self.pol, other.pol))
        wavelengths = np.concatenate((self.wavelengths, other.wavelengths))
        t = np.concatenate((self.t, other.t))
        last_hit_triangles = np.concatenate((self.last_hit_triangles, other.last_hit_triangles))
        flags = np.concatenate((self.flags, other.flags))
        weights = np.concatenate((self.weights, other.weights))
        evidx = np.concatenate((self.evidx, other.evidx))
        channel = np.concatenate((self.channel, other.channel))
        return Photons(pos, dir, pol, wavelengths, t,
                       last_hit_triangles, flags, weights, evidx, channel)

    def __len__(self):
        '''Returns the number of photons in self.'''
        return len(self.pos)
        
    def __str__(self):
        if len(self.pos) == 1:
            return 'Photon(pos='+str(self.pos[0])+ \
                         ',dir='+str(self.dir[0])+ \
                         ',pol='+str(self.pol[0])+ \
                         ',wavelength='+str(self.wavelengths[0])+ \
                         ',t='+str(self.t[0])+ \
                         ',last_hit_triangle='+str(self.last_hit_triangles[0])+ \
                         ',flag='+str(self.flags[0])+ \
                         ',weight='+str(self.weights[0])+ \
                         ')'
        else:
            return 'Photons['+str(len(self.pos))+']'
    
    __repr__ = __str__

    def __getitem__(self, key):
        return Photons(self.pos[key], self.dir[key], self.pol[key],
                       self.wavelengths[key], self.t[key],
                       self.last_hit_triangles[key], self.flags[key],
                       self.weights[key],self.evidx[key],self.channel[key])

    def reduced(self, reduction_factor=1.0):
        '''Return a new Photons object with approximately
        len(self)*reduction_factor photons.  Photons are selected
        randomly.'''
        n = len(self)
        choice = np.random.permutation(n)[:int(n*reduction_factor)]
        return self[choice]

class Channels(object):
    def __init__(self, hit, t, q, flags=None, evidx=None):
        '''Create a list of n channels.  All channels in the detector must 
        be included, regardless of whether they were hit.

           hit: numpy.ndarray(dtype=bool, shape=n)
             Hit state of each channel.

           t: numpy.ndarray(dtype=numpy.float32, shape=n)
             Hit time of each channel. (ns)

           q: numpy.ndarray(dtype=numpy.float32, shape=n)
             Integrated charge from hit.  (units same as charge 
             distribution in detector definition)
        '''
        self.hit = hit
        self.t = t
        self.q = q
        self.flags = flags
        self.evidx = evidx

    def hit_channels(self,return_flags=False):
        '''Extract a list of hit channels.
        
        Returns: array of hit channel IDs, array of hit times, array of charges on hit channels
        '''
        if return_flags:
            return self.hit.nonzero()[0], self.t[self.hit], self.q[self.hit], self.flags[self.hit]
        else:
            return self.hit.nonzero()[0], self.t[self.hit], self.q[self.hit]

class Event(object):
    def __init__(self, id=0, vertices=None, photons_beg=None, photons_end=None, photon_tracks=None, photon_parent_trackids=None, hits=None, flat_hits=None, channels=None):
        '''Create an event.

            id: int
              ID number of this event
              
            vertices: list of chroma.event.Vertex objects
              Starting vertices to propagate in this event.  By default
              this is the primary vertex, but complex interactions
              can be representing by putting vertices for the
              outgoing products in this list.

            photons_beg: chroma.event.Photons
              Set of initial photon vertices in this event

            photons_end: chroma.event.Photons
              Set of final photon vertices in this event
              
            photon_tracks: a python list where each index is a chroma.event.Photons
              object that gives the state of the photon at each step or None

            hits: dict of chroma.event.Photons for each channel with hits
              Set photons that were detected by PMTs grouped by channel
              
            flat_hits: chroma.event.Photons
              A regular Photons object for photons detected by PMTs
              
            channels: chroma.event.Channels
              Electronics channel readout information.  Every channel
              should be included, with hit or not hit status indicated
              by the channels.hit flags.
        '''
        self.id = id

        self.nphotons = None

        if vertices is not None:
            if np.iterable(vertices):
                self.vertices = vertices
            else:
                self.vertices = [vertices]
        else:
            self.vertices = []
        
        self.photons_beg = photons_beg
        self.photons_end = photons_end
        self.photon_tracks = photon_tracks
        self.photon_parent_trackids = photon_parent_trackids
        self.hits = hits
        self.flat_hits = flat_hits
        self.channels = channels
