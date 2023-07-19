import os, os.path
import shutil
import numpy as np
import chroma.event as event
from chroma.tools import count_nonzero
from chroma.rootimport import ROOT
import array

# Check if we have already imported the ROOT class due to a user's
# rootlogon.C script
if not hasattr(ROOT, 'Vertex') or not hasattr(ROOT, 'Channel'):
    print('Setting up ROOT datatypes.')
    # Create .chroma directory if it doesn't exist
    chroma_dir = os.path.expanduser('~/.chroma')
    if not os.path.isdir(chroma_dir):
        if os.path.exists(chroma_dir):
            raise Exception('$HOME/.chroma file exists where directory should be')
        else:
            os.mkdir(chroma_dir)
    # Check if latest ROOT file is present
    package_root_C = os.path.join(os.path.dirname(__file__), 'root.C')
    home_root_C = os.path.join(chroma_dir, 'root.C')
    if not os.path.exists(home_root_C) or \
            os.stat(package_root_C).st_mtime > os.stat(home_root_C).st_mtime:
        shutil.copy2(src=package_root_C, dst=home_root_C)
    # ACLiC problem with ROOT
    # see http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=14280&start=15
    # no longer an issue for root 6+
    # ROOT.gSystem.Load('libCint')
    # Import this C file for access to data structure
    if not hasattr(ROOT, 'Event'):
        print(f"Critical Warning: ROOT library ({home_root_C}) is missing the Event class")
    ROOT.gROOT.ProcessLine('.L '+home_root_C+'+')


def tvector3_to_ndarray(vec):
    '''Convert a ROOT.TVector3 into a numpy np.float32 array'''
    return np.array((vec.X(), vec.Y(), vec.Z()), dtype=np.float32)

def make_photon_with_arrays(size):
    '''Returns a new chroma.event.Photons object for `size` number of
    photons with empty arrays set for all the photon attributes.'''
    return event.Photons(pos=np.empty((size,3), dtype=np.float32),
                         dir=np.empty((size,3), dtype=np.float32),
                         pol=np.empty((size,3), dtype=np.float32),
                         wavelengths=np.empty(size, dtype=np.float32),
                         t=np.empty(size, dtype=np.float32),
                         flags=np.empty(size, dtype=np.uint32),
                         last_hit_triangles=np.empty(size, dtype=np.int32))

def root_vertex_to_python_vertex(vertex):
    "Returns a chroma.event.Vertex object from a root Vertex object."
    if len(vertex.step_x):
        n = len(vertex.step_x)
        steps = event.Steps(np.empty(n),np.empty(n),np.empty(n),np.empty(n),
              np.empty(n),np.empty(n),np.empty(n),
              np.empty(n),np.empty(n),np.empty(n))
        ROOT.get_steps(vertex,n,steps.x,steps.y,steps.z,steps.t,
                       steps.dx,steps.dy,steps.dz,
                       steps.ke,steps.edep,steps.qedep)
    else:
        steps = None
    if len(vertex.children) > 0:
        children = [root_vertex_to_python_vertex(child) for child in vertex.children]
    else:
        children = None
    return event.Vertex(str(vertex.particle_name),
                        pos=tvector3_to_ndarray(vertex.pos),
                        dir=tvector3_to_ndarray(vertex.dir),
                        ke=vertex.ke,
                        t0=vertex.t0,
                        pol=tvector3_to_ndarray(vertex.pol),
                        trackid=vertex.trackid,
                        pdgcode=vertex.pdgcode,
                        steps=steps,
                        children=children)

def python_vertex_to_root_vertex(pvertex,rvertex):
    rvertex.particle_name = pvertex.particle_name
    rvertex.pos.SetXYZ(*pvertex.pos)
    rvertex.dir.SetXYZ(*pvertex.dir)
    if pvertex.pol is not None:
        rvertex.pol.SetXYZ(*vertex.pol)
    rvertex.ke = pvertex.ke
    rvertex.t0 = pvertex.t0
    rvertex.trackid = pvertex.trackid
    rvertex.pdgcode = pvertex.pdgcode
    if pvertex.steps:
        ROOT.fill_steps(rvertex,len(pvertex.steps.x),pvertex.steps.x,pvertex.steps.y,pvertex.steps.z,
                        pvertex.steps.t,pvertex.steps.dx,pvertex.steps.dy,pvertex.steps.dz,
                        pvertex.steps.ke,pvertex.steps.edep,pvertex.steps.qedep)
    else:
        nil = np.empty(0,dtype=np.float64)
        ROOT.clear_steps(rvertex)
    if pvertex.children is not None and len(pvertex.children) > 0:
        rvertex.children.resize(len(pvertex.children))
        any(python_vertex_to_root_vertex(pchild,rchild) for pchild,rchild in zip(pvertex.children,rvertex.children))
    else:
        rvertex.children.resize(0)

def root_event_to_python_event(ev):
    '''Returns a new chroma.event.Event object created from the
    contents of the ROOT event `ev`.'''
    pyev = event.Event(ev.id)
    
    for vertex in ev.vertices:
        pyev.vertices.append(root_vertex_to_python_vertex(vertex))

    # photon begin
    if ev.photons_beg.size() > 0:
        photons = make_photon_with_arrays(ev.photons_beg.size())
        ROOT.get_photons(ev.photons_beg,
                         photons.pos.ravel(),
                         photons.dir.ravel(),
                         photons.pol.ravel(),
                         photons.wavelengths,
                         photons.t,
                         photons.last_hit_triangles,
                         photons.flags, 
                         photons.channel)
        pyev.photons_beg = photons

    # photon end
    if ev.photons_end.size() > 0:
        photons = make_photon_with_arrays(ev.photons_end.size())
        ROOT.get_photons(ev.photons_end,
                         photons.pos.ravel(),
                         photons.dir.ravel(),
                         photons.pol.ravel(),
                         photons.wavelengths,
                         photons.t,
                         photons.last_hit_triangles,
                         photons.flags, 
                         photons.channel)
        pyev.photons_end = photons

    # photon tracks
    if ev.photon_tracks.size() > 0:
        photon_tracks = []
        for i in range(ev.photon_tracks.size()):
            photons = make_photon_with_arrays(ev.photon_tracks[i].size())
            ROOT.get_photons(ev.photon_tracks[i],
                             photons.pos.ravel(),
                             photons.dir.ravel(),
                             photons.pol.ravel(),
                             photons.wavelengths,
                             photons.t,
                             photons.last_hit_triangles,
                             photons.flags, 
                             photons.channel)
            photon_tracks.append(photons)
        pyev.photon_tracks = photon_tracks
        pyev.photon_parent_trackids = np.asarray(ev.photon_parent_trackids).copy()    
    
    # hits
    if ev.hits.size() > 0:
        pyev.hits = {}
        for hit in ev.hits:
            photons = make_photon_with_arrays(hit.second.size())
            ROOT.get_photons(hit.second,
                          photons.pos.ravel(),
                          photons.dir.ravel(),
                          photons.pol.ravel(),
                          photons.wavelengths, photons.t,
                          photons.last_hit_triangles, photons.flags,
                          photons.channel)
            pyev.hits[hit.first] = photons

    # flat_hits
    if ev.flat_hits.size() > 0:
        photons = make_photon_with_arrays(ev.flat_hits.size())
        ROOT.get_photons(ev.flat_hits,
                      photons.pos.ravel(),
                      photons.dir.ravel(),
                      photons.pol.ravel(),
                      photons.wavelengths, photons.t,
                      photons.last_hit_triangles, photons.flags,
                      photons.channel)
        pyev.flat_hits = photons

    # photon end
    if ev.photons_end.size() > 0:
        photons = make_photon_with_arrays(ev.photons_end.size())
        ROOT.get_photons(ev.photons_end,
                         photons.pos.ravel(),
                         photons.dir.ravel(),
                         photons.pol.ravel(),
                         photons.wavelengths,
                         photons.t,
                         photons.last_hit_triangles,
                         photons.flags,
                         photons.channel)
        pyev.photons_end = photons

    # channels
    if ev.nchannels > 0:
        hit = np.zeros(ev.nchannels, dtype=np.int32)
        t = np.zeros(ev.nchannels, dtype=np.float32)
        q = np.zeros(ev.nchannels, dtype=np.float32)
        flags = np.zeros(ev.nchannels, dtype=np.uint32)

        ROOT.get_channels(ev, hit, t, q, flags)
        pyev.channels = event.Channels(hit.astype(bool), t, q, flags)
    else:
        pyev.channels = None

    return pyev
    
class RootReader(object):
    '''Reader of Chroma events from a ROOT file.  This class can be used to 
    navigate up and down the file linearly or in a random access fashion.
    All returned events are instances of the chroma.event.Event class.

    It implements the iterator protocol, so you can do

       for ev in RootReader('electron.root'):
           # process event here
    '''

    def __init__(self, filename):
        '''Open ROOT file named `filename` containing TTree `T`.'''
        self.f = ROOT.TFile(filename)
        
        if hasattr(self.f,'CH'):
            ch_info = self.f.CH
            ch_num = ch_info.GetEntries()
            self.ch_pos = np.empty((ch_num,3),dtype=np.float32)
            self.ch_type = np.empty((ch_num,),dtype=np.int32)
            for i in range(ch_num):
                ch_info.GetEntry(i)
                ch_info.pos.GetXYZ(self.ch_pos[i])
                self.ch_type[i] = ch_info.type
            
        self.T = self.f.T
        self.i = -1
        
    def __len__(self):
        '''Returns number of events in this file.'''
        return self.T.GetEntries()

    def __iter__(self):
        for i in range(self.T.GetEntries()):
            self.T.GetEntry(i)
            yield root_event_to_python_event(self.T.ev)

    def __next__(self):
        '''Return the next event in the file. Raises StopIteration
        when you get to the end.'''
        if self.i + 1 >= len(self):
            raise StopIteration

        self.i += 1
        self.T.GetEntry(self.i)
        return root_event_to_python_event(self.T.ev)

    def prev(self):
        '''Return the next event in the file. Raises StopIteration if
        that would go past the beginning.'''
        if self.i <= 0:
            raise StopIteration

        self.i -= 1
        self.T.GetEntry(self.i)
        return root_event_to_python_event(self.T.ev)

    def current(self):
        '''Return the current event in the file.'''
        self.T.GetEntry(self.i) # in case we were iterated over elsewhere
        return root_event_to_python_event(self.T.ev)

    def jump_to(self, index):
        '''Return the event at `index`.  Updates current location.'''
        if index < 0 or index >= len(self):
            raise IndexError
        
        self.i = index

        self.T.GetEntry(self.i)
        return root_event_to_python_event(self.T.ev)

    def index(self):
        '''Return the current event index'''
        return self.i

class RootWriter(object):
    def __init__(self, filename, detector=None):
        self.filename = filename
        self.file = ROOT.TFile(filename, 'RECREATE')
        
        if detector is not None:
            ch_info = ROOT.TTree('CH', 'Chroma channel info')
            ch_pos = ROOT.TVector3()
            ch_type = array.array( 'i', [0])
            ch_info.Branch('pos',ch_pos)
            ch_info.Branch('type',ch_type,'type/I')
            for pos,chtype in zip(detector.channel_index_to_position,
                                  detector.channel_index_to_channel_type):
                ch_pos.SetXYZ(*pos)
                ch_type[0] = chtype
                ch_info.Fill()
            ch_info.Write()
        self.T = ROOT.TTree('T', 'Chroma events')
        self.ev = ROOT.Event()
        self.T.Branch('ev', self.ev)

    def write_event(self, pyev):
        "Write an event.Event object to the ROOT tree as a ROOT.Event object."
        self.ev.id = pyev.id
        
        if pyev.photons_beg is not None:
            photons = pyev.photons_beg
            if len(photons.pos) > 0:
                ROOT.fill_photons(self.ev.photons_beg,
                              len(photons.pos),
                              photons.pos.ravel(),
                              photons.dir.ravel(),
                              photons.pol.ravel(),
                              photons.wavelengths, photons.t,
                              photons.last_hit_triangles, photons.flags, 
                              photons.channel)
        else:
            self.ev.photons_beg.resize(0)

        if pyev.photons_end is not None:
            photons = pyev.photons_end
            if len(photons.pos) > 0:
                ROOT.fill_photons(self.ev.photons_end,
                              len(photons.pos),
                              photons.pos.ravel(),
                              photons.dir.ravel(),
                              photons.pol.ravel(),
                              photons.wavelengths, photons.t,
                              photons.last_hit_triangles, photons.flags, 
                              photons.channel)
        else:
            self.ev.photons_end.resize(0)
        
        if pyev.photon_tracks is not None:
            self.ev.photon_tracks.resize(len(pyev.photon_tracks))
            for i in range(len(pyev.photon_tracks)):
                photons = pyev.photon_tracks[i]
                if len(photons.pos) > 0:
                    ROOT.fill_photons(self.ev.photon_tracks[i],
                              len(photons.pos),
                              photons.pos.ravel(),
                              photons.dir.ravel(),
                              photons.pol.ravel(),
                              photons.wavelengths, photons.t,
                              photons.last_hit_triangles, photons.flags, 
                              photons.channel)
        else:
            self.ev.photon_tracks.resize(0)
        if pyev.photon_parent_trackids is not None:
            self.ev.photon_parent_trackids.resize(len(pyev.photon_parent_trackids))
            np.asarray(self.ev.photon_parent_trackids)[:] = pyev.photon_parent_trackids
        else:
            self.ev.photon_parent_trackids.resize(0)
        
        if pyev.vertices is not None:
            self.ev.vertices.resize(len(pyev.vertices))
            for i, vertex in enumerate(pyev.vertices):
                python_vertex_to_root_vertex(vertex,self.ev.vertices[i])
        else:
            self.ev.vertices.resize(0)
        
        if pyev.hits is not None:
            self.ev.hits.clear()
            for hit in pyev.hits:
                photons = pyev.hits[hit]
                if len(photons.pos) > 0:
                    ROOT.fill_photons(self.ev.hits[hit],len(photons.pos),
                              photons.pos.ravel(),
                              photons.dir.ravel(),
                              photons.pol.ravel(),
                              photons.wavelengths, photons.t,
                              photons.last_hit_triangles, photons.flags, 
                              photons.channel)
        else:
            self.ev.hits.clear()
        
        if pyev.flat_hits is not None:
            photons = pyev.flat_hits
            if len(photons.pos) > 0:
                ROOT.fill_photons(self.ev.flat_hits,
                              len(photons.pos),
                              photons.pos.ravel(),
                              photons.dir.ravel(),
                              photons.pol.ravel(),
                              photons.wavelengths, photons.t,
                              photons.last_hit_triangles, photons.flags, 
                              photons.channel)
        else:
            self.ev.flat_hits.resize(0)
        
        if pyev.channels is not None:
            hit_channels = pyev.channels.hit.nonzero()[0].astype(np.uint32)
            if len(hit_channels) > 0:
                ROOT.fill_channels(self.ev, len(hit_channels), hit_channels, 
                                   len(pyev.channels.hit), 
                                   pyev.channels.t.astype(np.float32), 
                                   pyev.channels.q.astype(np.float32), 
                                   pyev.channels.flags.astype(np.uint32))
            else:
                self.ev.nhit = 0
                self.ev.nchannels = 0
                self.ev.channels.resize(0)
        else:
            self.ev.nhit = 0
            self.ev.nchannels = 0
            self.ev.channels.resize(0)

        self.T.Fill()

    def close(self):
        self.T.Write()
        self.file.Close()
