#!/usr/bin/env python
import time
import os
import numpy as np

from chroma import generator
from chroma import gpu
from chroma import event
from chroma import itertoolset
from chroma.tools import profile_if_possible

import pycuda.driver as cuda

from timeit import default_timer as timer

def pick_seed():
    """
    Returns a seed for a random number generator selected using
    a mixture of the current time and the current process ID.
    """
    return int(time.time()) ^ (os.getpid() << 16) & 2**32-1

class Simulation(object):
    def __init__(self, detector, seed=None, cuda_device=None, particle_tracking=False, photon_tracking=False,
                 geant4_processes=4, nthreads_per_block=64, max_blocks=1024):
        self.detector = detector

        self.nthreads_per_block = nthreads_per_block
        self.max_blocks = max_blocks
        self.photon_tracking = photon_tracking

        if seed is None:
            self.seed = pick_seed()
        else:
            self.seed = seed

        # We have three generators to seed: numpy.random, GEANT4, and CURAND.
        # The latter two are done below.
        np.random.seed(self.seed)

        if geant4_processes > 0:
            self.photon_generator = generator.photon.G4ParallelGenerator(geant4_processes, detector.detector_material, base_seed=self.seed, tracking=particle_tracking)
        else:
            self.photon_generator = None

        self.context = gpu.create_cuda_context(cuda_device)

        if hasattr(detector, 'num_channels'):
            print(f'Using GPU detector with {detector.num_channels()} channels.')
            if detector.num_channels() > 0:
                self.gpu_geometry = gpu.GPUDetector(detector)
                self.gpu_daq = gpu.GPUDaq(self.gpu_geometry)
                self.gpu_pdf = gpu.GPUPDF()
                self.gpu_pdf_kernel = gpu.GPUKernelPDF()
            else:
                raise ValueError(f'Detector has {detector.num_channels()} channels.')
        else:
            self.gpu_geometry = gpu.GPUGeometry(self.gpu_geometry)

        self.rng_states = gpu.get_rng_states(self.nthreads_per_block*self.max_blocks, seed=self.seed)

        self.pdf_config = None
     
    def _simulate_batch(self,batch_events,keep_photons_beg=False,keep_photons_end=False,keep_hits=True,keep_flat_hits=True,run_daq=False, max_steps=100, verbose=False):
        """
        Assumes batch_events is a list of Event objects with photons_beg having evidx set to the index in the array.
        Yields the fully formed events. Do not call directly.
        """
        
        t_start = timer()
        
        #Idea: allocate memory on gpu and copy photons into it, instead of concatenating on CPU?
        batch_photons = event.Photons.join([ev.photons_beg for ev in batch_events])
        batch_bounds = np.cumsum(np.concatenate([[0],[len(ev.photons_beg) for ev in batch_events]]))
        
        #This copy to gpu has a _lot_ of overhead, want 100k photons at least, hence batches
        #Assume triangles, and weights are unimportant to copy to GPU
        t_copy_start = timer()
        gpu_photons = gpu.GPUPhotons(batch_photons,copy_triangles=False,copy_weights=False)
        t_copy_end = timer()
        if verbose:
            print('GPU copy took %0.2f s' % (t_copy_end-t_copy_start))
        
        t_prop_start = timer()
        tracking = gpu_photons.propagate(self.gpu_geometry, self.rng_states,
                              nthreads_per_block=self.nthreads_per_block,
                              max_blocks=self.max_blocks,
                              max_steps=max_steps,track=self.photon_tracking)
            
        t_prop_end = timer()
        if verbose:
            print('GPU propagate took %0.2f s' % (t_prop_end-t_prop_start))
                              
        t_end = timer()
        if verbose:
            print('Batch took %0.2f s' % (t_end-t_start))

        if keep_photons_end:
            batch_photons_end = gpu_photons.get()
            
        if hasattr(self.detector, 'num_channels') and (keep_hits or keep_flat_hits):
            batch_hits = gpu_photons.get_flat_hits(self.gpu_geometry)
                
        for i,(batch_ev,(start_photon,end_photon)) in enumerate(zip(batch_events,zip(batch_bounds[:-1],batch_bounds[1:]))):
                    
            if not keep_photons_beg:
                batch_ev.photons_beg = None
                
            if self.photon_tracking:
                step_photon_ids,step_photons = tracking
                nphotons = end_photon-start_photon
                photon_tracks = [[] for i in range(nphotons)]
                for step_ids,step_photons in zip(step_photon_ids,step_photons):
                    mask = np.logical_and(step_ids >= start_photon,step_ids<end_photon)
                    if np.count_nonzero(mask) == 0:
                        break
                    photon_ids = step_ids[mask]-start_photon
                    photons = step_photons[mask]
                    #Indexing Photons with a scalar changes the internal array shapes...
                    any(photon_tracks[id].append(photons[i]) for i,id in enumerate(photon_ids))
                batch_ev.photon_tracks = [event.Photons.join(photons,concatenate=False) if len(photons) > 0 else event.Photons() for photons in photon_tracks]

            if keep_photons_end:
                batch_ev.photons_end = batch_photons_end[start_photon:end_photon]
                        
            if hasattr(self.detector, 'num_channels') and (keep_hits or keep_flat_hits):
                ev_hits = batch_hits[batch_hits.evidx == i]
                if keep_hits:
                    #This is kind of expensive computationally, but keep_hits is for diagnostics
                    batch_ev.hits = { int(chan):ev_hits[ev_hits.channel == chan] for chan in np.unique(ev_hits.channel) }
                if keep_flat_hits:
                    batch_ev.flat_hits = ev_hits
                    
                        
            if hasattr(self, 'gpu_daq') and run_daq:
                #Must run DAQ per event, or design a much more complicated daq algorithm
                self.gpu_daq.begin_acquire()
                self.gpu_daq.acquire(gpu_photons, self.rng_states, 
                                     start_photon=start_photon, 
                                     nphotons=(end_photon-start_photon),
                                     nthreads_per_block=self.nthreads_per_block, 
                                     max_blocks=self.max_blocks)
                gpu_channels = self.gpu_daq.end_acquire()
                batch_ev.channels = gpu_channels.get()
                    
            yield batch_ev

    def simulate(self, iterable, keep_photons_beg=False, keep_photons_end=False,
                 keep_hits=True, keep_flat_hits=True, run_daq=False, max_steps=1000,
                 photons_per_batch=1000000):
        print("Begin Simulate")
        if isinstance(iterable, event.Photons):
            first_element, iterable = iterable, [iterable]
            print("A")
        else:
            first_element, iterable = itertoolset.peek(iterable)
            print("B")

        if isinstance(first_element, event.Event):
            iterable = self.photon_generator.generate_events(iterable)
            print("1")
        elif isinstance(first_element, event.Photons):
            iterable = (event.Event(photons_beg=x) for x in iterable)
            print("2")
        elif isinstance(first_element, event.Vertex):
            iterable = (event.Event(vertices=[vertex]) for vertex in iterable)
            iterable = self.photon_generator.generate_events(iterable)
            print("3")

        nphotons = 0
        batch_events = []
        
        print("hi > ", iterable)
        for ev in iterable:
            print(f"nphotons: {len(ev.photons_beg)}")
            ev.nphotons = len(ev.photons_beg)
            ev.photons_beg.evidx[:] = len(batch_events)
            
            nphotons += ev.nphotons
            batch_events.append(ev)
            
            #FIXME need an alternate implementation to split an event that is too large
            if nphotons >= photons_per_batch:
                yield from self._simulate_batch(batch_events,
                                                keep_photons_beg=keep_photons_beg,
                                                keep_photons_end=keep_photons_end,
                                                keep_hits=keep_hits,
                                                keep_flat_hits=keep_flat_hits,
                                                run_daq=run_daq, max_steps=max_steps)
                nphotons = 0
                batch_events = []
        if len(batch_events) != 0:
            yield from self._simulate_batch(batch_events,
                                            keep_photons_beg=keep_photons_beg,
                                            keep_photons_end=keep_photons_end,
                                            keep_hits=keep_hits,
                                            keep_flat_hits=keep_flat_hits,
                                            run_daq=run_daq, max_steps=max_steps)

    def create_pdf(self, iterable, tbins, trange, qbins, qrange, nreps=1):
        """Returns tuple: 1D array of channel hit counts, 3D array of
        (channel, time, charge) pdfs."""
        first_element, iterable = itertoolset.peek(iterable)

        if isinstance(first_element, event.Event):
            iterable = self.photon_generator.generate_events(iterable)

        pdf_config = (tbins, trange, qbins, qrange)
        if pdf_config != self.pdf_config:
            self.pdf_config = pdf_config
            self.gpu_pdf.setup_pdf(self.detector.num_channels(), tbins, trange,
                                   qbins, qrange)
        else:
            self.gpu_pdf.clear_pdf()

        if nreps > 1:
            iterable = itertoolset.repeating_iterator(iterable, nreps)

        for ev in iterable:
            gpu_photons = gpu.GPUPhotons(ev.photons_beg)
            gpu_photons.propagate(self.gpu_geometry, self.rng_states,
                                  nthreads_per_block=self.nthreads_per_block,
                                  max_blocks=self.max_blocks)
            self.gpu_daq.begin_acquire()
            self.gpu_daq.acquire(gpu_photons, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
            gpu_channels = self.gpu_daq.end_acquire()
            self.gpu_pdf.add_hits_to_pdf(gpu_channels)
        
        return self.gpu_pdf.get_pdfs()

    @profile_if_possible
    def eval_pdf(self, event_channels, iterable, min_twidth, trange, min_qwidth, qrange, min_bin_content=100, nreps=1, ndaq=1, nscatter=1, time_only=True):
        """Returns tuple: 1D array of channel hit counts, 1D array of PDF
        probability densities."""
        ndaq_per_rep = 64
        ndaq_reps = ndaq // ndaq_per_rep
        gpu_daq = gpu.GPUDaq(self.gpu_geometry, ndaq=ndaq_per_rep)

        self.gpu_pdf.setup_pdf_eval(event_channels.hit,
                                    event_channels.t,
                                    event_channels.q,
                                    min_twidth,
                                    trange,
                                    min_qwidth,
                                    qrange,
                                    min_bin_content=min_bin_content,
                                    time_only=True)

        first_element, iterable = itertoolset.peek(iterable)

        if isinstance(first_element, event.Event):
            iterable = self.photon_generator.generate_events(iterable)
        elif isinstance(first_element, event.Photons):
            iterable = (event.Event(photons_beg=x) for x in iterable)

        for ev in iterable:
            gpu_photons_no_scatter = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps)
            gpu_photons_scatter = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps*nscatter)
            gpu_photons_no_scatter.propagate(self.gpu_geometry, self.rng_states,
                                             nthreads_per_block=self.nthreads_per_block,
                                             max_blocks=self.max_blocks,
                                             use_weights=True,
                                             scatter_first=-1,
                                             max_steps=10)
            gpu_photons_scatter.propagate(self.gpu_geometry, self.rng_states,
                                          nthreads_per_block=self.nthreads_per_block,
                                          max_blocks=self.max_blocks,
                                          use_weights=True,
                                          scatter_first=1,
                                          max_steps=5)
            nphotons = gpu_photons_no_scatter.true_nphotons # same for scatter
            for i in range(gpu_photons_no_scatter.ncopies):
                start_photon = i * nphotons
                gpu_photon_no_scatter_slice = gpu_photons_no_scatter.select(event.SURFACE_DETECT,
                                                                            start_photon=start_photon,
                                                                            nphotons=nphotons)
                gpu_photon_scatter_slices = [gpu_photons_scatter.select(event.SURFACE_DETECT,
                                                                        start_photon=(nscatter*i+j)*nphotons,
                                                                        nphotons=nphotons)
                                             for j in range(nscatter)]
                
                if len(gpu_photon_no_scatter_slice) == 0:
                    continue

                #weights = gpu_photon_slice.weights.get()
                #print 'weights', weights.min(), weights.max()
                for j in range(ndaq_reps):
                    gpu_daq.begin_acquire()
                    gpu_daq.acquire(gpu_photon_no_scatter_slice, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
                    for scatter_slice in gpu_photon_scatter_slices:
                        gpu_daq.acquire(scatter_slice, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks, weight=1.0/nscatter)
                    gpu_channels = gpu_daq.end_acquire()
                    self.gpu_pdf.accumulate_pdf_eval(gpu_channels, nthreads_per_block=ndaq_per_rep)
        
        return self.gpu_pdf.get_pdf_eval()

    def setup_kernel(self, event_channels, bandwidth_iterable,
                         trange, qrange, 
                         nreps=1, ndaq=1, time_only=True, scale_factor=1.0):
        '''Call this before calling eval_pdf_kernel().  Sets up the
        event information and computes an appropriate kernel bandwidth'''
        nchannels = len(event_channels.hit)
        self.gpu_pdf_kernel.setup_moments(nchannels, trange, qrange,
                                          time_only=time_only)
        # Compute bandwidth
        first_element, bandwidth_iterable = itertoolset.peek(bandwidth_iterable)
        if isinstance(first_element, event.Event):
            bandwidth_iterable = \
                self.photon_generator.generate_events(bandwidth_iterable)
        for ev in bandwidth_iterable:
            gpu_photons = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps)
            gpu_photons.propagate(self.gpu_geometry, self.rng_states,
                                  nthreads_per_block=self.nthreads_per_block,
                                  max_blocks=self.max_blocks)
            for gpu_photon_slice in gpu_photons.iterate_copies():
                for idaq in range(ndaq):
                    self.gpu_daq.begin_acquire()
                    self.gpu_daq.acquire(gpu_photon_slice, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
                    gpu_channels = self.gpu_daq.end_acquire()
                    self.gpu_pdf_kernel.accumulate_moments(gpu_channels)
            
        self.gpu_pdf_kernel.compute_bandwidth(event_channels.hit,
                                              event_channels.t,
                                              event_channels.q,
                                              scale_factor=scale_factor)

    def eval_kernel(self, event_channels,
                    kernel_iterable,
                    trange, qrange, 
                    nreps=1, ndaq=1, naverage=1, time_only=True):
        """Returns tuple: 1D array of channel hit counts, 1D array of PDF
        probability densities."""

        self.gpu_pdf_kernel.setup_kernel(event_channels.hit,
                                         event_channels.t,
                                         event_channels.q)
        first_element, kernel_iterable = itertoolset.peek(kernel_iterable)
        if isinstance(first_element, event.Event):
            kernel_iterable = \
                self.photon_generator.generate_events(kernel_iterable)

        # Evaluate likelihood using this bandwidth
        for ev in kernel_iterable:
            gpu_photons = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps)
            gpu_photons.propagate(self.gpu_geometry, self.rng_states,
                                  nthreads_per_block=self.nthreads_per_block,
                                  max_blocks=self.max_blocks)
            for gpu_photon_slice in gpu_photons.iterate_copies():
                for idaq in range(ndaq):
                    self.gpu_daq.begin_acquire()
                    self.gpu_daq.acquire(gpu_photon_slice, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
                    gpu_channels = self.gpu_daq.end_acquire()
                    self.gpu_pdf_kernel.accumulate_kernel(gpu_channels)
        
        return self.gpu_pdf_kernel.get_kernel_eval()

    def __del__(self):
        self.context.pop()
