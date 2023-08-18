from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Iterable, Optional, Union, Any, List
import uproot
from sys import getsizeof
import awkward as ak
from chroma.detector import Detector
from chroma.event import Photons, Event, Vertex, Channels
from pathlib import Path

class Serializer(ABC):
    _fname: Union[str, Path]
    
    def open(self) -> None:
        raise NotImplementedError
    
    def __enter__(self):
        self.open()
        return self

    def close(self) -> None:
        raise NotImplementedError
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write_metadata(self, metadata: dict) -> None:
        raise NotImplementedError

    def set_event_structure(self, dtype: Dict[str, np.dtype]) -> None:
        raise NotImplementedError
    
    def write_event(self, event: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError
    

class RootSerializer(Serializer):
    
    def __init__(self, fname: Union[str, Path]) -> None:
        self._fname = fname
        self._file: Optional[uproot.WritableDirectory] = None
        self._evt_tree = None
        self._event_buffer = {}

    def open(self) -> None:
        self._file = uproot.recreate(self._fname)

    def close(self) -> None:
        if np.any([len(data) > 0 for data in self._event_buffer.values()]):
            self._flush_buffer()
        if self._file is not None:
            self._file.close()

    def write_metadata(self, metadata: dict) -> None:
        assert self._file is not None, "File not open"
        for key, item in metadata.items():
            metadata[key] = np.asarray([item])
        self._file["meta"] = metadata

    def write_event(self, event: Dict[str, np.ndarray]) -> None:
        for entry in event:
            if entry not in self._event_buffer:
                self._event_buffer[entry] = ak.ArrayBuilder()
            self._event_buffer[entry].append(event[entry])
        if self._getBufSize() > 1e5:
            self._flush_buffer()

    
    def _flush_buffer(self)->None:
        assert self._file is not None, "File not open"
        # convert to awkward array for writing
        for entry in self._event_buffer:
            self._event_buffer[entry] = ak.Array(self._event_buffer[entry])
        if 'output' in self._file:
            self._file["output"].extend(self._event_buffer)
        else:
            self._file["output"] = self._event_buffer
        # clear buffer
        for entry in self._event_buffer:
            self._event_buffer[entry] = ak.ArrayBuilder()

    def _getBufSize(self)->int:
        size: int = 0
        for entry, data in self._event_buffer.items():
            size += data.snapshot().nbytes
        return size

_mc_particle_fields: Dict = {
    'mcpdg': np.dtype('i4'),
    'mcx':   np.dtype('f8'),
    'mcy':   np.dtype('f8'),
    'mcz':   np.dtype('f8'),
    'mcu':   np.dtype('f8'),
    'mcv':   np.dtype('f8'),
    'mcw':   np.dtype('f8'),
    'mct':   np.dtype('f8'),
    'mcke':  np.dtype('f8'),
}

class NTupleWriter(object):
    def __init__(self, filename: str, detector: Optional[Detector] = None,
                 write_vertices: bool = True,
                 write_mcphotons: bool = False,
                 write_mcpes: bool = True,
                 write_hits: bool = True,
                ):
        self.filename: Path = Path(filename)
        assert self.filename.parent.is_dir(), f"Directory {self.filename.parent} does not exist"
        if self.filename.suffix == ".root":
            self._serializer: Serializer = RootSerializer(self.filename)
            self._serializer.open()
        else:
            raise NotImplementedError(f"File type {self.filename.suffix} not supported")
        if detector is not None:
            metadata: dict = {}
            metadata['n_channels'] = len(detector.channel_index_to_position)
            metadata['ch_types'] = np.asarray(detector.channel_index_to_channel_type)
            channel_pos = np.asarray(detector.channel_index_to_position)
            metadata['ch_pos_x'] = channel_pos[:,0]
            metadata['ch_pos_y'] = channel_pos[:,1]
            metadata['ch_pos_z'] = channel_pos[:,2]
            self._serializer.write_metadata(metadata)

        self._write_vertices = write_vertices
        self._write_mcphotons = write_mcphotons
        self._write_mcpe = write_mcpes
        self._write_hits = write_hits

    def write_event(self, event: Event) -> None:
        event_dict: Dict[str, Any] = {'evid': event.id}
        if self._write_vertices:
            self._fill_vertices_fields(event_dict, event.vertices)

        if self._write_mcphotons:
            if event.photons_beg is not None:
                self._fill_photons_fields(event_dict, 'photons_beg', event.photons_beg)
            if event.photons_end is not None:
                self._fill_photons_fields(event_dict, 'photons_end', event.photons_end)

        if self._write_mcpe:
            if event.flat_hits is not None and len(event.flat_hits) > 0:
                self._fill_photons_fields(event_dict, 'mcpe', event.flat_hits, write_channel=True)
            elif event.hits is not None and len(event.hits) > 0:
                flat_hits = Photons.join([photon for _, photon in event.hits.items()])
                self._fill_photons_fields(event_dict, 'mcpe', flat_hits, write_channel=True)

        if self._write_hits:
            if event.channels is not None:
                hit_channels, hit_times, hit_charge = event.channels.hit_channels()
                event_dict['hit_pmt'] = np.asarray(hit_channels)
                event_dict['hit_time'] = np.asarray(hit_times)
                event_dict['hit_charge'] = np.asarray(hit_charge)
        self._serializer.write_event(event_dict)

    def close(self) -> None:
        self._serializer.close()
    
    @staticmethod
    def _fill_vertices_fields(event_dict: Dict[str, Any], vertices: List[Vertex]) -> None:
        vertex_dict = {}
        for field in _mc_particle_fields:
                vertex_dict[field] = []
        vertex: Vertex
        for vertex in vertices:
            vertex_dict['mcpdg'].append(vertex.pdgcode)
            vertex_dict['mcx'].append(vertex.pos[0])
            vertex_dict['mcy'].append(vertex.pos[1])
            vertex_dict['mcz'].append(vertex.pos[2])
            vertex_dict['mcu'].append(vertex.dir[0])
            vertex_dict['mcv'].append(vertex.dir[1])
            vertex_dict['mcw'].append(vertex.dir[2])
            vertex_dict['mct'].append(vertex.t0)
            vertex_dict['mcke'].append(vertex.ke)
        for field in _mc_particle_fields:
            vertex_dict[field] = np.asarray(vertex_dict[field])
        event_dict['vertices'] = ak.zip(vertex_dict)

    @staticmethod
    def _fill_photons_fields(event_dict: Dict[str, Any], prefix: str, photons: Photons, write_channel: bool = False) -> None:
        photon_dict = {}
        photon_dict['x'] = np.asarray(photons.pos[:,0])
        photon_dict['y'] = np.asarray(photons.pos[:,1])
        photon_dict['z'] = np.asarray(photons.pos[:,2])
        photon_dict['u'] = np.asarray(photons.dir[:,0])
        photon_dict['v'] = np.asarray(photons.dir[:,1])
        photon_dict['w'] = np.asarray(photons.dir[:,2])
        photon_dict['t'] = np.asarray(photons.t)
        photon_dict['wavelength'] = np.asarray(photons.wavelengths)
        photon_dict['flags'] = np.asarray(photons.flags)
        if write_channel:
            photon_dict['channel'] = np.asarray(photons.channel)
        event_dict[prefix] = ak.zip(photon_dict)
