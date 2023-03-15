print("1")
from chroma.generator.mute import *

print("2")
import numpy as np
from chroma.event import Photons, Vertex, Steps
from chroma.tools import argsort_direction

print("3 ... how did we get here??-")
#g4mute()
#from Geant4 import *
#from geant4_pybind import *
#from geant4_pybind import G4ParticleGun, G4Material, G4MaterialPropertiesTable, G4VModularPhysicsList
#from geant4_pybind import G4VModularPhysicsList
print("4")
#g4unmute()
#import g4py.ezgeom
#import g4py.NISTmaterials
#import g4py.ParticleGun
from chroma.generator import _g4chroma
print("5")
import chroma.geometry as geometry
print("6")

def add_prop(prop_table,name,material,prop_str,option=None):
    if prop_str not in material.__dict__:
        return
    if option is None:
        transform = lambda data: (list(data[:, 0].astype(float)),list(data[:, 1].astype(float)))
    elif option == 'wavelength':
        transform = lambda data: (list((2*pi*hbarc / (data[::-1,0] * nanometer)).astype(float)),list(data[::-1, 1].astype(float)))
    elif option == 'dy_dwavelength':
        transform = lambda data: (list((2*pi*hbarc / (data[::-1,0] * nanometer)).astype(float)),list((data[::-1, 1]*np.square(data[::-1, 0])*nanometer/2/pi/hbarc).astype(float)))
    
    data = material.__dict__[prop_str]
    if data is not None:
        if type(data) is dict:
            for prefix,_data in data.items():
                energy, values = transform(_data)
                prop_table.AddProperty(name+prefix, energy, values)
        else:
            energy, values = transform(data)
            prop_table.AddProperty(name, energy, values)
    

def create_g4material(material):
    g4material = G4Material(material.name, material.density * g / cm3,
                            len(material.composition))
    # Add elements
    for element_name, element_frac_by_weight in material.composition.items():
        g4material.AddElement(G4Element.GetElement(element_name, True), element_frac_by_weight)
    
    # Add properties necessary for primary scintillation generation
    prop_table = G4MaterialPropertiesTable()
    #prop_table = g4material.GetMaterialPropertiesTable()
    add_prop(prop_table,'RINDEX',material,'refractive_index',option='wavelength')
    add_prop(prop_table,'SCINTILLATION',material,'scintillation_spectrum',option='dy_dwavelength')
    add_prop(prop_table,'SCINTWAVEFORM',material,'scintillation_waveform') #could be a PDF but this requires time constants
    add_prop(prop_table,'SCINTMOD',material,'scintillation_mod')
    if 'scintillation_light_yield' in material.__dict__:
        data = material.scintillation_light_yield 
        if data is not None:
            prop_table.AddConstProperty('LIGHT_YIELD',data)

    # Load properties into material
    g4material.SetMaterialPropertiesTable(prop_table)
    return g4material


class G4Generator(object):
    def __init__(self, material, seed=None):
        """Create generator to produce photons inside the specified material.

           material: chroma.geometry.Material object with density, 
                     composition dict and refractive_index.

                     composition dictionary should be 
                        { element_symbol : fraction_by_weight, ... }.
                        
                     OR
                     
                     a callback function to build a geant4 geometry and
                     return a list of things to persist with this object

           seed: int, *optional*
               Random number generator seed for HepRandom. If None, generator
               is not seeded.
        """
        print("In Gen 0")
        if seed is not None:
            HepRandom.setTheSeed(seed)
        
        #g4py.NISTmaterials.Construct()
        
        print("In Gen 1")
        if isinstance(material,geometry.Material):
        
            self.world_material = create_g4material(material)
            #g4py.ezgeom.Construct()
            
            #g4py.ezgeom.SetWorldMaterial(self.world_material)
            #g4py.ezgeom.ResizeWorld(100*m, 100*m, 100*m)
            #self.world = g4py.ezgeom.G4EzVolume('world')

            #self.world.CreateBoxVolume(self.world_material, 100*m, 100*m, 100*m)
            solid = G4Box('world', 100*m, 100*m, 100*m)
            lv = G4LogicalVolume(solid, self.world_material, 'world')
            aworld = G4PVPlacement(0, G4ThreeVector(0, 0, 0), "world", lv, 0, false, 0)
            ## EzGeom is gone?
            self.world = aworld
            print("In Gen 2")
            
        else:
            #material is really a function to build the geometry
            self.world = material()
        
        print("In Gen 3")
        self.physics_list = _g4chroma.ChromaPhysicsList()
        gRunManager.SetUserInitialization(self.physics_list)
        self.particle_gun = G4ParticleGun()
        print("In Gen 4")
            
        self.stepping_action = _g4chroma.SteppingAction()
        gRunManager.SetUserAction(self.stepping_action)
        self.tracking_action = _g4chroma.TrackingAction()
        gRunManager.SetUserAction(self.tracking_action)
        #g4mute()
        gRunManager.Initialize()
        print("In Gen 5")
        #g4unmute()
        # preinitialize the process by running a simple event
        self.generate_photons([Vertex('e-', (0,0,0), (1,0,0), 0.5, 1.0)], mute=True)
        
    def _extract_photons_from_tracking_action(self, sort=False):
        n = self.tracking_action.GetNumPhotons()
        pos = np.zeros(shape=(n,3), dtype=np.float32)
        pos[:,0] = self.tracking_action.GetX()
        pos[:,1] = self.tracking_action.GetY()
        pos[:,2] = self.tracking_action.GetZ()

        dir = np.zeros(shape=(n,3), dtype=np.float32)
        dir[:,0] = self.tracking_action.GetDirX()
        dir[:,1] = self.tracking_action.GetDirY()
        dir[:,2] = self.tracking_action.GetDirZ()

        pol = np.zeros(shape=(n,3), dtype=np.float32)
        pol[:,0] = self.tracking_action.GetPolX()
        pol[:,1] = self.tracking_action.GetPolY()
        pol[:,2] = self.tracking_action.GetPolZ()
        
        wavelengths = self.tracking_action.GetWavelength().astype(np.float32)

        t0 = self.tracking_action.GetT0().astype(np.float32)

        flags = self.tracking_action.GetFlags().astype(np.uint32)

        if sort: #why would you ever do this
            reorder = argsort_direction(dir)
            pos = pos[reorder]
            dir = dir[reorder]
            pol = pol[reorder]
            wavelengths = wavelengths[reorder]
            t0 = t0[reorder]
            flags = flags[reorder]

        return Photons(pos, dir, pol, wavelengths, t0, flags=flags)
    
    def _extract_vertex_from_stepping_action(self, index=1):
        track = self.stepping_action.getTrack(index)
        steps = Steps(track.getStepX(),track.getStepY(),track.getStepZ(),track.getStepT(),
                      track.getStepDX(),track.getStepDY(),track.getStepDZ(),track.getStepKE(),
                      track.getStepEDep(),track.getStepQEDep())
        children = [self._extract_vertex_from_stepping_action(track.getChildTrackID(id)) for id in range(track.getNumChildren())]
        return Vertex(track.name, np.array([steps.x[0],steps.y[0],steps.z[0]]), 
                        np.array([steps.dx[0],steps.dy[0],steps.dz[0]]), 
                        steps.ke[0], steps.t[0], steps=steps, children=children, trackid=index, pdgcode=track.pdg_code)
        

    def generate_photons(self, vertices, mute=False, tracking=False):
        """Use GEANT4 to generate photons produced by propagating `vertices`.
           
        Args:
            vertices: list of event.Vertex objects
                List of initial particle vertices.

            mute: bool
                Disable GEANT4 output to console during generation.  (GEANT4 can
                be quite chatty.)

        Returns:
            photons: event.Photons
                Photon vertices generated by the propagation of `vertices`.
        """
        if mute:
            pass
            #g4mute()
            
        self.stepping_action.EnableTracking(tracking);

        photons = Photons()
        if tracking:
            photon_parent_tracks = []
        
        try:
            tracked_vertices = []
            for vertex in vertices:
                self.particle_gun.SetParticleByName(vertex.particle_name)
                #Geant4 seems to call 'ParticleEnergy' KineticEnergy - see G4ParticleGun 
                kinetic_energy = vertex.ke*MeV
                self.particle_gun.SetParticleEnergy(kinetic_energy)

                # Must be float type to call GEANT4 code
                pos = np.asarray(vertex.pos, dtype=np.float64)
                dir = np.asarray(vertex.dir, dtype=np.float64)

                self.particle_gun.SetParticlePosition(G4ThreeVector(*pos)*mm)
                self.particle_gun.SetParticleMomentumDirection(G4ThreeVector(*dir).unit())
                self.particle_gun.SetParticleTime(vertex.t0*ns)

                if vertex.pol is not None:
                    self.particle_gun.SetParticlePolarization(G4ThreeVector(*vertex.pol).unit())

                self.tracking_action.Clear()
                self.stepping_action.ClearTracking()
                gRunManager.BeamOn(1)
                
                if tracking:
                    vertex = self._extract_vertex_from_stepping_action()
                    photon_parent_tracks.append(self.tracking_action.GetParentTrackID().astype(np.int32))
                tracked_vertices.append(vertex)
                photons += self._extract_photons_from_tracking_action()
            if tracking:    
                photon_parent_tracks = [track for track in photon_parent_tracks if len(track)>0]
                photon_parent_tracks = np.concatenate(photon_parent_tracks) if len(photon_parent_tracks) > 0 else []
                
        finally:
            if mute:
                pass
                #g4unmute()
        
        if tracking:
            return (tracked_vertices,photons,photon_parent_tracks)
        else:
            return (tracked_vertices,photons)
