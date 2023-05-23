#from chroma.generator.mute import *

import numpy as np
from chroma.event import Photons, Vertex, Steps
from chroma.tools import argsort_direction
#from chroma.generator import _g4chroma
#from chroma.generator._g4chroma import *
from geant4_pybind import *
import chroma.geometry as geometry


def add_prop(prop_table,name,material,prop_str,option=None):
    if prop_str not in material.__dict__:
        return
    if option is None:
        transform = lambda data: (list(data[:, 0].astype(float)),list(data[:, 1].astype(float)))
    elif option == 'wavelength':
        transform = lambda data: (list((2*pi*hbarc / (data[::-1,0] * nanometer)).astype(float)),list(data[::-1, 1].astype(float)))
    elif option == 'dy_dwavelength':
        transform = lambda data: (list((2*pi*hbarc / (data[::-2,0] * nanometer)).astype(float)),list((data[::-1, 1]*np.square(data[::-1, 0])*nanometer/2/pi/hbarc).astype(float)))
    
    data = material.__dict__[prop_str]
    if data is not None:
        if type(data) is dict:
            for prefix,_data in data.items():
                energy, values = transform(_data)
                print(f'Energies: {energy}, and values: {values}')
                prop_table.AddProperty(name+prefix, G4doubleVector(energy), G4doubleVector(values))
        else:
            energy, values = transform(data)
            print(f'Energies: {energy}, and values: {values}')
            prop_table.AddProperty(name, G4doubleVector(energy), G4doubleVector(values))
    

def create_g4material(material):
    print(f'Creating material with name {material.name}, density {material.density}, composition {material.composition}')
    #g4material = G4Material(material.name, material.density * g / cm3,
    #                        len(material.composition))
    g4material = G4Material("Carbon", 6, 12.01*g/mole, 2.0*g/cm3)
    print("G4Element.GetNumberOfElements()", G4Element.GetNumberOfElements())
    # Add elements -- fixme
    #for element_name, element_frac_by_weight in material.composition.items():
    #    g4material.AddElement(G4Element.GetElement(element_name, True), element_frac_by_weight)
    
    print("okay so far")
    # Add properties necessary for primary scintillation generation
    prop_table = G4MaterialPropertiesTable()
    #prop_table = g4material.GetMaterialPropertiesTable()
    add_prop(prop_table,'RINDEX',material,'refractive_index',option='wavelength')
    add_prop(prop_table,'SCINTILLATION',material,'scintillation_spectrum',option='dy_dwavelength')
    add_prop(prop_table,'SCINTWAVEFORM',material,'scintillation_waveform') #could be a PDF but this requires time constants
    add_prop(prop_table,'SCINTMOD',material,'scintillation_mod')
    print("prop table okay")
    if 'scintillation_light_yield' in material.__dict__:
        data = material.scintillation_light_yield 
        if data is not None:
            prop_table.AddConstProperty('LIGHT_YIELD',data)

    # Load properties into material
    g4material.SetMaterialPropertiesTable(prop_table)
    print("did a thing no issue")
    return g4material

class G4Detector(G4VUserDetectorConstruction):
    def __init__(self, material):
        super().__init__()
        self.material = material

    def Construct(self):
        nist = G4NistManager.Instance()
        if isinstance(self.material,geometry.Material):
            # Fix me maybe
            print("This is the test instance")
            self.world_material = create_g4material(self.material)
            #self.world_material = nist.FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE")
            #self.world_material = nist.FindOrBuildMaterial("G4_WATER", True)
            print(f'Type of world material: {type(self.world_material)}')
            prop_table = self.world_material.GetMaterialPropertiesTable()
            print(f'Type of prop table: {type(prop_table)}')
            prop_table.DumpTable()

            solidWorld = G4Box('world', 100*m, 100*m, 100*m)
            logicWorld = G4LogicalVolume(solidWorld, self.world_material, 'world')
            physicalWorld = G4PVPlacement(G4RotationMatrix(), G4ThreeVector(0, 0, 0),
                                          'world', logicWorld, None, False, 0)
            self.world = physicalWorld
        else:
            print("DB: New world mat")
            self.world_material = nist.FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE")
            self.world_material = G4Material.GetMaterial("G4_PLASTIC_SC_VINYLTOLUENE")

            solidWorld = G4Box('world', 100*m, 100*m, 100*m)
            logicWorld = G4LogicalVolume(solidWorld, self.world_material, 'world')
            physicalWorld = G4PVPlacement(G4RotationMatrix(), G4ThreeVector(0, 0, 0),
                                          'world', logicWorld, None, False, 0)
            self.world = physicalWorld
        return self.world

class G4Primary(G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.particle_gun = G4ParticleGun(1)
        self.particle_table = G4ParticleTable.GetParticleTable()

    def set_vertex(self, vertex):
        self.vertex = vertex

    def GeneratePrimaries(self, anEvent):
        #self.particle_gun.SetParticleByName(self.vertex.particle_name)
        chosen_particle = self.particle_table.FindParticle(self.vertex.particle_name)
        self.particle_gun.SetParticleDefinition(chosen_particle)
        print("A.2.1.2", self.vertex.particle_name)
        #Geant4 seems to call 'ParticleEnergy' KineticEnergy - see G4ParticleGun 
        kinetic_energy = self.vertex.ke*MeV
        print(f"A.2.1.3 -> {kinetic_energy}")

        # Must be float type to call GEANT4 code
        pos = np.asarray(self.vertex.pos, dtype=np.float64)
        dir = np.asarray(self.vertex.dir, dtype=np.float64)

        self.particle_gun.SetParticlePosition(G4ThreeVector(*pos)*mm)
        self.particle_gun.SetParticleMomentumDirection(G4ThreeVector(*dir).unit())
        self.particle_gun.SetParticleTime(self.vertex.t0*ns)
        self.particle_gun.SetParticleEnergy(kinetic_energy)
        print("A.2.1.4")
        if self.vertex.pol is not None:
            self.particle_gun.SetParticlePolarization(G4ThreeVector(*(self.vertex.pol)).unit())
        self.particle_gun.GeneratePrimaryVertex(anEvent)

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
        if seed is not None:
            HepRandom.setTheSeed(seed)

        self.run_manager = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
        #self.run_manager.SetVerboseLevel(100)

        # Test material
        material = geometry.Material('omnom')
        material.refractive_index = np.array([[100.0, 1.5],[900.0, 1.5]])
        material.density = 1.0
        material.composition = {'C': 1.0}
        #material.scintillation_light_yield = 10000.0
        #material.scintillation_waveform = [1.0, 1.0, 1.0]
        #material.scintillation_decay_time = 1.0
        #material.scintillation_rise_time = 0.1
        #material.scintillation_yield_ratio = 1.0

        self.detector_construction = G4Detector(material)

        self.run_manager.SetUserInitialization( self.detector_construction )
        self.physics_list = ChromaPhysicsList()

        self.run_manager.SetUserInitialization(self.physics_list)
        self.particle_gun = G4ParticleGun()

        self.primary_generator = G4Primary()
        self.stepping_action = SteppingAction()
        self.tracking_action = TrackingAction()
        self.run_manager.SetUserAction(self.primary_generator)
        self.run_manager.SetUserAction(self.stepping_action)
        self.run_manager.SetUserAction(self.tracking_action)

        UImanager = G4UImanager.GetUIpointer()
        #UImanager.ApplyCommand("/tracking/verbose 1")
        #UImanager.ApplyCommand("/tracking/storeTrajectory 1")

        ## Test test test
        print("HELP!")
        visManager = G4VisExecutive()

        # Initialization
        self.run_manager.Initialize()
        self.generate_photons([Vertex('e-', (0,0,0), (1,0,0), 5.0, 1.0)], mute=False, tracking=True)
        #self.generate_photons([Vertex('opticalphoton', (0,0,0), (1,0,0), 2e-6, 1.0)], mute=False, tracking=True)
        # Testing Visualization
        visManager.Initialize()
        ui = G4UIExecutive(1, ['--interactive'])
        #UImanager.ApplyCommand("/control/execute vis.mac")
        #print("HELP!")
        #ui.SessionStart()

    def _extract_photons_from_tracking_action(self, sort=False):
        n = self.tracking_action.GetNumPhotons()
        print(f"extracted {n} photons!!")
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
        print('test', track.pdg_code)
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
                self.primary_generator.set_vertex(vertex)
                self.tracking_action.Clear()
                self.stepping_action.ClearTracking()
                self.run_manager.BeamOn(1)

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
