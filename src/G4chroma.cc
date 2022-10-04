#include "G4chroma.hh"
#include "GLG4Scint.hh"
#include <G4SteppingManager.hh>
#include <G4OpticalPhysics.hh>
#include <G4EmPenelopePhysics.hh>
#include <G4TrackingManager.hh>
#include <G4TrajectoryContainer.hh>
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4OpticalParameters.hh"
#include <G4Alpha.hh>
#include <G4Neutron.hh>

#include <iostream>

using namespace std;

ChromaPhysicsList::ChromaPhysicsList():  G4VModularPhysicsList()
{
  // default cut value  (1.0mm) 
  defaultCutValue = 1.0*mm;

  // General Physics
  RegisterPhysics( new G4EmPenelopePhysics(0) );
  // Optical Physics w/o Scintillation
  auto opticalParams = G4OpticalParameters::Instance();
  opticalParams->SetProcessActivation("Scintillation",false);

  G4OpticalPhysics* opticalPhysics = new G4OpticalPhysics();
  RegisterPhysics( opticalPhysics );
  // Scintillation (handled by stepping!)
  new GLG4Scint(); 
  double neutronMass = G4Neutron::Neutron()->GetPDGMass();
  new GLG4Scint("neutron", 0.9*neutronMass);
  double alphaMass = G4Alpha::Alpha()->GetPDGMass();
  new GLG4Scint("alpha", 0.9*alphaMass);
}

ChromaPhysicsList::~ChromaPhysicsList()
{
}

void ChromaPhysicsList::SetCuts(){
  //  " G4VUserPhysicsList::SetCutsWithDefault" method sets 
  //   the default cut value for all particle types 
  SetCutsWithDefault();   
}

SteppingAction::SteppingAction()
{
    scint = true;
    tracking = false;
    children_mapped = false;
}

SteppingAction::~SteppingAction()
{
}

void SteppingAction::EnableScint(bool enabled) {
    scint = enabled;
}


void SteppingAction::EnableTracking(bool enabled) {
    tracking = enabled;
}


void SteppingAction::UserSteppingAction(const G4Step *step) {

    double qedep = step->GetTotalEnergyDeposit();

    if (scint) {
        
        G4VParticleChange * pParticleChange = GLG4Scint::GenericPostPostStepDoIt(step);
        
        if (pParticleChange) {

            qedep = GLG4Scint::GetLastEdepQuenched();
            
            const size_t nsecondaries = pParticleChange->GetNumberOfSecondaries();
            
            for (size_t i = 0; i < nsecondaries; i++) { 
                G4Track * tempSecondaryTrack = pParticleChange->GetSecondary(i);
                fpSteppingManager->GetfSecondary()->push_back( tempSecondaryTrack );
            }
            
            pParticleChange->Clear();
        }
        
    }
    
    if (tracking) {
        
        const G4Track *g4track = step->GetTrack();
        const int trackid = g4track->GetTrackID();
        Track &track = trackmap[trackid];
        if (track.id == -1) {
            track.id = trackid;
            track.parent_id = g4track->GetParentID();
            track.pdg_code = g4track->GetDefinition()->GetPDGEncoding();
            track.weight = g4track->GetWeight();
            track.name = g4track->GetDefinition()->GetParticleName();
            track.appendStepPoint(step->GetPreStepPoint(), step, 0.0, true);
        }
        track.appendStepPoint(step->GetPostStepPoint(), step, qedep);
        
    }
    
}


void SteppingAction::ClearTracking() {
    trackmap.clear();    
    children_mapped = false;
}

Track& SteppingAction::getTrack(int id) {
    if (!children_mapped) mapChildren();
    return trackmap[id];
}

void SteppingAction::mapChildren() {
    for (auto it = trackmap.begin(); it != trackmap.end(); it++) {
        const int parent = it->second.parent_id;
        trackmap[parent].addChild(it->first);
    }
    children_mapped = true;
}

int Track::getNumSteps() { 
    return steps.size(); 
}  

void Track::appendStepPoint(const G4StepPoint* point, const G4Step* step, double qedep, const bool initial) {
    const double len = initial ? 0.0 : step->GetStepLength();
    
    const G4ThreeVector &position = point->GetPosition();
    const double x = position.x();
    const double y = position.y();
    const double z = position.z();
    const double t = point->GetGlobalTime();

    const G4ThreeVector &direction = point->GetMomentumDirection();
    const double dx = direction.x();
    const double dy = direction.y();
    const double dz = direction.z();
    const double ke = point->GetKineticEnergy();

    const double edep = step->GetTotalEnergyDeposit();


    const G4VProcess *process = point->GetProcessDefinedStep();
    string procname;
    if (process) {
        procname = process->GetProcessName();
    } else if (step->GetTrack()->GetCreatorProcess()) {
        procname =  step->GetTrack()->GetCreatorProcess()->GetProcessName();
    } else {
        procname = "---";
    }
    
    steps.emplace_back(x,y,z,t,dx,dy,dz,ke,edep,qedep,procname);
}

TrackingAction::TrackingAction() {
}

TrackingAction::~TrackingAction() {
}

int TrackingAction::GetNumPhotons() const {
    return pos.size();
}

void TrackingAction::Clear() {
    pos.clear();
    dir.clear();
    pol.clear();
    wavelength.clear();
    t0.clear();
    parentTrackID.clear();
    flags.clear();
}

void TrackingAction::PreUserTrackingAction(const G4Track *track) {
    G4ParticleDefinition *particle = track->GetDefinition();
    if (particle->GetParticleName() == "opticalphoton") {
        uint32_t flag = 0;
        G4String process = track->GetCreatorProcess()->GetProcessName();
        switch (process[0]) {
            case 'S':
                flag |= 1 << 11; //see chroma/cuda/photons.h
                break;
            case 'C':
                flag |= 1 << 10; //see chroma/cuda/photons.h
                break;
        }
        flags.push_back(flag);
        pos.push_back(track->GetPosition()/mm);
        dir.push_back(track->GetMomentumDirection());
        pol.push_back(track->GetPolarization());
        wavelength.push_back( (h_Planck * c_light / track->GetKineticEnergy()) / nanometer );
        t0.push_back(track->GetGlobalTime() / ns);
        parentTrackID.push_back(track->GetParentID());
        const_cast<G4Track *>(track)->SetTrackStatus(fStopAndKill);
    }
}

#define PhotonCopy(type,name,accessor) \
void TrackingAction::name(type *arr) const { \
    for (unsigned i=0; i < pos.size(); i++) arr[i] = accessor; \
}
    
PhotonCopy(double,GetX,pos[i].x())
PhotonCopy(double,GetY,pos[i].y())
PhotonCopy(double,GetZ,pos[i].z())
PhotonCopy(double,GetDirX,dir[i].x())
PhotonCopy(double,GetDirY,dir[i].y())
PhotonCopy(double,GetDirZ,dir[i].z())
PhotonCopy(double,GetPolX,pol[i].x())
PhotonCopy(double,GetPolY,pol[i].y())
PhotonCopy(double,GetPolZ,pol[i].z())
PhotonCopy(double,GetWavelength,wavelength[i])
PhotonCopy(double,GetT0,t0[i])
PhotonCopy(uint32_t,GetFlags,flags[i])
PhotonCopy(int,GetParentTrackID,parentTrackID[i])

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

#define PhotonAccessor(type,name,accessor) \
np::ndarray PTA_##name(const TrackingAction *pta) { \
    np::ndarray r = np::empty(p::make_tuple(pta->GetNumPhotons()),np::dtype::get_builtin<type>()); \
    pta->accessor((type*)r.get_data()); \
    return r; \
}

PhotonAccessor(double,GetX,GetX)
PhotonAccessor(double,GetY,GetY)
PhotonAccessor(double,GetZ,GetZ)
PhotonAccessor(double,GetDirX,GetDirX)
PhotonAccessor(double,GetDirY,GetDirY)
PhotonAccessor(double,GetDirZ,GetDirZ)
PhotonAccessor(double,GetPolX,GetPolX)
PhotonAccessor(double,GetPolY,GetPolY)
PhotonAccessor(double,GetPolZ,GetPolZ)
PhotonAccessor(double,GetWave,GetWavelength)
PhotonAccessor(double,GetT0,GetT0)
PhotonAccessor(uint32_t,GetFlags,GetFlags)
PhotonAccessor(int,GetParentTrackID,GetParentTrackID)

#define StepAccessor(type,name,stepvar) \
np::ndarray PTA_##name(Track *pta) { \
    const vector<Step> &steps = pta->getSteps(); \
    const size_t sz = steps.size(); \
    np::ndarray r = np::empty(p::make_tuple(sz),np::dtype::get_builtin<type>()); \
    for (size_t i = 0; i < sz; i++) r[i] = steps[i].stepvar; \
    return r; \
}
    
StepAccessor(double,getStepX,x)
StepAccessor(double,getStepY,y)
StepAccessor(double,getStepZ,z)
StepAccessor(double,getStepT,t)
StepAccessor(double,getStepDX,dx)
StepAccessor(double,getStepDY,dy)
StepAccessor(double,getStepDZ,dz)
StepAccessor(double,getStepKE,ke)
StepAccessor(double,getStepEDep,edep)
StepAccessor(double,getStepQEDep,qedep)
//StepAccessor(std::string,getStepProcess,procname)

using namespace boost::python;

void export_Chroma()
{
  class_<ChromaPhysicsList, ChromaPhysicsList*, bases<G4VModularPhysicsList>, boost::noncopyable > ("ChromaPhysicsList", "EM+Optics physics list")
    .def(init<>())
    ;
    
    
  class_<Track, Track*, boost::noncopyable> ("Track", "Particle track")
    .def(init<>())
    .def_readonly("track_id",&Track::id)
    .def_readonly("parent_track_id",&Track::parent_id)
    .def_readonly("pdg_code",&Track::pdg_code)
    .def_readonly("weight",&Track::weight)
    .def_readonly("name",&Track::name)
    .def("getNumSteps",&Track::getNumSteps)
    .def("getStepX",PTA_getStepX)
    .def("getStepY",PTA_getStepY)
    .def("getStepZ",PTA_getStepZ)
    .def("getStepT",PTA_getStepT)
    .def("getStepDX",PTA_getStepDX)
    .def("getStepDY",PTA_getStepDY)
    .def("getStepDZ",PTA_getStepDZ)
    .def("getStepKE",PTA_getStepKE)
    .def("getStepEDep",PTA_getStepEDep)
    .def("getStepQEDep",PTA_getStepQEDep)
    //.def("getStepProcess",PTA_getStepProcess)
    .def("getNumChildren",&Track::getNumChildren)
    .def("getChildTrackID",&Track::getChildTrackID)
    ;  

  class_<SteppingAction, SteppingAction*, bases<G4UserSteppingAction>,
	 boost::noncopyable > ("SteppingAction", "Stepping action for hacking purposes")
    .def(init<>())
    .def("EnableScint",&SteppingAction::EnableScint)
    .def("EnableTracking",&SteppingAction::EnableTracking)
    .def("ClearTracking",&SteppingAction::ClearTracking)
    .def("getTrack",&SteppingAction::getTrack,return_value_policy<reference_existing_object>())
    ;  
  
  class_<TrackingAction, TrackingAction*, bases<G4UserTrackingAction>,
	 boost::noncopyable > ("TrackingAction", "Tracking action that saves photons")
    .def(init<>())
    .def("GetNumPhotons", &TrackingAction::GetNumPhotons)
    .def("Clear", &TrackingAction::Clear)
    .def("GetX", PTA_GetX)
    .def("GetY", PTA_GetY)
    .def("GetZ", PTA_GetZ)
    .def("GetDirX", PTA_GetDirX)
    .def("GetDirY", PTA_GetDirY)
    .def("GetDirZ", PTA_GetDirZ)
    .def("GetPolX", PTA_GetPolX)
    .def("GetPolY", PTA_GetPolY)
    .def("GetPolZ", PTA_GetPolZ)
    .def("GetWavelength", PTA_GetWave)
    .def("GetT0", PTA_GetT0)
    .def("GetParentTrackID", PTA_GetParentTrackID)
    .def("GetFlags", PTA_GetFlags)
    ;
}

BOOST_PYTHON_MODULE(_g4chroma)
{
  Py_Initialize();
  np::initialize();
  export_Chroma();
}
