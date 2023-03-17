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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <typename T, void (TrackingAction::*Method)(T*) const>
py::array_t<T> PhotonAccessor(const TrackingAction *pta) {
  py::array_t<T> r(pta->GetNumPhotons());
  (pta->*Method)((T*)r.request().ptr );
  return r;
}

template <typename T, const T (Step::*Method)>
py::array_t<T> StepAccessor(Track *pta) {
  const vector<Step> &steps = pta->getSteps();
  py::array_t<T> r(steps.size());
  T* np_ptr = (T*)r.request().ptr;
  for (size_t i=0; i < steps.size(); i++){
    np_ptr[i] = steps[i].*Method;
  }
  return r;
}

PYBIND11_MODULE(_g4chroma, mod)
{
  py::class_<G4VModularPhysicsList>(mod, "G4VModularPhysicsList", py::module_local())
    .def(py::init<>());

  py::class_<G4UserTrackingAction>(mod, "G4UserTrackingAction", py::module_local())
    .def(py::init<>());

  py::class_<G4UserSteppingAction>(mod, "G4UserSteppingAction", py::module_local())
    .def(py::init<>());

  py::class_<ChromaPhysicsList, G4VModularPhysicsList>(mod, "ChromaPhysicsList")
    .def(py::init<>());

  py::class_<Track>(mod, "Track")
    .def(py::init<>())
    .def_readonly("track_id",&Track::id)
    .def_readonly("parent_track_id",&Track::parent_id)
    .def_readonly("pdg_code",&Track::pdg_code)
    .def_readonly("weight",&Track::weight)
    .def_readonly("name",&Track::name)
    .def("getNumSteps",&Track::getNumSteps)
    .def("getStepX",StepAccessor<double, &Step::x>)
    .def("getStepY",StepAccessor<double, &Step::y>)
    .def("getStepZ",StepAccessor<double, &Step::z>)
    .def("getStepT",StepAccessor<double, &Step::t>)
    .def("getStepDX",StepAccessor<double, &Step::dx>)
    .def("getStepDY",StepAccessor<double, &Step::dy>)
    .def("getStepDZ",StepAccessor<double, &Step::dz>)
    .def("getStepKE",StepAccessor<double, &Step::ke>)
    .def("getStepEDep",StepAccessor<double, &Step::edep>)
    .def("getStepQEDep",StepAccessor<double, &Step::qedep>)
    .def("getNumChildren",&Track::getNumChildren)
    .def("getChildTrackID",&Track::getChildTrackID);
  
  py::class_<TrackingAction, G4UserTrackingAction>(mod, "TrackingAction")
    .def(py::init<>())
    .def("GetNumPhotons", &TrackingAction::GetNumPhotons)
    .def("Clear", &TrackingAction::Clear)
    .def("GetX", PhotonAccessor<double, &TrackingAction::GetX>)
    .def("GetY", PhotonAccessor<double, &TrackingAction::GetY>)
    .def("GetZ", PhotonAccessor<double, &TrackingAction::GetZ>)
    .def("GetDirX", PhotonAccessor<double, &TrackingAction::GetDirX>)
    .def("GetDirY", PhotonAccessor<double, &TrackingAction::GetDirY>)
    .def("GetDirZ", PhotonAccessor<double, &TrackingAction::GetDirZ>)
    .def("GetPolX", PhotonAccessor<double, &TrackingAction::GetPolX>)
    .def("GetPolY", PhotonAccessor<double, &TrackingAction::GetPolY>)
    .def("GetPolZ", PhotonAccessor<double, &TrackingAction::GetPolZ>)
    .def("GetWavelength", PhotonAccessor<double, &TrackingAction::GetWavelength>)
    .def("GetT0", PhotonAccessor<double, &TrackingAction::GetT0>)
    .def("GetParentTrackID", PhotonAccessor<int, &TrackingAction::GetParentTrackID>)
    .def("GetFlags", PhotonAccessor<uint32_t, &TrackingAction::GetFlags>);

  py::class_<SteppingAction, G4UserSteppingAction>(mod, "SteppingAction")
    .def(py::init<>())
    .def("EnableScint",&SteppingAction::EnableScint)
    .def("EnableTracking",&SteppingAction::EnableTracking)
    .def("ClearTracking",&SteppingAction::ClearTracking)
    //.def("getTrack",&SteppingAction::getTrack,return_value_policy<reference_existing_object>())
    .def("getTrack",&SteppingAction::getTrack);
}
