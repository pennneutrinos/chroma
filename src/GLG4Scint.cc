/**
 * @file GLG4Scint.cc
 *
 * For GLG4Scint class, providing advanced scintillation process.
 * Distantly based on an extensively modified version of G4Scintillation.cc.
 *
 * This file is part of the GenericLAND software library.
 * $Id: GLG4Scint.cc,v 1.2 2006/03/08 03:52:41 volsung Exp $
 *
 * @author Glenn Horton-Smith (Tohoku) 28-Jan-1999
 *
 * 4 January, 2009
 * V.V. Golovko changed method GetPhotonMomentum()
 *                          to GetPhotonEnergy()
 * V.V. Golovko changed method GetMinPhotonMomentum()
 *                          to GetMinPhotonEnergy()
 * V.V. Golovko changed method GetMaxPhotonMomentum()
 *                          to GetMaxPhotonEnergy()
 *
 */

// [see detailed class description in GLG4Scint.hh]

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4UnitsTable.hh"
#include "GLG4Scint.hh"
#include "G4ios.hh"
#include "G4Timer.hh"
#include "Randomize.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"
#include "G4TrackFastVector.hh"

#include "G4IonTable.hh"

#include "G4hZiegler1985Nuclear.hh"
#include "G4hZiegler1985p.hh"
#include "G4hIonEffChargeSquare.hh"
#include "G4hParametrisedLossModel.hh"
#include "G4PSTARStopping.hh"
#include "G4AtomicShells.hh"
#include "G4ParticleTable.hh"

#include <G4Event.hh>
#include <G4EventManager.hh>
#include <sstream>
#include <iostream>

// //////////////
// Helpers
// //////////////

G4PhysicsOrderedFreeVector* Integrate_MPV_to_POFV(G4MaterialPropertyVector *inputVector) {
    G4PhysicsOrderedFreeVector *aPhysicsOrderedFreeVector = new G4PhysicsOrderedFreeVector();

    // Retrieve the first intensity point in vector
    // of (photon momentum, intensity) pairs

    unsigned int i         = 0;
    G4double     currentIN = (*inputVector)[i];

    if (currentIN >= 0.0)  {
        // Create first (photon momentum, Scintillation Integral pair
        G4double currentPM = inputVector->Energy(i);
        G4double currentCII = 0.0;
        aPhysicsOrderedFreeVector->InsertValues(currentPM, currentCII);

        // Set previous values to current ones prior to loop
        G4double prevPM  = currentPM;
        G4double prevCII = currentCII;
        G4double prevIN  = currentIN;

        // loop over all (photon momentum, intensity)
        // pairs stored for this material
        while (i < inputVector->GetVectorLength() - 1) {
            i++;
            currentPM = inputVector->Energy(i);
            currentIN = (*inputVector)[i];

            currentCII = 0.5 * (prevIN + currentIN);

            currentCII = prevCII +  (currentPM - prevPM) * currentCII;

            aPhysicsOrderedFreeVector->
            InsertValues(currentPM, currentCII);

            prevPM  = currentPM;
            prevCII = currentCII;
            prevIN  = currentIN;
        }
    }

    return aPhysicsOrderedFreeVector;
}

// //////////////
// Static data members
// //////////////

std::vector<GLG4Scint *> GLG4Scint::masterVectorOfGLG4Scint;
G4UIdirectory *GLG4Scint::GLG4ScintDir      = NULL;
G4int GLG4Scint::maxTracksPerStep           = 180000;
G4double GLG4Scint::meanPhotonsPerSecondary = 1.0;
G4bool   GLG4Scint::doScintillation         = true;
G4double GLG4Scint::totEdep                 = 0.0;
G4double GLG4Scint::lastEdep_quenched       = 0.0;
G4double GLG4Scint::totEdep_quenched        = 0.0;
G4double GLG4Scint::totEdep_time            = 0.0;
G4ThreeVector GLG4Scint::scintCentroidSum(0.0, 0.0, 0.0);
G4double GLG4Scint::QuenchingFactor = 1.0;
G4bool   GLG4Scint::UserQF          = false;
DummyProcess GLG4Scint::scintProcess("Scintillation", fUserDefined);

// ///////////////
// Constructors
// ///////////////

GLG4Scint::GLG4Scint(const G4String& tablename, G4double lowerMassLimit) {
    verboseLevel = 0;
    myLowerMassLimit = lowerMassLimit;
    
    myPhysicsTable = MyPhysicsTable::FindOrBuild(tablename);
    myPhysicsTable->IncUsedBy();

    if (verboseLevel) myPhysicsTable->Dump();

    // Add to ordered list (largest minimum mass first)
    if ((masterVectorOfGLG4Scint.size() == 0) || (lowerMassLimit < masterVectorOfGLG4Scint.back()->myLowerMassLimit)) {
        masterVectorOfGLG4Scint.push_back(this);
    } else {
        for (std::vector<GLG4Scint *>::iterator i = masterVectorOfGLG4Scint.begin(); i != masterVectorOfGLG4Scint.end(); i++) {
            if (lowerMassLimit > (*i)->myLowerMassLimit) {
                masterVectorOfGLG4Scint.insert(i, this);
                break;
            }
        }
    }

    // Create UI commands if necessary
    if (GLG4ScintDir == NULL) {
        GLG4ScintDir = new G4UIdirectory("/glg4scint/");
        GLG4ScintDir->SetGuidance("scintillation process control.");
        G4UIcommand *cmd;
        cmd = new G4UIcommand("/glg4scint/on", this);
        cmd->SetGuidance("Turn on scintillation");
        cmd = new G4UIcommand("/glg4scint/off", this);
        cmd->SetGuidance("Turn off scintillation");
        cmd->SetParameter(new G4UIparameter("status", 's', false));
        cmd = new G4UIcommand("/glg4scint/maxTracksPerStep", this);
        cmd->SetGuidance("Set maximum number of opticalphoton tracks per step\n (If more real photons are needed, weight of tracked particles is increased.)\n");
        cmd->SetParameter(new G4UIparameter("maxTracksPerStep", 'i', false));
        cmd = new G4UIcommand("/glg4scint/meanPhotonsPerSecondary", this);
        cmd->SetGuidance("Set mean number of \"real\" photons per secondary\n");
        cmd->SetParameter(new G4UIparameter("meanPhotonsPerSecondary", 'd', false));
        cmd = new G4UIcommand("/glg4scint/verbose", this);
        cmd->SetGuidance("Set verbose level");
        cmd->SetParameter(new G4UIparameter("level", 'i', false));
        cmd = new G4UIcommand("/glg4scint/dump", this);
        cmd->SetGuidance("Dump tables");
        cmd = new G4UIcommand("/glg4scint/setQF", this);
        cmd->SetGuidance("Set a constant quenching factor, default is 1");
        cmd->SetParameter(new G4UIparameter("QuenchingFactor", 'd', false));
        cmd->SetGuidance("Set Time Precision Goal in the scintillation time delay, default is 0.001 ns");
        cmd->SetGuidance("Set maximum number of iterations in the scintillation time delay, default is 1000");
    }

#ifdef G4VERBOSE
    G4cout << "GLG4Scint[" << tablename << "]" << " is created " << G4endl;
#endif // ifdef G4VERBOSE
}

// //////////////
// Destructors
// //////////////

GLG4Scint::~GLG4Scint()  {
    myPhysicsTable->DecUsedBy();

    for (std::vector<GLG4Scint *>::iterator i = masterVectorOfGLG4Scint.begin();
         i != masterVectorOfGLG4Scint.end();
         i++) {
        if (*i == this) {
            masterVectorOfGLG4Scint.erase(i);
            break;
        }
    }
}

// //////////
// Methods
// //////////

// Sets the quenching factor
void GLG4Scint::SetQuenchingFactor(G4double qf = 1.0) {
    QuenchingFactor = qf;
}

// This routine is called for each step of any particle
// in a scintillator.  For accurate energy deposition, must be called
// from user-supplied UserSteppingAction, which also must stack
// any particles created.  A pseudo-Poisson-distributed number of
// photons is generated according to the scintillation yield formula,
// distributed evenly along the track segment and uniformly into 4pi.
G4VParticleChange *
GLG4Scint::PostPostStepDoIt(const G4Track& aTrack, const G4Step& aStep) {
    
    // prepare to generate an event, organizing to
    // check for things that cause an early exit.
    aParticleChange.Initialize(aTrack);
    aParticleChange.SetNumberOfSecondaries(0);
    
    // Now we are done if we are not actually making photons here
    if (!doScintillation)  return &aParticleChange;
    
    if (aTrack.GetDefinition() == G4OpticalPhoton::OpticalPhoton()) return &aParticleChange;
    
    const G4Material *aMaterial               = aTrack.GetMaterial();
    const MyPhysicsTable::Entry *physicsEntry =  myPhysicsTable->GetEntry(aMaterial->GetIndex());

    if (!physicsEntry) return &aParticleChange;
    
    // Retrieve the Light Yield or Scintillation Integral for this material
    G4double ScintillationYield                       = physicsEntry->light_yield;
    G4PhysicsOrderedFreeVector *ScintillationIntegral = physicsEntry->spectrumIntegral;

    if (!ScintillationIntegral) return &aParticleChange;
    
    G4double TotalEnergyDeposit = aStep.GetTotalEnergyDeposit();
    
    if (TotalEnergyDeposit <= 0.0) return &aParticleChange;
    
    // Finds E-dependent QF, unless the user provided an E-independent one
    if (!UserQF) {
        if (physicsEntry->QuenchingArray) {
            // This interpolates or uses first/last value if out of range
            SetQuenchingFactor(physicsEntry->QuenchingArray->Value(aTrack.GetVertexKineticEnergy()));
        } else {
            SetQuenchingFactor(1.0);
        }
    }
    
    // If no LY defined Max Scintillation Integral == ScintillationYield
    
    if (!ScintillationYield) {
        ScintillationYield = ScintillationIntegral->GetMaxValue();
    }
    
    // Set positions, directions, etc.
    G4StepPoint *pPreStepPoint  = aStep.GetPreStepPoint();
    G4StepPoint *pPostStepPoint = aStep.GetPostStepPoint();

    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4ThreeVector p0 = pPreStepPoint->GetMomentumDirection();
    G4double t0      = pPreStepPoint->GetGlobalTime();

    // Finally ready to start generating the event
    // figure out how many photons we want to make
    G4int numSecondaries;
    G4double weight;

    // Apply Birk's law
    // Astr. Phys. 30 (2008) 12 uses custom dE/dx, different from G4/Ziegler's
    G4double birksConstant              = physicsEntry->birksConstant;
    G4double QuenchedTotalEnergyDeposit = TotalEnergyDeposit * 1.0;

    if (birksConstant != 0.0) {
        G4double dE_dx = TotalEnergyDeposit /  aStep.GetStepLength();
        QuenchedTotalEnergyDeposit /= (1.0 + birksConstant * dE_dx);
    }

    // Track total edep, quenched edep
    totEdep          += TotalEnergyDeposit;
    totEdep_quenched += QuenchedTotalEnergyDeposit;
    lastEdep_quenched = QuenchedTotalEnergyDeposit;
    totEdep_time      = t0;
    scintCentroidSum + QuenchedTotalEnergyDeposit * (x0 + p0 * (0.5 * aStep.GetStepLength()));
    
    // Calculate MeanNumPhotons
    G4double MeanNumPhotons = (ScintillationYield * GetQuenchingFactor() * QuenchedTotalEnergyDeposit * (1.0 + birksConstant * (physicsEntry->ref_dE_dx)));

    if (MeanNumPhotons <= 0.0) {
        return &aParticleChange;
    }

    // Randomize number of TRACKS (not photons)
    // this gets statistics right for number of PE after applying
    // boolean random choice to final absorbed track (change from
    // old method of applying binomial random choice to final absorbed
    // track, which did want poissonian number of photons divided
    // as evenly as possible into tracks)
    // Note for weight=1, there's no difference between tracks and photons.
    G4double MeanNumTracks = (MeanNumPhotons / meanPhotonsPerSecondary);

    G4double resolutionScale = physicsEntry->resolutionScale;

    if (MeanNumTracks > 12.0) {
        numSecondaries = (G4int)(CLHEP::RandGauss::shoot(MeanNumTracks,  resolutionScale  * sqrt(MeanNumTracks)));
    } else {
        if (resolutionScale > 1.0) {
            MeanNumTracks =  CLHEP::RandGauss::shoot(MeanNumTracks, (sqrt(resolutionScale * resolutionScale - 1.0) * MeanNumTracks));
        }
        numSecondaries =  (G4int)(CLHEP::RandPoisson::shoot(MeanNumTracks));
    }

    weight = meanPhotonsPerSecondary;

    if (numSecondaries > maxTracksPerStep) {
        // It's probably better to just set meanPhotonsPerSecondary to
        // a big number if you want a small number of secondaries, but
        // this feature is retained for backwards compatibility.
        weight         = weight * numSecondaries / maxTracksPerStep;
        numSecondaries = maxTracksPerStep;
    }

    // if there are no photons, then we're all done now
    if (numSecondaries <= 0) {
        return &aParticleChange;
    }

    // Okay, we will make at least one secondary.
    // Notify the proper authorities.
    aParticleChange.SetNumberOfSecondaries(numSecondaries);

    if (aTrack.GetTrackStatus() == fAlive) {
        aParticleChange.ProposeTrackStatus(fSuspend);
    }

    // Now look up waveform information we need to add the secondaries
    G4PhysicsOrderedFreeVector *WaveformIntegral = physicsEntry->timeIntegral;

    for (G4int iSecondary = 0; iSecondary < numSecondaries; iSecondary++) {
    
        // Normal scintillation
        G4double CIIvalue = G4UniformRand() * ScintillationIntegral->GetMaxValue();
        
        // Determine photon momentum
        G4double sampledMomentum = ScintillationIntegral->GetEnergy(CIIvalue);

        // Generate random photon direction
        G4double cost = 1.0 - 2.0 * G4UniformRand();
        G4double sint = sqrt(1.0 - cost * cost);  // FIXED BUG from G4Scint

        G4double phi = 2.0 * M_PI * G4UniformRand();
        G4double sinp = sin(phi);
        G4double cosp = cos(phi);

        G4double px = sint*cosp;
        G4double py = sint*sinp;
        G4double pz = cost;

        // Create photon momentum direction vector
        G4ParticleMomentum photonMomentum(px, py, pz);

        // Determine polarization of new photon
        G4double sx = cost * cosp;
        G4double sy = cost * sinp;
        G4double sz = -sint;

        G4ThreeVector photonPolarization(sx, sy, sz);

        G4ThreeVector perp = photonMomentum.cross(photonPolarization);

        phi = 2 * M_PI * G4UniformRand();
        sinp = sin(phi);
        cosp = cos(phi);

        photonPolarization = cosp * photonPolarization + sinp * perp;
        photonPolarization = photonPolarization.unit();

        // Generate a new photon
        G4DynamicParticle* aScintillationPhoton = new G4DynamicParticle(G4OpticalPhoton::OpticalPhoton(), photonMomentum);
        aScintillationPhoton->SetPolarization(photonPolarization.x(), photonPolarization.y(), photonPolarization.z());
        aScintillationPhoton->SetKineticEnergy(sampledMomentum);

        // Generate new G4Track object
        G4double delta = G4UniformRand() * aStep.GetStepLength();
        G4ThreeVector aSecondaryPosition = x0 + delta * p0;

        // Start deltaTime based on where on the track it happened
        G4double deltaTime = (delta / ((pPreStepPoint->GetVelocity() + pPostStepPoint->GetVelocity()) / 2.0));

        // Delay for scintillation time
        if (WaveformIntegral) {
            G4double WFvalue = G4UniformRand()*WaveformIntegral->GetMaxValue();
            G4double sampledDelayTime = WaveformIntegral->GetEnergy(WFvalue);
            deltaTime += sampledDelayTime;
        }

        // Set secondary time
        G4double aSecondaryTime = t0 + deltaTime;

        // Create secondary track
        G4Track* aSecondaryTrack = new G4Track(aScintillationPhoton, aSecondaryTime, aSecondaryPosition);
        aSecondaryTrack->SetWeight(weight);
        aSecondaryTrack->SetParentID(aTrack.GetTrackID());
        aSecondaryTrack->SetCreatorProcess(&scintProcess);
        
        //RAT::TrackInfo* trackInfo = new RAT::TrackInfo();
        //trackInfo->SetCreatorStep(aTrack.GetCurrentStepNumber());
        //trackInfo->SetCreatorProcess(scintProcess.GetProcessName());
        //aSecondaryTrack->SetUserInformation(trackInfo);

        // Add the secondary to the ParticleChange object
        aParticleChange.SetSecondaryWeightByProcess(true); // recommended
        aParticleChange.AddSecondary(aSecondaryTrack);

        // AddSecondary() overrides our setting of the secondary track weight
        // in Geant4.3.1 & earlier (and also later, at least until Geant4.7?).
        // Maybe not required if SetWeightByProcess(true) called,
        // but we do both, just to be sure
        aSecondaryTrack->SetWeight(weight);
    }

    return &aParticleChange;
}

// The generic (static) PostPostStepDoIt
G4VParticleChange * GLG4Scint::GenericPostPostStepDoIt(const G4Step *pStep) {
    G4Track *track = pStep->GetTrack();
    G4double mass  = track->GetDynamicParticle()->GetMass();
    // Choose the set of properties with the largest minimum mass less than this mass
    for (size_t i = 0; i < masterVectorOfGLG4Scint.size(); i++) {
        if (mass > masterVectorOfGLG4Scint[i]->myLowerMassLimit) {
            return masterVectorOfGLG4Scint[i]->PostPostStepDoIt(*track, *pStep);
        }
    }
    return NULL;
}

// //////////////////////////////////////////////////////////////
// MyPhysicsTable (nested class) definitions
// //////////////////////////////////////////////////////////////

// //////////////
// "Static" members of the class
// [N.B. don't use "static" keyword here, because it means something
// entirely different in this context.]
// //////////////

GLG4Scint::MyPhysicsTable *GLG4Scint::MyPhysicsTable::head = NULL;


// Constructor
GLG4Scint::MyPhysicsTable::MyPhysicsTable() {
    name          = 0;
    next          = 0;
    used_by_count = 0;
    data          = 0;
    length        = 0;
}

// Destructor
GLG4Scint::MyPhysicsTable::~MyPhysicsTable() {
    if (used_by_count != 0) {
        G4cerr << "Error, GLG4Scint::MyPhysicsTable is being deleted with "
               << "used_by_count = " << used_by_count << G4endl;
        return;
    }
    delete name;
    delete[] data;
}

// //////////////
// Member functions
// //////////////

void GLG4Scint::MyPhysicsTable::Dump() const {
    G4cout << " GLG4Scint::MyPhysicsTable {\n"
           << "  name=" << (*name) << G4endl
           << "  length=" << length << G4endl
           << "  used_by_count=" << used_by_count << G4endl;

    for (G4int i = 0; i < length; i++) {
        G4cout << "  data[" << i << "]= { // "
               << (*G4Material::GetMaterialTable())[i]->GetName() << G4endl;
        G4cout << "   spectrumIntegral=";

        if (data[i].spectrumIntegral) (data[i].spectrumIntegral)->DumpValues();
        else G4cout << "NULL" << G4endl;

        G4cout << "   timeIntegral=";

        if (data[i].timeIntegral) (data[i].timeIntegral)->DumpValues();
        else G4cout << "NULL" << G4endl;
        G4cout << "   resolutionScale=" << data[i].resolutionScale
               << "   birksConstant=" << data[i].birksConstant
               << "   ref_dE_dx=" << data[i].ref_dE_dx << G4endl
               << "   light yield=" << data[i].light_yield << G4endl;

        G4cout << "Quenching = \n";

        if (data[i].QuenchingArray != NULL) data[i].QuenchingArray->DumpValues();
        else G4cout << "NULL" << G4endl << "  }\n";
    }

    G4cout << " }\n";
}

GLG4Scint::MyPhysicsTable *
GLG4Scint::MyPhysicsTable::FindOrBuild(const G4String& name) {
    // Head should always exist and should always be the default (name=="")
    if (head == NULL) {
        head = new MyPhysicsTable;
        head->Build("");
    }

    MyPhysicsTable *rover = head;

    while (rover) {
        if (name == *(rover->name)) return rover;

        rover = rover->next;
    }

    rover = new MyPhysicsTable;
    rover->Build(name);
    rover->next = head->next; // Always keep head pointing to default
    head->next  = rover;

    return rover;
}

void GLG4Scint::MyPhysicsTable::Build(const G4String& newname) {
    delete name;
    delete[] data;

    // Name in the physics list, i.e. "" or "heavy" or "alpha" etc.
    // This is a suffix on material property vectors in RATDB
    name = new G4String(newname);

    const G4MaterialTable *theMaterialTable = G4Material::GetMaterialTable();
    length = G4Material::GetNumberOfMaterials();

    // vector of Entrys for everything in MATERIALS
    data = new Entry[length];

    // Create new physics tables
    for (G4int i = 0; i < length; i++) {
        const G4Material *aMaterial = (*theMaterialTable)[i];
        data[i].Build(*name, i, aMaterial->GetMaterialPropertiesTable());
    }
}

// Constructor for Entry
GLG4Scint::MyPhysicsTable::Entry::Entry() {
    I_own_spectrumIntegral = I_own_timeIntegral = false;
    resolutionScale        = 1.0;
    light_yield            = 0.0;
    DMsConstant            = birksConstant = ref_dE_dx = 0.0;
    QuenchingArray         = NULL;
}

// Destructor for Entry
GLG4Scint::MyPhysicsTable::Entry::~Entry() {
    if (I_own_spectrumIntegral) {
        delete spectrumIntegral;
    }

    if (I_own_timeIntegral) delete timeIntegral;

    delete QuenchingArray;
}

// Build for Entry
void GLG4Scint::MyPhysicsTable::Entry::Build(
    const G4String           & _name,
    int                        i,
    G4MaterialPropertiesTable *aMaterialPropertiesTable) {
    // Delete old data
    if (I_own_spectrumIntegral) {
        delete spectrumIntegral;
    }

    if (I_own_timeIntegral) {
        delete timeIntegral;
    }

    // Set defaults
    spectrumIntegral = timeIntegral = NULL;
    resolutionScale  = 1.0;
    birksConstant    = ref_dE_dx = 0.0;
    light_yield      = 0.0;
    QuenchingArray   = NULL;

    // Exit, leaving default values, if no material properties
    if (!aMaterialPropertiesTable) {
        return;
    }

    //aMaterialPropertiesTable->DumpTable();

    // Retrieve vector of scintillation wavelength intensity
    // for the material from the material's optical
    // properties table ("SCINTILLATION")
    std::stringstream property_string;

    property_string.str("");
    property_string << "SCINTILLATION" << _name;
    G4MaterialPropertyVector *theScintillationLightVector =
        aMaterialPropertiesTable->GetProperty(property_string.str().c_str());

    if (theScintillationLightVector) {
        // find the integral
        if (theScintillationLightVector == NULL) {
            spectrumIntegral = NULL;
        } else {
            spectrumIntegral = Integrate_MPV_to_POFV(theScintillationLightVector);
        }
        I_own_spectrumIntegral = true;
    } else {
        // Use default integral (possibly null)
        spectrumIntegral       = MyPhysicsTable::GetDefault()->GetEntry(i)->spectrumIntegral;
        I_own_spectrumIntegral = false;
    }
    
    property_string.str("");
    property_string << "LIGHT_YIELD" << _name;
    if (aMaterialPropertiesTable->ConstPropertyExists(property_string.str().c_str())) {
        light_yield = aMaterialPropertiesTable->GetConstProperty(property_string.str().c_str());
    } else {
        light_yield = MyPhysicsTable::GetDefault()->GetEntry(i)->light_yield;
    }

    // Retrieve vector of scintillation time profile
    // for the material from the material's optical
    // properties table ("SCINTWAVEFORM")
    property_string.str("");
    property_string << "SCINTWAVEFORM" << _name;
    G4MaterialPropertyVector *theWaveForm =
        aMaterialPropertiesTable->GetProperty(property_string.str().c_str());

    if (theWaveForm) {
        // Do we have time-series or decay-time data?
        if (theWaveForm->GetEnergy(0) >= 0.0) {
            // We have digitized waveform (time-series) data
            // Find the integral
            timeIntegral       = Integrate_MPV_to_POFV(theWaveForm);
            I_own_timeIntegral = true;
        }
        else {
            // We have decay-time data.
            // Sanity-check user's values:
            // Issue a warning if they are nonsense, but continue
            if (theWaveForm->Energy(theWaveForm->GetVectorLength() - 1) > 0.0) {
                G4cerr << "GLG4Scint::MyPhysicsTable::Entry::Build():  "
                       << "SCINTWAVEFORM" << _name
                       << " has both positive and negative X values.  "
                       << " Undefined results will ensue!\n";
            }

            G4double maxtime   = -3.0 * (theWaveForm->GetMaxEnergy());
            G4double mintime   = -1.0 * (theWaveForm->GetEnergy(0));
            G4double bin_width = mintime / 100;
            int nbins          = (int)(maxtime / bin_width) + 1;
            G4double *tval     = new G4double[nbins];
            G4double *ival     = new G4double[nbins];
            {
                for (int ii = 0; ii < nbins; ii++) {
                    tval[ii] = ii * maxtime / nbins;
                    ival[ii] = 0.0;
                }
            }

            for (unsigned int j = 0; j < theWaveForm->GetVectorLength(); j++) {
                G4double ampl = (*theWaveForm)[j];
                G4double decy = theWaveForm->Energy(j);
                {
                    for (int ii = 0; ii < nbins; ii++) {
                        ival[ii] += ampl * (1.0 - exp(tval[ii] / decy));
                    }
                }
            }

            {
                for (int ii = 0; ii < nbins; ii++) {
                    ival[ii] /= ival[nbins - 1];
                }
            }

            timeIntegral       = new G4PhysicsOrderedFreeVector(tval, ival, nbins);
            I_own_timeIntegral = true;

            // in Geant4.0.0, G4PhysicsOrderedFreeVector makes its own copy
            // of any array passed to its constructor, so ...
            delete[] tval;
            delete[] ival;
        }
    }
    else {
        // Use default integral (possibly null)
        timeIntegral       = MyPhysicsTable::GetDefault()->GetEntry(i)->timeIntegral;
        I_own_timeIntegral = false;
    }

    // Retrieve vector of scintillation "modifications"
    // for the material from the material's optical
    // properties table ("SCINTMOD")
    property_string.str("");
    property_string << "SCINTMOD" << _name;
    G4MaterialPropertyVector *theScintModVector =
        aMaterialPropertiesTable->GetProperty(property_string.str().c_str());

    if (theScintModVector == NULL) {
        // Use default if not particle-specific value given
        theScintModVector =
            aMaterialPropertiesTable->GetProperty("SCINTMOD");
    }

    if (theScintModVector) {
        // Parse the entries in ScintMod
        // ResolutionScale= ScintMod(0);
        // BirksConstant= ScintMod(1);
        // Ref_dE_dx= ScintMod(2);
        for (unsigned int ii = 0; ii < theScintModVector->GetVectorLength(); ii++) {
            G4double key   = theScintModVector->Energy(ii);
            G4double value = (*theScintModVector)[ii];

            if (key == 0) {
                resolutionScale = value;
            }
            else if (key == 1) {
                birksConstant = value;
            }
            else if (key == 2) {
                ref_dE_dx = value;
            }
            else {
                G4cerr << "GLG4Scint::MyPhysicsTable::Entry::Build"
                       << ":  Warning, unknown key " << key
                       << "in SCINTMOD" << _name << G4endl;
            }
        }
    }

    property_string.str("");
    property_string << "QF" << _name;
    QuenchingArray = aMaterialPropertiesTable->GetProperty(property_string.str().c_str());
}

void GLG4Scint::SetNewValue(G4UIcommand *command, G4String newValues) {
    G4String commandName = command->GetCommandName();

    if (commandName == "on") {
        doScintillation = true;
    }
    else if (commandName == "off") {
        doScintillation = false;
    }
    else if (commandName == "maxTracksPerStep") {
        G4int i = strtol((const char *)newValues, NULL, 0);

        if (i > 0) {
            maxTracksPerStep = i;
        }
        else {
            G4cerr << "Value must be greater than 0, old value unchanged" << G4endl;
        }
    }
    else if (commandName == "meanPhotonsPerSecondary") {
        G4double d = strtod((const char *)newValues, NULL);

        if (d >= 1.0) {
            meanPhotonsPerSecondary = d;
        }
        else {
            G4cerr << "Value must be >= 1.0, old value unchanged" << G4endl;
        }
    }
    else if (commandName == "verbose") {
        // Sets same verbosity for all tables
        for (unsigned int i = 0; i < masterVectorOfGLG4Scint.size();
             i++) masterVectorOfGLG4Scint[i]->SetVerboseLevel(strtol((const char *)newValues, NULL, 0));
    }
    else if (commandName == "dump") {
        std::vector<GLG4Scint *>::iterator it = masterVectorOfGLG4Scint.begin();

        for (; it != masterVectorOfGLG4Scint.end(); it++) {
            (*it)->DumpInfo();
        }
    }
    else if (commandName == "setQF") {
        G4double d = strtod((const char *)newValues, NULL);

        if (d <= 1.0) {
            SetQuenchingFactor(d);
            UserQF = true;
        }
        else {
            G4cerr << "The quenching factor is <= 1.0, old value unchanged" << G4endl;
        }
    }
    else {
        G4cerr << "No GLG4Scint command named " << commandName << G4endl;
    }
}

G4String GLG4Scint::GetCurrentValue(G4UIcommand *command) {
    G4String commandName = command->GetCommandName();

    if ((commandName == "on") || (commandName == "off")) {
        return doScintillation ? "on" : "off";
    }
    else if (commandName == "maxTracksPerStep") {
        char outbuff[64];
        sprintf(outbuff, "%d", maxTracksPerStep);
        return G4String(outbuff);
    }
    else if (commandName == "meanPhotonsPerSecondary") {
        char outbuff[64];
        sprintf(outbuff, "%g", meanPhotonsPerSecondary);
        return G4String(outbuff);
    }
    else if (commandName == "verbose") {
        char outbuff[64];
        sprintf(outbuff, "%d", verboseLevel);
        return G4String(outbuff);
    }
    else if (commandName == "dump") {
        return "?/glg4scint/dump not supported";
    }
    else if (commandName == "setQF") {
        char outbuff[64];
        sprintf(outbuff, "%g", GetQuenchingFactor());
        return G4String(outbuff);
    }
    else {
        return commandName + " is not a valid GLG4Scint command";
    }
}
