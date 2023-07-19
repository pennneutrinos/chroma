#ifndef __G4chroma_hh__
#define __G4chroma_hh__

#include <G4VModularPhysicsList.hh>
class ChromaPhysicsList: public G4VModularPhysicsList
{
public:
  ChromaPhysicsList();
  virtual ~ChromaPhysicsList();
  virtual void SetCuts();
};

#include <G4UserSteppingAction.hh>
#include <G4UserTrackingAction.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <vector>
#include <map>

class Step {
public:
    inline Step(const double _x, const double _y, const double _z, 
                const double _t, 
                const double _dx, const double _dy, const double _dz, 
                const double _ke, const double _edep, const double _qedep, 
                const std::string &_procname) :
                x(_x), y(_y), z(_z), t(_t), dx(_dx), dy(_dy), dz(_dz), 
                ke(_ke), edep(_edep), qedep(_qedep), procname(_procname) {
    }
    
    inline ~Step() { }

    const double x,y,z,t,dx,dy,dz,ke,edep,qedep;
    const std::string procname;
};

class Track {
public:
    inline Track() : id(-1) { }
    inline ~Track() { }
    
    int id, parent_id, pdg_code;
    double weight;
    std::string name;
    
    void appendStepPoint(const G4StepPoint* point, const G4Step* step, double qedep, const bool initial = false);  
    inline const std::vector<Step>& getSteps() { return steps; };  
    int getNumSteps();
    
    inline int getNumChildren() { return children.size(); }
    inline int getChildTrackID(int i) { return children[i]; }
    
    inline void addChild(int trackid) { children.push_back(trackid); }
    
private:
    std::vector<Step> steps;
    std::vector<int> children;
};

class SteppingAction : public G4UserSteppingAction
{
public:
  SteppingAction();
  virtual ~SteppingAction();
  
  void EnableScint(bool enabled);
  void EnableTracking(bool enabled);
  
  void UserSteppingAction(const G4Step* aStep);
  
  void ClearTracking();
  Track& getTrack(int id);
  
private:
    bool scint;
    
    bool tracking;
    bool children_mapped;
    void mapChildren();
    std::map<int,Track> trackmap;

};

class TrackingAction : public G4UserTrackingAction
{
public:
  TrackingAction();
  virtual ~TrackingAction();
  
  int GetNumPhotons() const;
  void Clear();
  
  void GetX(double *x) const;
  void GetY(double *y) const;
  void GetZ(double *z) const;
  void GetDirX(double *dir_x) const;
  void GetDirY(double *dir_y) const;
  void GetDirZ(double *dir_z) const;
  void GetPolX(double *pol_x) const;
  void GetPolY(double *pol_y) const;
  void GetPolZ(double *pol_z) const;

  void GetWavelength(double *wl) const;
  void GetT0(double *t) const;
  
  void GetParentTrackID(int *t) const;
  void GetFlags(uint32_t *flags) const;

  virtual void PreUserTrackingAction(const G4Track *);

protected:
  std::vector<G4ThreeVector> pos;
  std::vector<G4ThreeVector> dir;
  std::vector<G4ThreeVector> pol;
  std::vector<int> parentTrackID;
  std::vector<uint32_t> flags;
  std::vector<double> wavelength;
  std::vector<double> t0;
};

#endif
