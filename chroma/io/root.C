#include <TVector3.h>
#include <vector>
#include <TTree.h>
#include <string>
#include <map>

struct Vertex {
  virtual ~Vertex() { };

  std::string particle_name;
  TVector3 pos;
  TVector3 dir;
  TVector3 pol;
  double ke;
  double t0;
  int trackid;
  int pdgcode;
  
  std::vector<Vertex> children;
  std::vector<double> step_x,step_y,step_z,step_t,step_dx,step_dy,step_dz,step_ke,step_edep,step_qedep;

  ClassDef(Vertex, 1);
};

struct Photon {
  virtual ~Photon() { };

  double t;
  TVector3 pos;
  TVector3 dir;
  TVector3 pol;
  double wavelength; // nm
  unsigned int flag;
  int last_hit_triangle;
  int channel;

  ClassDef(Photon, 1);
};

struct Channel {
  Channel() : id(-1), t(-1e9), q(-1e9) { };
  virtual ~Channel() { };

  int id;
  double t;
  double q;
  unsigned int flag;

  ClassDef(Channel, 1);
};

struct Event {
  virtual ~Event() { };

  int id;
  unsigned int nhit;
  unsigned int nchannels;
  
  double TotalQ() const {
    double sum = 0.0;
    for (unsigned int i=0; i < channels.size(); i++)
      sum += channels[i].q;
    return sum;
  }


  std::vector<Vertex> vertices;
  std::vector<Photon> photons_beg;
  std::vector<Photon> photons_end;
  std::vector<std::vector<Photon>> photon_tracks;
  std::vector<int> photon_parent_trackids;
  std::map<int,std::vector<Photon>> hits;
  std::vector<Photon> flat_hits;
  std::vector<Channel> channels;

  ClassDef(Event, 1);
};

void clear_steps(Vertex *vtx) {
  vtx->step_x.resize(0);
  vtx->step_y.resize(0);
  vtx->step_z.resize(0);
  vtx->step_t.resize(0);
  vtx->step_dx.resize(0);
  vtx->step_dy.resize(0);
  vtx->step_dz.resize(0);
  vtx->step_ke.resize(0);
  vtx->step_edep.resize(0);
  vtx->step_qedep.resize(0);
}

void fill_steps(Vertex *vtx, unsigned int nsteps, double *x, double *y, double *z,
        double *t, double *dx, double *dy, double *dz, double *ke, double *edep, double *qedep) {
  vtx->step_x.resize(nsteps);
  vtx->step_y.resize(nsteps);
  vtx->step_z.resize(nsteps);
  vtx->step_t.resize(nsteps);
  vtx->step_dx.resize(nsteps);
  vtx->step_dy.resize(nsteps);
  vtx->step_dz.resize(nsteps);
  vtx->step_ke.resize(nsteps);
  vtx->step_edep.resize(nsteps);
  vtx->step_qedep.resize(nsteps);
  for (unsigned int i=0; i < nsteps; i++) {
      vtx->step_x[i] = x[i];
      vtx->step_y[i] = y[i];
      vtx->step_z[i] = z[i];
      vtx->step_t[i] = t[i];
      vtx->step_dx[i] = dx[i];
      vtx->step_dy[i] = dy[i];
      vtx->step_dz[i] = dz[i];
      vtx->step_ke[i] = ke[i];
      vtx->step_edep[i] = edep[i];
      vtx->step_qedep[i] = qedep[i];
  }
}

void get_steps(Vertex *vtx, unsigned int nsteps, double *x, double *y, double *z,
        double *t, double *dx, double *dy, double *dz, double *ke, double *edep, double *qedep) {
  for (unsigned int i=0; i < nsteps; i++) {
      x[i] = vtx->step_x[i];
      y[i] = vtx->step_y[i];
      z[i] = vtx->step_z[i];
      t[i] = vtx->step_t[i];
      dx[i] = vtx->step_dx[i];
      dy[i] = vtx->step_dy[i];
      dz[i] = vtx->step_dz[i];
      ke[i] = vtx->step_ke[i];
      edep[i] = vtx->step_edep[i];
      qedep[i] = vtx->step_qedep[i];
  }
}

void fill_channels(Event *ev, unsigned int nhit, unsigned int *hit_id, 
           unsigned int nchannels, float *t, float *q, unsigned int *flags)
{
  ev->nhit = nhit;
  ev->nchannels = nchannels;
  ev->channels.resize(nhit);

  for (unsigned int i = 0; i < nhit; i++) {
      Channel *ch = &ev->channels[i];
      unsigned int id = hit_id[i];
      ch->id = id;
      ch->t = t[id];
      ch->q = q[id];
      ch->flag = flags[id];
  }
}

void get_channels(Event *ev, int *hit, float *t, float *q, unsigned int *flags)
{
  for (unsigned int i=0; i < ev->channels.size(); i++) {
    unsigned int id = ev->channels[i].id;
    hit[id] = 1;
    t[id] = ev->channels[i].t;
    q[id] = ev->channels[i].q;
    flags[id] = ev->channels[i].flag;
  }
}

void get_photons(const std::vector<Photon> &photons, float *pos, float *dir,
		 float *pol, float *wavelengths, float *t,
		 int *last_hit_triangles, unsigned int *flags, unsigned int *channels)
{
  for (unsigned int i=0; i < photons.size(); i++) {
    const Photon &photon = photons[i];
    pos[3*i] = photon.pos.X();
    pos[3*i+1] = photon.pos.Y();
    pos[3*i+2] = photon.pos.Z();

    dir[3*i] = photon.dir.X();
    dir[3*i+1] = photon.dir.Y();
    dir[3*i+2] = photon.dir.Z();
    
    pol[3*i] = photon.pol.X();
    pol[3*i+1] = photon.pol.Y();
    pol[3*i+2] = photon.pol.Z();

    wavelengths[i] = photon.wavelength;
    t[i] = photon.t;
    flags[i] = photon.flag;
    last_hit_triangles[i] = photon.last_hit_triangle;
    channels[i] = photon.channel;
  }
}
		 
void fill_photons(std::vector<Photon> &photons,
		  unsigned int nphotons, float *pos, float *dir,
		  float *pol, float *wavelengths, float *t,
		  int *last_hit_triangles, unsigned int *flags,
		  unsigned int *channels)
{
  photons.resize(nphotons);
  
  for (unsigned int i=0; i < nphotons; i++) {
    Photon &photon = photons[i];
    photon.t = t[i];
    photon.pos.SetXYZ(pos[3*i], pos[3*i + 1], pos[3*i + 2]);
    photon.dir.SetXYZ(dir[3*i], dir[3*i + 1], dir[3*i + 2]);
    photon.pol.SetXYZ(pol[3*i], pol[3*i + 1], pol[3*i + 2]);
    photon.wavelength = wavelengths[i];
    photon.last_hit_triangle = last_hit_triangles[i];
    photon.flag = flags[i];
    photon.channel = channels[i];

  }
}

#ifdef __ROOTCLING__
#pragma link C++ class Vertex+;
#pragma link C++ class Photon+;
#pragma link C++ class Event+;
#pragma link C++ class std::vector<Vertex>+;
#pragma link C++ class std::vector<Photon>+;
#pragma link C++ class std::vector<Channel>+;
#pragma link C++ class std::map<int,std::vector<Photon>>+;
#endif


