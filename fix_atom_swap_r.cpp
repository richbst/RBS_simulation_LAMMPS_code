/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Paul Crozier (SNL)
                         Alexander Stukowski
------------------------------------------------------------------------- */

/*----------------------<><><>----------------------------
  Added optional capabilities
    Limit distance or type separation between swap pairs
      by forcing to pick again if separation too large
      uses keyword 'dist_sep' which takes max_swap_sep >= 1.0
                   optional keyword before separation 'x', 'y', 'z', 'xy', 'xz', or 'yz' to limit restriction to specified dimension(s)
      or 'type_sep' which takes max_typ_sep > 1
    Random choice among large number of swap types
      uses keyword 'poly' which then requires more than two types after the 'type' keyword
    Record distribution of swap pair separations (only in conjunction with type_sep or dist_sep)
      uses keyword 'distribution' with values max_sep value and sep_no (the number of bins up to that max. Generates 2n + 1 numbers covering -max_sep to +max_sep. positive(negative) for accepted(unaccepted) swaps and zero if no satisfactory pair found.
  added subroutine:
     int swap_sep() - returns -1 if proposed swap separation larger than max_swap_sep
  added variables:
     int distance_flag to invoke max separation limit
     int x_flag, y_flag, z_flag to specify axes for limit
     int distrib_flag to return  type/distance separations for swaps
     int typ_flag to invoke max type diff limit
     int poly_flag to allow random choice among many atom types
     int ipick, jpick to indicate first and second choice
     int itype, jtype, iqtype, jqtype placeholders used for swapping (inside attempt_swap)
     double max_swap_sep to specify max allowed swap pair distance or type separation
     array swap_sep_dist, len 2*sep_no + 1 to hold the distance or type separation distribution
 added a few error messages
 
   Rich Stephens November 2021
 ----------------------<><><>----------------------------*/

#include "fix_atom_swap.h"

#include <cmath>
#include <cctype>
#include <cfloat>
#include <cstring>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "compute.h"
#include "group.h"
#include "domain.h"
#include "region.h"
#include "random_park.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "error.h"
#include "neighbor.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */
/* ---<><><>---
 added swp_sep_dist and typ_sep_dist to contain record of swap separation distribution
 - array size 2*sep_no + 1 (if distrib_flag)
 ---<><><>---*/
FixAtomSwap::FixAtomSwap(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  idregion(nullptr), type_list(nullptr), mu(nullptr), qtype(nullptr),
  sqrt_mass_ratio(nullptr), local_swap_iatom_list(nullptr),
  local_swap_jatom_list(nullptr), local_swap_atom_list(nullptr),
  random_equal(nullptr), random_unequal(nullptr), c_pe(nullptr)
{
  if (narg < 10) error->all(FLERR,"Illegal fix atom/swap:narg command");

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 43;  //---<><><>--- this swap vector length - 43 allows sep_no up to 20
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // required args

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  ncycles = utils::inumeric(FLERR,arg[4],false,lmp);
  seed = utils::inumeric(FLERR,arg[5],false,lmp);
  double temperature = utils::numeric(FLERR,arg[6],false,lmp);
  beta = 1.0/(force->boltz*temperature);

  if (nevery <= 0) error->all(FLERR,"Illegal fix atom/swap:nevery command");
  if (ncycles < 0) error->all(FLERR,"Illegal fix atom/swap:ncycles command");
  if (seed <= 0) error->all(FLERR,"Illegal fix atom/swap:seed command");

  memory->create(type_list,atom->ntypes,"atom/swap:type_list");
  memory->create(mu,atom->ntypes+1,"atom/swap:mu");
  for (int i = 1; i <= atom->ntypes; i++) mu[i] = 0.0;

  // read options from end of input line

  options(narg-7,&arg[7]);

  // random number generator, same for all procs

  random_equal = new RanPark(lmp,seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp,seed);

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // zero out counters

  nswap_attempts = 0.0;
  nswap_successes = 0.0;

  atom_swap_nmax = 0;
  local_swap_atom_list = nullptr;
  local_swap_iatom_list = nullptr;
  local_swap_jatom_list = nullptr;
  swap_sep_dist = nullptr;   //---<><><>---

  // set comm size needed by this Fix

  if (atom->q_flag) comm_forward = 2;
  else comm_forward = 1;

}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixAtomSwap::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR,"Illegal fix atom/swap:narg command");

  regionflag = 0;
  conserve_ke_flag = 1;
  distance_flag = 0;           // ---<><><>--- swap separation restriction
  x_flag = 1;                  // ---<><><>--- restrict on x axis
  y_flag = 1;                  // ---<><><>--- restrict on y axis
  z_flag = 1;                  // ---<><><>--- restrict on z axis
  typ_flag = 0;                // ---<><><>--- type separation restriction
  distrib_flag = 0;            // ---<><><>--- record distance/type separation for swaps
  poly_flag = 0;               // ---<><><>--- allow random j-type choice from type list
  semi_grand_flag = 0;
  nswaptypes = 0;
  nmutypes = 0;
  iregion = -1;
    
  /*-----------------<><><>----------------------
   key word 'distribution' added - value >= 1.0
   key word 'dist_sep' added - values max_sep > 0.0 and sep_no > 1
   key word 'type_sep' added - values max_sep > 1 and sep_no > 1
             (either 'typesep' or 'dist' but not both)
   key word 'poly' added - no value
   ------------------<><><>---------------------*/

  int iarg = 0;
    while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix atom/swap:region command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix atom/swap does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      regionflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"ke") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix atom/swap:ke command");
      if (strcmp(arg[iarg+1],"no") == 0) conserve_ke_flag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) conserve_ke_flag = 1;
      else error->all(FLERR,"Illegal fix atom/swap:ke y/n command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"semi-grand") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix atom/swap:semi-grand command");
      if (strcmp(arg[iarg+1],"no") == 0) semi_grand_flag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) semi_grand_flag = 1;
      else error->all(FLERR,"Illegal fix atom/swap:semi-grand y/n command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"dist_sep") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
      distance_flag = 1;
      if (!(strcmp(arg[iarg+1],"x") || strcmp(arg[iarg+1],"y") || strcmp(arg[iarg+1],"z") || strcmp(arg[iarg+1],"xy") || strcmp(arg[iarg+1],"xz") || strcmp(arg[iarg+1],"yz"))) {
        if (isalpha(arg[iarg+1][0])) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
         max_swap_sep= utils::numeric(FLERR,arg[iarg+1],false,lmp);
         if (max_swap_sep < 1.0)
            error->all(FLERR,"Swap pair distance separation limit must be >= 1.0");
         iarg += 2;
      }
      else if (strcmp(arg[iarg+1],"x")) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        if (isalpha(arg[iarg+2][0])) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        max_swap_sep= utils::numeric(FLERR,arg[iarg+2],false,lmp);
        y_flag = 0;
        z_flag = 0;
        if (max_swap_sep < 1.0)
           error->all(FLERR,"Swap pair distance x separation limit must be >= 1.0");
        iarg += 3;
      }
      else if (strcmp(arg[iarg+1],"y")) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        if (isalpha(arg[iarg+2][0])) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        max_swap_sep= utils::numeric(FLERR,arg[iarg+2],false,lmp);
        x_flag = 0;
        z_flag = 0;
        if (max_swap_sep < 1.0)
           error->all(FLERR,"Swap pair distance y separation limit must be >= 1.0");
        iarg += 3;
      }
      else if (strcmp(arg[iarg+1],"z")) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        if (isalpha(arg[iarg+2][0])) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        max_swap_sep= utils::numeric(FLERR,arg[iarg+2],false,lmp);
        x_flag = 0;
        y_flag = 0;
        if (max_swap_sep < 1.0)
           error->all(FLERR,"Swap pair distance z separation limit must be >= 1.0");
        iarg += 3;
      }
      else if (strcmp(arg[iarg+1],"xy")) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        if (isalpha(arg[iarg+2][0])) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        max_swap_sep= utils::numeric(FLERR,arg[iarg+2],false,lmp);
        z_flag = 0;
        if (max_swap_sep < 1.0)
           error->all(FLERR,"Swap pair distance xy separation limit must be >= 1.0");
        iarg += 3;
      }
      else if (strcmp(arg[iarg+1],"xz")) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        if (isalpha(arg[iarg+2][0])) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        max_swap_sep= utils::numeric(FLERR,arg[iarg+2],false,lmp);
        y_flag = 0;
        if (max_swap_sep < 1.0)
           error->all(FLERR,"Swap pair distance xz separation limit must be >= 1.0");
        iarg += 3;
      }
      else if (strcmp(arg[iarg+1],"yz")) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        if (isalpha(arg[iarg+2][0])) error->all(FLERR,"Illegal fix atom/swap:dist_sep command");
        max_swap_sep= utils::numeric(FLERR,arg[iarg+2],false,lmp);
        x_flag = 0;
        if (max_swap_sep < 1.0)
           error->all(FLERR,"Swap pair distance yz separation limit must be >= 1.0");
        iarg += 3;
      }
    } else if (strcmp(arg[iarg],"type_sep") == 0) {
      std::string mystring2 = "recognized type_sep \n";
      utils::logmesg(lmp,mystring2);
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix atom/swap:type_sep command");
      if (isalpha(arg[iarg+1][0])) error->all(FLERR,"Illegal fix atom/swap:type_sep command");
      typ_flag = 1;
      max_swap_sep= utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (max_swap_sep <= 1)
            error->all(FLERR,"Swap pair distance/type separation limit must be > 1");
      iarg += 2;
    } else if (strcmp(arg[iarg],"distribution") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix atom/swap:distribution command");
      max_sep = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (max_sep <= 0) error->all(FLERR,"Illegal fix atom/swap:distribution command");
      sep_no = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      if ((sep_no <= 1) || (sep_no > 20)) error->all(FLERR,"Illegal fix atom/swap:distribution command");
      distrib_flag = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg],"poly") == 0) {
      std::string mystring2 = "recognized poly \n";
      utils::logmesg(lmp,mystring2);
      poly_flag =1;
      iarg++;
    } else if (strcmp(arg[iarg],"types") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix atom/swap:types command");
      iarg++;
      while (iarg < narg) {
        if (isalpha(arg[iarg][0])) break;
        if (nswaptypes >= atom->ntypes) error->all(FLERR,"Illegal fix atom/swap:types command");
        type_list[nswaptypes] = utils::numeric(FLERR,arg[iarg],false,lmp);
        nswaptypes++;
        iarg++;
      }
    } else if (strcmp(arg[iarg],"mu") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix atom/swap:mu command");
      iarg++;
      while (iarg < narg) {
        if (isalpha(arg[iarg][0])) break;
        nmutypes++;
        if (nmutypes > atom->ntypes) error->all(FLERR,"Illegal fix atom/swap:mu command");
        mu[nmutypes] = utils::numeric(FLERR,arg[iarg],false,lmp);
        iarg++;
      }
    } else error->all(FLERR,"Illegal fix atom/swap:not recognized command");
  }
}

/* ---------------------------------------------------------------------- */

FixAtomSwap::~FixAtomSwap()
{
  memory->destroy(type_list);
  memory->destroy(mu);
  memory->destroy(qtype);
  memory->destroy(sqrt_mass_ratio);
  if (distrib_flag == 1) {
    memory->destroy(swap_sep_dist);
  }
  if (regionflag) delete [] idregion;
  delete random_equal;
  delete random_unequal;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwap::init()
{
  char *id_pe = (char *) "thermo_pe";
  int ipe = modify->find_compute(id_pe);
  c_pe = modify->compute[ipe];

  int *type = atom->type;

  if (nswaptypes < 2)
    error->all(FLERR,"Must specify at least 2 types in fix atom/swap command");
// ---<><><>--- require more than 2 type if the random keyword is used
  if (semi_grand_flag) {
    if (nswaptypes != nmutypes)
      error->all(FLERR,"Need nswaptypes mu values in fix atom/swap command");
  } else {
    if (nswaptypes == 2 && poly_flag == 1)
      error->all(FLERR,"More than 2 types required when using poly in fix atom/swap command");
    if (nswaptypes != 2 && poly_flag == 0)
      error->all(FLERR,"Only 2 types allowed when not using semi-grand or poly in fix atom/swap command");
    if (nmutypes != 0)
      error->all(FLERR,"Mu not allowed when not using semi-grand in fix atom/swap command");
  }
  // ---<><><>--- type_sep and dist_sep routines can't be used simultaneously
  if ((typ_flag == 1) && (distance_flag == 1)) {
    error->all(FLERR,"Can't use both dist_sep and type_sep key words");
  }
  // ---<><><>--- distrib_flag can only be used in conjunction with typ_flag or distance_flag
  if ((distrib_flag == 1) && not((typ_flag == 1) || (distance_flag == 1))) {
    error->all(FLERR,"Must use 'dist_sep' or 'type_sep' with 'distribution'");
  }

  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    if (type_list[iswaptype] <= 0 || type_list[iswaptype] > atom->ntypes)
      error->all(FLERR,"Invalid atom type in fix atom/swap command");
  // ---<><><>--- allows j_pick subroutine to see what the i_pick selected
  //              and the neighbor subroutine to report on selections
  int ipick = 0;
  int jpick = 0;
  int iatomtype = 0;
  int jatomtype = 0;

  // this is only required for non-semi-grand
  // in which case, nswaptypes = 2
  //---<><><>--- nswaptypes > 2 if poly keyword is used

  if (atom->q_flag && !semi_grand_flag) {
    double qmax,qmin;
    int firstall,first;
    memory->create(qtype,nswaptypes,"atom/swap:qtype");
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      first = 1;
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[iswaptype]) {
            if (first) {
              qtype[iswaptype] = atom->q[i];
              first = 0;
            } else if (qtype[iswaptype] != atom->q[i])
              error->one(FLERR,"All atoms of a swapped type must have the same charge.");
          }
        }
      }
      MPI_Allreduce(&first,&firstall,1,MPI_INT,MPI_MIN,world);
      if (firstall) error->all(FLERR,"At least one atom of each swapped type must be present to define charges.");
      if (first) qtype[iswaptype] = -DBL_MAX;
      MPI_Allreduce(&qtype[iswaptype],&qmax,1,MPI_DOUBLE,MPI_MAX,world);
      if (first) qtype[iswaptype] = DBL_MAX;
      MPI_Allreduce(&qtype[iswaptype],&qmin,1,MPI_DOUBLE,MPI_MIN,world);
      if (qmax != qmin) error->all(FLERR,"All atoms of a swapped type must have same charge.");
    }
  }
  
  // ---<><><>---if swap distribution flag is set, create array to hold distribution
  if (distrib_flag == 1) {
    int nsize = 2*sep_no+1;
    memory->create(swap_sep_dist,nsize,"atom/swap:swap_sep_dist");
    for (int i=0;i<nsize;i++) {
      swap_sep_dist[i] = 0;
    }
  }

  memory->create(sqrt_mass_ratio,atom->ntypes+1,atom->ntypes+1,"atom/swap:sqrt_mass_ratio");
  for (int itype = 1; itype <= atom->ntypes; itype++)
    for (int jtype = 1; jtype <= atom->ntypes; jtype++)
      sqrt_mass_ratio[itype][jtype] = sqrt(atom->mass[itype]/atom->mass[jtype]);

  // check to see if itype and jtype cutoffs are the same
  // if not, reneighboring will be needed between swaps

  double **cutsq = force->pair->cutsq;
  unequal_cutoffs = false;
  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++)
      for (int ktype = 1; ktype <= atom->ntypes; ktype++)
        if (cutsq[type_list[iswaptype]][ktype] != cutsq[type_list[jswaptype]][ktype])
          unequal_cutoffs = true;

  // check that no swappable atoms are in atom->firstgroup
  // swapping such an atom might not leave firstgroup atoms first

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

    if (flagall)
      error->all(FLERR,"Cannot do atom/swap on atoms in atom_modify first group");
  }
}

/* ----------------------------------------------------------------------
   attempt Monte Carlo swaps
------------------------------------------------------------------------- */

void FixAtomSwap::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);

  energy_stored = energy_full();

  int nsuccess = 0;
  if (semi_grand_flag) {
    update_semi_grand_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_semi_grand();
  } else {
    update_swap_atoms_list();
    for (int i = 0; i < ncycles; i++) {
      nsuccess += attempt_swap();
      int sndx = 0;
      if (distrib_flag == 1) {
        sndx = int((sw_sep_val/max_sep + 1.0)*sep_no+0.49);
        int nsize = 2*sep_no+1;
        if (sndx >= nsize) sndx= nsize-1;
        if (sndx < 0) sndx = 0;
        /* ---<><><>---
        std::string mystring2 = "adding 1 to swap_sep_dist("+std::to_string(sndx)+"), swap types:"+std::to_string(iatomtype)+", "+std::to_string(jatomtype)+" \n";
        utils::logmesg(lmp,mystring2);
         ---<><><>---*/
        swap_sep_dist[sndx] ++;
      }
    }
  }
  // ---<><><>--- record distribution of swap pair separations in swap_sep_dist
  nswap_attempts += ncycles;
  nswap_successes += nsuccess;
  energy_full();
  next_reneighbor = update->ntimestep + nevery;
}

/* ----------------------------------------------------------------------
Note: atom charges are assumed equal and so are not updated
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_semi_grand()
{
  if (nswap == 0) return 0;

  double energy_before = energy_stored;

  int itype,jtype,jswaptype;
  int i = pick_semi_grand_atom();
  if (i >= 0) {
    jswaptype = static_cast<int> (nswaptypes*random_unequal->uniform());
    jtype = type_list[jswaptype];
    itype = atom->type[i];
    while (itype == jtype) {
      jswaptype = static_cast<int> (nswaptypes*random_unequal->uniform());
      jtype = type_list[jswaptype];
    }
    atom->type[i] = jtype;
  }

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm_fix(this);
  }

  if (force->kspace) force->kspace->qsum_qsq();
  double energy_after = energy_full();

  int success = 0;
  if (i >= 0)
    if (random_unequal->uniform() <
      exp(beta*(energy_before - energy_after
            + mu[jtype] - mu[itype]))) success = 1;

  int success_all = 0;
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);

  if (success_all) {
    update_semi_grand_atoms_list();
    energy_stored = energy_after;
    if (conserve_ke_flag) {
      if (i >= 0) {
        atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
      }
    }
    return 1;
  } else {
    if (i >= 0) {
      atom->type[i] = itype;
    }
    if (force->kspace) force->kspace->qsum_qsq();
    energy_stored = energy_before;

    if (unequal_cutoffs) {
      if (domain->triclinic) domain->x2lamda(atom->nlocal);
      domain->pbc();
      comm->exchange();
      comm->borders();
      if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
      if (modify->n_pre_neighbor) modify->pre_neighbor();
      neighbor->build(1);
    } else {
      comm->forward_comm_fix(this);
    }
  }
  return 0;
}


/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_swap()
{
  sw_sep_val = 0.0;
  if ((niswap == 0) || (njswap == 0)) return 0;

  int itype, jtype, iqtype, jqtype;
  double energy_before = energy_stored;

  int i = pick_i_swap_atom();
  ipick = i;                    //---<><><>---ipick used in pick_j_swap_atom
  itype = atom->type[i];
  iatomtype = itype;            // ---<><><>---iatomtype used in j_pick to choose j_type, and in neighbor to report selection
  int j = pick_j_swap_atom();
  if (j == -1) {
    sw_sep_val = 0.0;      // ---<><><>--- unsuccessful at finding a suitable j-atom
    return 0;
  }
  jtype = atom->type[j];
  jatomtype = jtype;
    
  /*---<><><>---get type, q from the selected atoms rather than type_list[0] and type_list[1]
                here and after energy_full, if the switch is undone*/
  if (i >= 0) {
    itype = atom->type[i];
    if (atom->q_flag) iqtype = atom->q[i];
  }
  if (j >= 0) {
    jtype = atom->type[j];
    if (atom->q_flag) jqtype = atom->q[j];
  }
  if (i >= 0) {
    atom->type[i] = jtype;
    if (atom->q_flag) atom->q[i] = jqtype;
  }
  if (j >= 0) {
    atom->type[j] = itype;
    if (atom->q_flag) atom->q[j] = iqtype;
  }

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm_fix(this);
  }

  double energy_after = energy_full();
/* ---<><><>---
 if distrib_flag is set then set sw_sep_val
 = 0 if no pair selected, the separation if successful, or its negative if not
 ---<><><>--- */
  if (random_equal->uniform() <
      exp(beta*(energy_before - energy_after))) {
    update_swap_atoms_list();
    energy_stored = energy_after;
    if (conserve_ke_flag) {
      if (i >= 0) {
        atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
      }
      if (j >= 0) {
        atom->v[j][0] *= sqrt_mass_ratio[jtype][itype];
        atom->v[j][1] *= sqrt_mass_ratio[jtype][itype];
        atom->v[j][2] *= sqrt_mass_ratio[jtype][itype];
      }
    }
    return 1;
  } else {
    if (i >= 0) {
      atom->type[i] =  itype;
      if (atom->q_flag) atom->q[i] = iqtype;
    }
    if (j >= 0) {
      atom->type[j] =  jtype;
      if (atom->q_flag) atom->q[j] = jqtype;
    }
    energy_stored = energy_before;

    if (unequal_cutoffs) {
      if (domain->triclinic) domain->x2lamda(atom->nlocal);
      domain->pbc();
      comm->exchange();
      comm->borders();
      if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
      if (modify->n_pre_neighbor) modify->pre_neighbor();
      neighbor->build(1);
    } else {
      comm->forward_comm_fix(this);
    }
  }
  sw_sep_val = -sw_sep_val; // pair was picked, but swap unsuccessful
  return 0;
}

/* ----------------------------------------------------------------------
   compute system potential energy
------------------------------------------------------------------------- */

double FixAtomSwap::energy_full()
{
  int eflag = 1;
  int vflag = 0;

  if (modify->n_pre_neighbor) modify->pre_neighbor();
  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag,vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) force->kspace->compute(eflag,vflag);

  if (modify->n_post_force) modify->post_force(vflag);
  if (modify->n_end_of_step) modify->end_of_step();

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  return total_energy;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_semi_grand_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (nswap*random_equal->uniform());
  if ((iwhichglobal >= nswap_before) &&
      (iwhichglobal < nswap_before + nswap_local)) {
    int iwhichlocal = iwhichglobal - nswap_before;
    i = local_swap_atom_list[iwhichlocal];
  }

  return i;
}

/* ----------------------------------------------------------------------
 ---<><><>--- if poly flag is set the iswap list contains atoms from all listed types
------------------------------------------------------------------------- */

int FixAtomSwap::pick_i_swap_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (niswap*random_equal->uniform());
  if ((iwhichglobal >= niswap_before) &&
      (iwhichglobal < niswap_before + niswap_local)) {
    int iwhichlocal = iwhichglobal - niswap_before;
    i = local_swap_iatom_list[iwhichlocal];
  }

  return i;
}

/* --------------------------<><><>----------------------------------
 the while loop forces repeated random selections until a j-atom is found matching separation condition
 - or max number of attempts is reached. 30 attempts are allowed
 - that should be sufficient for thin region with lateral dimension 10, and max sep 3
 - for which the odds of success are 1:10.
----------------------------<><><>-------------------------------------- */

int FixAtomSwap::pick_j_swap_atom()
{
  int j = -1;
  int jwhichglobal;
  int indx = 0;
  //---<><><>--- repeat selection until i,j types are diff, and separation is acceptable
  while (j == -1) {
    indx++;
    jwhichglobal = static_cast<int> (njswap*random_equal->uniform());
    if ((jwhichglobal >= njswap_before) &&
        (jwhichglobal < njswap_before + njswap_local)) {
      int jwhichlocal = jwhichglobal - njswap_before;
      j = local_swap_jatom_list[jwhichlocal];
      jatomtype = atom->type[j];
      sw_sep_val = 0;
      if (jatomtype == iatomtype) {
        j = -1;
      } else if (distrib_flag == 1) {
        jpick = j;
        j = calc_swap_sep();
      }
    }
    if (indx > 30) break;
  }
  
  return j;
}

/* -----------------<><><>-------------------------------------
 in calculating swap separation distance need to account for effects of periodic b/c
 flags for p.b.c. and boxlo[*], boxhi[*] are public variables from domain
 we assume here that the simulation region is an orthogonal box
 (not a problem for type separation)
 >>>>>>there is no protection against fancier shapes <<<<<<
-------------------<><><>------------------------------------ */

int FixAtomSwap::calc_swap_sep()
{
  int j = -1;
  double **x = atom->x;
  double del_x, del_y, del_z;
  double sep2 = -1.0;
  double xmax = (domain->boxhi[0] - domain->boxlo[0])/2.0;
  double ymax = (domain->boxhi[1] - domain->boxlo[1])/2.0;
  double zmax = (domain->boxhi[2] - domain->boxlo[2])/2.0;
  
  if (ipick >= 0 && jpick >= 0) {
    del_x = x_flag*(x[ipick][0] - x[jpick][0]);
    del_y = y_flag*(x[ipick][1] - x[jpick][1]);
    del_z = z_flag*(x[ipick][2] - x[jpick][2]);
    if (domain->xperiodic && (abs(del_x) > xmax) ) del_x = abs(del_x) - xmax;
    if (domain->yperiodic && (abs(del_y) > xmax) ) del_y = abs(del_y) - ymax;
    if (domain->zperiodic && (abs(del_z) > xmax) ) del_z = abs(del_z) - zmax;

    if ((distance_flag == 1) || (typ_flag == 1)) {
      if (typ_flag == 1) {
        sw_sep_val = abs(jatomtype - iatomtype);
      } else if (distance_flag == 1) {
        sw_sep_val = sqrt(del_x*del_x + del_y*del_y + del_z*del_z);
      }
      if ( sw_sep_val <= max_swap_sep ) {
        j  = jpick;
      } else {
        j = -1;
      }
    } else {
      j = jpick;
    }
  }
    /*---<><><>---
     std::string mystring = "in calc_swap_sep with i,j types: " + std::to_string(atom->type[ipick]) + ", " + std::to_string(atom->type[jpick]) + " and separation: " + std::to_string(sqrt(sep2)) + " \n";
    utils::logmesg(lmp,mystring);
     ----<><><>---*/
    return j;
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixAtomSwap::update_semi_grand_atoms_list()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;

  if (atom->nmax > atom_swap_nmax) {
    memory->sfree(local_swap_atom_list);
    atom_swap_nmax = atom->nmax;
    local_swap_atom_list = (int *) memory->smalloc(atom_swap_nmax*sizeof(int),
     "MCSWAP:local_swap_atom_list");
  }

  nswap_local = 0;
  
  if (regionflag) {
    
    for (int i = 0; i < nlocal; i++) {
      if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) == 1) {
        if (atom->mask[i] & groupbit) {
          int itype = atom->type[i];
          int iswaptype;
          for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
            if (itype == type_list[iswaptype]) break;
          if (iswaptype == nswaptypes) continue;
          local_swap_atom_list[nswap_local] = i;
          nswap_local++;
        }
      }
    }
    
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (atom->mask[i] & groupbit) {
        int itype = atom->type[i];
        int iswaptype;
        for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
          if (itype == type_list[iswaptype]) break;
        if (iswaptype == nswaptypes) continue;
        local_swap_atom_list[nswap_local] = i;
        nswap_local++;
      }
    }
  }
  
  MPI_Allreduce(&nswap_local,&nswap,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&nswap_local,&nswap_before,1,MPI_INT,MPI_SUM,world);
  nswap_before -= nswap_local;
}


/* ----------------------------------------------------------------------
 update the list of gas atoms
 ------------------------------------------------------------------------- */

void FixAtomSwap::update_swap_atoms_list()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double **x = atom->x;
  
  if (atom->nmax > atom_swap_nmax) {
    memory->sfree(local_swap_iatom_list);
    memory->sfree(local_swap_jatom_list);
    atom_swap_nmax = atom->nmax;
    local_swap_iatom_list = (int *) memory->smalloc(atom_swap_nmax*sizeof(int),
      "MCSWAP:local_swap_iatom_list");
    local_swap_jatom_list = (int *) memory->smalloc(atom_swap_nmax*sizeof(int),
      "MCSWAP:local_swap_jatom_list");
  }
  
  niswap_local = 0;
  njswap_local = 0;
  /* ---<><><>--- include atoms from all listed types in swap list if poly flag is set */
  if (regionflag) {
    
    for (int i = 0; i < nlocal; i++) {
      if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) == 1) {
        if (atom->mask[i] & groupbit) {
          if (poly_flag == 0) {
            if (type[i] ==  type_list[0]) {
              local_swap_iatom_list[niswap_local] = i;
              niswap_local++;
            } else if (type[i] ==  type_list[1]) {
              local_swap_jatom_list[njswap_local] = i;
              njswap_local++;
            }
          } else {
            int accept = 0;
            for(int lndx = 0; lndx < atom->ntypes; lndx++) {
              if (type[i] == type_list[lndx]) accept = 1;
            }
            if (accept == 1) {
              local_swap_iatom_list[niswap_local] = i;
              niswap_local++;
              local_swap_jatom_list[njswap_local] = i;
              njswap_local++;
            }
          }
        }
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (atom->mask[i] & groupbit) {
        if (poly_flag == 0) {
          if (type[i] ==  type_list[0]) {
            local_swap_iatom_list[niswap_local] = i;
            niswap_local++;
          } else if (type[i] ==  type_list[1]) {
            local_swap_jatom_list[njswap_local] = i;
            njswap_local++;
          }
        }
      } else {
        int accept = 0;
        for(int lndx = 0; lndx < atom->ntypes; lndx++) {
          if (type[i] == type_list[lndx]) accept = 1;
        }
        if (accept == 1) {
          local_swap_iatom_list[njswap_local] = i;
          njswap_local++;
          local_swap_jatom_list[njswap_local] = i;
          niswap_local++;
        }
      }
    }
  }
    
    MPI_Allreduce(&niswap_local,&niswap,1,MPI_INT,MPI_SUM,world);
    MPI_Scan(&niswap_local,&niswap_before,1,MPI_INT,MPI_SUM,world);
    niswap_before -= niswap_local;
    
    MPI_Allreduce(&njswap_local,&njswap,1,MPI_INT,MPI_SUM,world);
    MPI_Scan(&njswap_local,&njswap_before,1,MPI_INT,MPI_SUM,world);
    njswap_before -= njswap_local;
  }

/* ---------------------------------------------------------------------- */

int FixAtomSwap::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;

  if (atom->q_flag) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
      buf[m++] = q[j];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwap::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;
  last = first + n;

  if (atom->q_flag) {
    for (i = first; i < last; i++) {
      type[i] = static_cast<int> (buf[m++]);
      q[i] = buf[m++];
    }
  } else {
    for (i = first; i < last; i++)
      type[i] = static_cast<int> (buf[m++]);
  }
}

/* ----------------------------------------------------------------------
  return acceptance ratio
------------------------------------------------------------------------- */
/* ---<><><>---
 compute_vector returns whatever element is requested
---<><><>--- */

double FixAtomSwap::compute_vector(int n)
{
  if (n == 0) return nswap_attempts;
  if (n == 1) return nswap_successes;
  if ((n >= 2) && (distrib_flag == 1)) {
    /*<<<<< ---<><><>---
     std::string mystring = "returning swap_sep_dist[ "+std::to_string(n-2)+"]\n";
    utils::logmesg(lmp,mystring);
     <<< ---<><><>--- */
    return swap_sep_dist[n-2];
  }
  return 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixAtomSwap::memory_usage()
{
  double bytes = atom_swap_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixAtomSwap::write_restart(FILE *fp)
{
  int n = 0;
  double list[6];
  list[n++] = random_equal->state();
  list[n++] = random_unequal->state();
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = nswap_attempts;
  list[n++] = nswap_successes;
  list[n++] = ubuf(update->ntimestep).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */
  /* ---<><><>---<<<<<<<< how does one include the swp_sep_dist in this restart??*/

void FixAtomSwap::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int> (list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = (bigint) ubuf(list[n++]).i;

  nswap_attempts = static_cast<int>(list[n++]);
  nswap_successes = static_cast<int>(list[n++]);

  bigint ntimestep_restart = (bigint) ubuf(list[n++]).i;
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR,"Must not reset timestep when restarting fix atom/swap");
}
