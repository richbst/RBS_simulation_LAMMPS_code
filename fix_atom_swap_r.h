/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
 
 -------------<><><>---------------
 added private variables:
   int poly_flag to allow random choice among many atom types
   int distance_flag to set max on pair swap distance separation
   int typ_flag to set max on pair swap type separation
   int distrib_flag to return type/distance separations for swaps
   int ipick, jpick to identify swap choice
   double max_swap_sep to specify maximum swap pair separation
   vector swap_sep_dist for distribution of swap pair separations
   int max_sep, sep_no for range and bin no of the distribution output
   int iatomtype, jatomtype to record selected atom types
 added public subroutine:
   int neighbor_sep() to force jpick to region near ipick by repeated random selections
 Rich Stephens - November 2021
 -------------<><><>---------------
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(atom/swap,FixAtomSwap);
// clang-format on
#else

#ifndef LMP_FIX_MCSWAP_H
#define LMP_FIX_MCSWAP_H

#include "fix.h"

namespace LAMMPS_NS {

class FixAtomSwap : public Fix {
 public:
  FixAtomSwap(class LAMMPS *, int, char **);
  ~FixAtomSwap();
  int setmask();
  void init();
  void pre_exchange();
  int attempt_semi_grand();
  int attempt_swap();
  double energy_full();
  int pick_semi_grand_atom();
  int pick_i_swap_atom();
  int pick_j_swap_atom();
  int calc_swap_sep();          // ---<><><>-- forces jpick to be near ipick
  void update_semi_grand_atoms_list();
  void update_swap_atoms_list();
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  double compute_vector(int);
  double memory_usage();
  void write_restart(FILE *);
  void restart(char *);

 private:
  int nevery, seed;
  int conserve_ke_flag;        // yes = conserve ke, no = do not conserve ke
  int semi_grand_flag;         // yes = semi-grand canonical, no = constant composition
  int distance_flag;           // ---<><><>--- 1 = swap restricted to nearby atoms, 0 = not
  int x_flag, y_flag, z_flag;  // ---<><><>--- 1 if distance is restricted in that dimension
  int typ_flag;                // ---<><><>---1 = swap restricted to similar types, 0 = not
  int distrib_flag;            // ---<><><>--- return separations for swaps
  int poly_flag;               // ---<><><>--- allow random choice among many atom types
  int ncycles;
  int niswap, njswap;                  // # of i,j swap atoms on all procs
  int niswap_local, njswap_local;      // # of i, j swap atoms on this proc
  int niswap_before, njswap_before;    // # of i, j swap atoms on procs < this proc
  int nswap;                           // # of swap atoms on all procs
  int nswap_local;                     // # of swap atoms on this proc
  int nswap_before;                    // # of swap atoms on procs < this proc
  int regionflag;                      // 0 = anywhere in box, 1 = specific region
  int iregion;                         // swap region
  char *idregion;                      // swap region id

  int nswaptypes, nmutypes;
  int *type_list;
  double *mu;
  double max_swap_sep;        // ---<><><>--- allowed swap pair separation. Must be >1
  double sw_sep_val;          // ---<><><>--- recorded swap pair separation.
                             // ---<><><>--- = 0 if no pair found, (-1)* if swap unsuccessful
  int sep_no;                // ---<><><>--- number of values in distribution
  int max_sep;               // ---<><><>--- max value for recorded distribution
  int ipick, jpick;           // ---<><><>--- i = the 1st atom choice, j = the 2nd
  int iatomtype, jatomtype;   // ----<><><>--- the types for the i, j choices

  double nswap_attempts;
  double nswap_successes;

  bool unequal_cutoffs;

  int atom_swap_nmax;
  double beta;
  double *qtype;
  double energy_stored;
  double **sqrt_mass_ratio;
  int *local_swap_iatom_list;
  int *local_swap_jatom_list;
  int *local_swap_atom_list;
  //---<><><>--- distribution of swap pair separations - neg if unsuccessful. len()=2*sep_no+1
  int *swap_sep_dist;

  class RanPark *random_equal;
  class RanPark *random_unequal;

  class Compute *c_pe;

  void options(int, char **);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix atom/swap does not exist

Self-explanatory.

E: Must specify at least 2 types in fix atom/swap command

Self-explanatory.

E: Need nswaptypes mu values in fix atom/swap command

Self-explanatory.

E: Only 2 types allowed when not using semi-grand in fix atom/swap command

Self-explanatory.

E: Mu not allowed when not using semi-grand in fix atom/swap command

Self-explanatory.

E: Invalid atom type in fix atom/swap command

The atom type specified in the atom/swap command does not exist.

E: All atoms of a swapped type must have the same charge.

Self-explanatory.

E: At least one atom of each swapped type must be present to define charges.

Self-explanatory.

E: All atoms of a swapped type must have same charge.

Self-explanatory.

E: Cannot do atom/swap on atoms in atom_modify first group

This is a restriction due to the way atoms are organized in a list to
enable the atom_modify first command.
 
 ---<><><>---added along with the swap sep restriction and distribution reporting
 
 E: Swap pair separation limit must be >= 1.0
 
 Self-explanatory.
 
 E: More than 2 types required when using poly in fix atom/swap command
 
 Self-explanatory.

*/
