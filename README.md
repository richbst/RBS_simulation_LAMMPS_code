The modified files (fix_atom_swap and fix_deposit) used in LAMMPS code 23 June 2022 for simulations when growing size-dispersed films.

They allow deposition or swap of atom types randomly selected from a specified list.
Details of the capabilities and command options are described in the pdf documents in this folder - the changes from original command in red.
It is useful to also modify both the version name and the CMakeLists script to indicate that the compiled version is different than the mainline.
The modifications to the LAMMPS files are indicated with --<><><>--.
