#ifndef _CUDA_SCALAR_H
#define _CUDA_SCALAR_H

extern "C"
{
#include "bluebottle.h"
#include "scalar.h"
}

#include "cuda_particle.h"

__global__ void pack_s_parts_e(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);

__global__ void pack_s_parts_w(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);

__global__ void pack_s_parts_n(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);

__global__ void pack_s_parts_s(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);

__global__ void pack_s_parts_t(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);

__global__ void pack_s_parts_b(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);

__global__ void copy_central_bin_s_parts_i(part_struct_scalar *tmp_parts,
  part_struct_scalar *parts, int *bin_start, int *bin_count, int *part_ind, int *offset);

__global__ void copy_central_bin_s_parts_j(part_struct_scalar *tmp_parts,
  part_struct_scalar *parts, int *bin_start, int *bin_count, int *part_ind, int *offset);

__global__ void copy_central_bin_s_parts_k(part_struct_scalar *tmp_parts,
  part_struct_scalar *parts, int *bin_start, int *bin_count, int *part_ind, int *offset);

__global__ void copy_ghost_bin_s_parts(part_struct_scalar *tmp_parts,
  part_struct_scalar *recv_parts, int nparts_recv, int offset, int plane,
  dom_struct *DOM);

__global__ void BC_s_W_D(real *s, real bc_s);

__global__ void BC_s_W_N(real *s, real bc_s);

__global__ void BC_s_E_D(real *s, real bc_s);

__global__ void BC_s_E_N(real *s, real bc_s);

__global__ void BC_s_N_D(real *s, real bc_s);

__global__ void BC_s_N_N(real *s, real bc_s);

__global__ void BC_s_S_D(real *s, real bc_s);

__global__ void BC_s_S_N(real *s, real bc_s);

__global__ void BC_s_B_D(real *s, real bc_s);

__global__ void BC_s_B_N(real *s, real bc_s);

__global__ void BC_s_T_D(real *s, real bc_s);

__global__ void BC_s_T_N(real *s, real bc_s);

__global__ void forcing_boussinesq_x(real alpha, real gx, real s_init, real *s, real *fx);

__global__ void forcing_boussinesq_y(real alpha, real gy, real s_init, real *s, real *fy);

__global__ void forcing_boussinesq_z(real alpha, real gz, real s_init, real *s, real *fz);

__global__ void copy_subdom_parts_with_scalar(part_struct *tmp_parts, part_struct *parts,
  part_struct_scalar *tmp_s_parts, part_struct_scalar *s_parts,
  int *bin_start, int *bin_count, int *part_ind, int *bin_offset);

#endif
