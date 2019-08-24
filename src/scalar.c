/*******************************************************************************
 ********************************* BLUEBOTTLE **********************************
 *******************************************************************************
 *
 *  Copyright 2015 - 2016 Yayun Wang, The Johns Hopkins University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Please contact the Johns Hopkins University to use Bluebottle for
 *  commercial and/or for-profit applications.
 ******************************************************************************/

#include "scalar.h"

BC_s bc_s;
real s_D;
real s_k;
real s_perturbation;
real s_init;
real s_init_rand;
real s_alpha;
real lamb_cut_scalar;
int SCALAR;
int s_ncoeffs_max;

real *s0;
real *s;
real *s_conv0;
real *s_conv;
real *s_diff0;
real *s_diff;

real *_s0;
real *_s;
real *_s_conv0;
real *_s_conv;
real *_s_diff0;
real *_s_diff;

part_struct_scalar *s_parts;
part_struct_scalar *_s_parts;

part_struct_scalar *_send_s_parts_e;
part_struct_scalar *_send_s_parts_w;
part_struct_scalar *_send_s_parts_n;
part_struct_scalar *_send_s_parts_s;
part_struct_scalar *_send_s_parts_t;
part_struct_scalar *_send_s_parts_b;

part_struct_scalar *_recv_s_parts_e;
part_struct_scalar *_recv_s_parts_w;
part_struct_scalar *_recv_s_parts_n;
part_struct_scalar *_recv_s_parts_s;
part_struct_scalar *_recv_s_parts_t;
part_struct_scalar *_recv_s_parts_b;

MPI_Datatype mpi_s_part_struct;

int *_nn_scalar;
int *_mm_scalar;

void scalar_init_fields(void)
{
  srand(time(0));
  for (int i = 0; i < dom[rank].Gcc.s3b; i++) {
    s[i] = s_init + s_init_rand * (2. * rand() / (real)RAND_MAX - 1.);
    s0[i] = s[i];
    s_conv[i] = 0.0;
    s_conv0[i] = 0.0;
    s_diff[i] = 0.0;
    s_diff0[i] = 0.0;
  }
}

void scalar_part_init(void)
{
  for (int i = 0; i < nparts; i++) {

    s_parts[i].s0 = s_parts[i].s;
    s_parts[i].q = 0.0;
    s_parts[i].iq = 0.0;

    s_parts[i].ncoeff = 0;
    // for each n, -n <= m <= n
    for(int j = 0; j <= s_parts[i].order; j++) {
      s_parts[i].ncoeff += 2*j + 1;
    }
    if(s_parts[i].ncoeff > S_MAX_COEFFS) {
      printf("Maximum order is 4.");
      exit(EXIT_FAILURE);
    }

    for(int j = 0; j < NNODES; j++) {
      s_parts[i].dsdr[j] = 0.0;
    }

    for (int j = 0; j < S_MAX_COEFFS; j++) {
      s_parts[i].anm_re[j] = 0.;
      s_parts[i].anm_im[j] = 0.;
      s_parts[i].anm_re0[j] = 0.;
      s_parts[i].anm_im0[j] = 0.;
      s_parts[i].anm_re00[j] = 0.;
      s_parts[i].anm_im00[j] = 0.;
	}
  }
}

void mpi_send_s_parts_i(void)
{
  // MPI Info hints
  //  - no_locks: window is never locked, e.g. only active sync
  //  - same_disp_unit: all use sizeof(part_struct)
  //  can also use SAME_DISP_UNIT
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_parts{e,w}
  MPI_Win_create(_recv_s_parts_e, nparts_recv[EAST] * sizeof(part_struct_scalar),
    sizeof(part_struct_scalar), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_e);
  MPI_Win_create(_recv_s_parts_w, nparts_recv[WEST] * sizeof(part_struct_scalar),
    sizeof(part_struct_scalar), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_w);

  // Fence and put _send_parts -> _recv_parts
  MPI_Win_fence(0, parts_recv_win_e);
  MPI_Win_fence(0, parts_recv_win_w);

  MPI_Put(_send_s_parts_e, nparts_send[EAST], mpi_s_part_struct, dom[rank].e,
    0, nparts_send[EAST], mpi_s_part_struct, parts_recv_win_w);

  MPI_Put(_send_s_parts_w, nparts_send[WEST], mpi_s_part_struct, dom[rank].w,
    0, nparts_send[WEST], mpi_s_part_struct, parts_recv_win_e);

  MPI_Win_fence(0, parts_recv_win_e);
  MPI_Win_fence(0, parts_recv_win_w);

  // Free
  MPI_Win_free(&parts_recv_win_e);
  MPI_Win_free(&parts_recv_win_w);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_s_parts_j(void)
{
  // MPI Info hints
  //  - no_locks: window is never locked, e.g. only active sync
  //  - same_disp_unit: all use sizeof(part_struct)
  //  can also use SAME_DISP_UNIT
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_parts{n,s}
  MPI_Win_create(_recv_s_parts_n, nparts_recv[NORTH] * sizeof(part_struct_scalar),
    sizeof(part_struct_scalar), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_n);
  MPI_Win_create(_recv_s_parts_s, nparts_recv[SOUTH] * sizeof(part_struct_scalar),
    sizeof(part_struct_scalar), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_s);

  // Fence and put _send_parts -> _recv_parts
  MPI_Win_fence(0, parts_recv_win_n);
  MPI_Win_fence(0, parts_recv_win_s);

  MPI_Put(_send_s_parts_n, nparts_send[NORTH], mpi_s_part_struct, dom[rank].n,
    0, nparts_send[NORTH], mpi_s_part_struct, parts_recv_win_s);
  MPI_Put(_send_s_parts_s, nparts_send[SOUTH], mpi_s_part_struct, dom[rank].s,
    0, nparts_send[SOUTH], mpi_s_part_struct, parts_recv_win_n);

  MPI_Win_fence(0, parts_recv_win_n);
  MPI_Win_fence(0, parts_recv_win_s);

  // Free
  MPI_Win_free(&parts_recv_win_n);
  MPI_Win_free(&parts_recv_win_s);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_s_parts_k(void)
{
  // MPI Info hints
  //  - no_locks: window is never locked, e.g. only active sync
  //  - same_disp_unit: all use sizeof(part_struct)
  //  can also use SAME_DISP_UNIT
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_parts{t,b}
  MPI_Win_create(_recv_s_parts_t, nparts_recv[TOP] * sizeof(part_struct_scalar),
    sizeof(part_struct_scalar), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_t);
  MPI_Win_create(_recv_s_parts_b, nparts_recv[BOTTOM] * sizeof(part_struct_scalar),
    sizeof(part_struct_scalar), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_b);

  // Fence and put _send_parts -> _recv_parts
  MPI_Win_fence(0, parts_recv_win_t);
  MPI_Win_fence(0, parts_recv_win_b);

  MPI_Put(_send_s_parts_t, nparts_send[TOP], mpi_s_part_struct, dom[rank].t,
    0, nparts_send[TOP], mpi_s_part_struct, parts_recv_win_b);
  MPI_Put(_send_s_parts_b, nparts_send[BOTTOM], mpi_s_part_struct, dom[rank].b,
    0, nparts_send[BOTTOM], mpi_s_part_struct, parts_recv_win_t);

  MPI_Win_fence(0, parts_recv_win_t);
  MPI_Win_fence(0, parts_recv_win_b);

  // Free
  MPI_Win_free(&parts_recv_win_t);
  MPI_Win_free(&parts_recv_win_b);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void scalar_part_free(void)
{
  free(s_parts);
}
