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

#ifndef _SCALAR_H
#define _SCALAR_H

#include "bluebottle.h"

#define S_MAX_COEFFS 25
#define SNSP 2
#define SP_YS_RE 0
#define SP_YS_IM 1

typedef struct BC_s {
  int sW;
  real sWD;
  real sWN;
  int sE;
  real sED;
  real sEN;
  int sN;
  real sND;
  real sNN;
  int sS;
  real sSD;
  real sSN;
  int sT;
  real sTD;
  real sTN;
  int sB;
  real sBD;
  real sBN;
} BC_s;
/*
 * PURPOSE
 * MEMBERS
 * * tW -- the boundary condition type
 * * tWD -- the DIRICHLET boundary conditon value
 * * tWN -- the NEUMANN boundary condition value
 */

extern BC_s bc_s;
extern BC_s *_bc_s;

/*
 * PURPOSE
 * MEMBERS
 * * s is the current time step scalar value
 * * update is 1 when the particle's temperature change with fluid, is 0 when particle surface temperature is fixed
 * * rs the integrate surface
 * * q is the intergral of hear flux across the particle surface
 * * cp is the particle specific heat
 * * order is the order to keep lamb solution, equals to index n in Ynm
 * * ncoeff is the corresponding m index in Ynm
*/

extern real s_D; // thermal diffusivity, used in diffusivity term, D\nabla_T^2
extern real s_k; //thermal conductivity of fluid, s_k = s_D*\rho_f*c_pp
extern real s_alpha; //coefficient of thermal expansion, used in bousinesq assumption, alpha*gravity*(T-T_ref)
extern int SCALAR; // SCALAR >= 1: calculate the temperature field
extern real lamb_cut_scalar; // lamb cut-off for scalar calculation
extern real s_init; //initial temperature for fluid
extern real s_init_rand;

extern real s_perturbation; // the perturbation solution for T by including the effect of rapidly changing particle surface temperature. See JCP paper.
extern int s_ncoeffs_max;

extern real *s0;
extern real *s;
extern real *s_conv0;
extern real *s_conv;
extern real *s_diff0;
extern real *s_diff;

extern real *_s0;
extern real *_s;
extern real *_s_conv0;
extern real *_s_conv;
extern real *_s_diff0;
extern real *_s_diff;

extern real *_int_Ys_re;
extern real *_int_Ys_im;

/******************************************************************************/
void scalar_init_fields(void);
void scalar_part_init(void);

void cuda_scalar_malloc_host(void);
void cuda_scalar_malloc_dev(void);
void cuda_scalar_push(void);
void cuda_scalar_pull(void);
void cuda_scalar_pull_debug(void);
void cuda_scalar_pull_restart(void);


void cuda_scalar_free(void);


void cuda_scalar_part_BC(real *array);
void cuda_scalar_part_fill(void);
void cuda_scalar_solve(void);
void cuda_scalar_lamb(void);
real cuda_scalar_lamb_err(void);
void cuda_store_s(void);
void cuda_scalar_update_part(void);

void cuda_scalar_BC(real *array);
void cuda_scalar_transfer_parts_i(void);
void cuda_scalar_transfer_parts_j(void);
void cuda_scalar_transfer_parts_k(void);
void mpi_send_s_parts_i(void);
void mpi_send_s_parts_j(void);
void mpi_send_s_parts_k(void);
void cuda_compute_boussinesq(void);
void cuda_scalar_partial_sum_i(void);
void cuda_scalar_partial_sum_j(void);
void cuda_scalar_partial_sum_k(void);
void mpi_send_s_psums_i(void);
void mpi_send_s_psums_j(void);
void mpi_send_s_psums_k(void);
void printMemInfo(void);

#endif
