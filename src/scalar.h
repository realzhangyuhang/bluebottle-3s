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

typedef struct part_struct_scalar {
  real s0;
  real s;
  int update;
  real rs;
  real q;
  real iq;
  real cp;
  int order;
  int ncoeff;
  real dsdr[NNODES];
  real anm_re[S_MAX_COEFFS];
  real anm_im[S_MAX_COEFFS];
  real anm_re0[S_MAX_COEFFS];
  real anm_im0[S_MAX_COEFFS];
  real anm_re00[S_MAX_COEFFS];
  real anm_im00[S_MAX_COEFFS];
} part_struct_scalar;
/*
 * PURPOSE
 * MEMBERS
 * * s0 is the previous time step scalar value
 * * s is the current time step scalar value
 * * update is 1 when the particle's temperature change with fluid, is 0 when particle surface temperature is fixed
 * * rs the integrate surface
 * * q is the intergral of hear flux across the particle surface
 * * iq is the lubircation correction for heat flux across the particle surface
 * * k is the termal conductivity for each particle
 * * cp is the particle specific heat
 * * order is the order to keep lamb solution, equals to index n in Ynm
 * * ncoeff is the corresponding m index in Ynm
 * * dsdr is the scalar gradient at particle surface for Lebsque nodes
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

extern part_struct_scalar *s_parts;
extern part_struct_scalar *_s_parts;

extern part_struct_scalar *_send_s_parts_e;
extern part_struct_scalar *_send_s_parts_w;
extern part_struct_scalar *_send_s_parts_n;
extern part_struct_scalar *_send_s_parts_s;
extern part_struct_scalar *_send_s_parts_t;
extern part_struct_scalar *_send_s_parts_b;

extern part_struct_scalar *_recv_s_parts_e;
extern part_struct_scalar *_recv_s_parts_w;
extern part_struct_scalar *_recv_s_parts_n;
extern part_struct_scalar *_recv_s_parts_s;
extern part_struct_scalar *_recv_s_parts_t;
extern part_struct_scalar *_recv_s_parts_b;

extern MPI_Datatype mpi_s_part_struct;

extern int *_nn_scalar;
extern int *_mm_scalar;

/*
void scalar_clean(void);
void scalar_out_restart(void);
void scalar_in_restart(void);
void parts_read_input_scalar_restart(void);
void parts_scalar_clean(void);
*/

/******************************************************************************/
void scalar_init_fields(void);
void scalar_part_init(void);
void scalar_part_free(void);
void cuda_scalar_malloc_host(void);
void cuda_scalar_malloc_dev(void);
void cuda_scalar_push(void);
void cuda_scalar_pull(void);
void cuda_scalar_pull_debug(void);
void cuda_scalar_pull_restart(void);
void cuda_scalar_part_malloc_dev(void);
void cuda_scalar_part_push(void);
void cuda_scalar_free(void);
void cuda_scalar_part_free(void);
void cuda_part_pull_with_scalar(void);

void cuda_scalar_BC(void);
void cuda_scalar_transfer_parts_i(void);
void cuda_scalar_transfer_parts_j(void);
void cuda_scalar_transfer_parts_k(void);
void mpi_send_s_parts_i(void);
void mpi_send_s_parts_j(void);
void mpi_send_s_parts_k(void);
void cuda_compute_boussinesq(void);

/*************************FUNCTION IN CUDA_SCALAR.CU********************/


void cuda_scalar_BC_s0(void);
/*
 * function
 * applying the boundary condition for scalar field before calulating
 */

void cuda_solve_scalar_explicit(void);

void cuda_update_scalar(void);

void cuda_quad_check_nodes_scalar(int dev,real *node_t, real *node_p, int nnodes);
/*
 * Function
 * check if the intergrat nodes inter-section with the wall
 */


void cuda_quad_interp_scalar(int dev, real *node_t, real *node_p, int nnodes, real *ss);
/*
 * Function
 * interpolate scalar field value into lebsque nodes
 * node_t is the theta for lebsque nodes
 * node_p is the phi for lebsque nodes
 ** nnodes is the number of lebsque nodes
 */

void cuda_scalar_lamb(void);
/*
 * FUNCTION
 * interpolate the outer field to Lebsque nodes and use the value to calculate the coefficents
 */

real cuda_scalar_lamb_err(void);
/*
 * FUNCTION
 * find the residue for lamb coefficents
 */

void cuda_part_BC_scalar_s0(void);
/*
 * FUNCTION
 * Apply the Dirichlet boundary condition to those nodes for scalar field
 */

void cuda_part_BC_scalar_fill(void);

void cuda_show_variable(void);

void cuda_part_heat_flux(void);

void cuda_store_coeffs_scalar(void);

void cuda_update_part_scalar(void);
#endif
