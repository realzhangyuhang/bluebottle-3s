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

#define NNODES 26
/*
 * PURPOSE
 *  Define the number of nodes used for the Lebedev quadrature scheme.
 ******
 */

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

extern part_struct_scalar *parts_s;

extern part_struct_scalar **_parts_s;

extern int coeff_stride_scalar;
/*
 * stores the maximum order for all particles for lamb solution, equal to max index n in Ynm
 */

extern real lamb_cut_scalar; // lamb cut-off for scalar calculation

extern real s_init; //initial temperature for fluid

extern real s_alpha; //coefficient of thermal expansion, used in bousinesq assumption, alpha*gravity*(T-T_ref)

extern real s_D; // thermal diffusivity, used in diffusivity term, D\nabla_T^2

extern real s_k; //thermal conductivity of fluid, s_k = s_D*\rho_f*c_pp

extern real s_perturbation; // the perturbation solution for T by including the effect of rapidly changing particle surface temperature. See JCP paper.

extern int scalar_on; // scalar_on = 1: calculate the temperature field


extern real *s0; //previous step fluid temperature
extern real *s; // current step fluid temperature
extern real *conv0_s;
extern real *conv_s;
extern real *diff0_s;
extern real *diff_s;

extern real **_s0;
extern real **_s;
extern real **_conv0_s;
extern real **_conv_s;
extern real **_diff0_s;
extern real **_diff_s;

extern real *anm_re; // lamb coefficients
extern real *anm_im;
extern real *anm_re0;
extern real *anm_im0;
extern real *anm_re00;
extern real *anm_im00;
extern real *anm_re_perturb;
extern real *anm_im_perturb;



extern real **_anm_re;
extern real **_anm_im;
extern real **_anm_re0;
extern real **_anm_im0;
extern real **_anm_re00;
extern real **_anm_im00;
extern real **_anm_re_perturb;
extern real **_anm_im_perturb;


extern int *_nn_scalar;
extern int *_mm_scalar;

void scalar_read_input(void);
/*
 * FUNCTION
 * read the scalar.config file
 ****
 */

void show_scalar_config(void);
/*
 * FUNCTION
 * show the scalar.config file
 ****
 */

void scalar_init(void);
/*
 * FUNCTION
 * allocate and init the variable
 ****
 */

void scalar_clean(void);
/*
 * FUNCTION
 * clean scalar field on host
 ****
 */

void scalar_out_restart(void);
/*
 * FUNCTION
 * write restart file for scalar field
 ***
 */

void scalar_in_restart(void);
/*
 * FUNCTION
 * read in restart file for scalar field
 ***
 */

void parts_read_input_scalar(void);
/*
 * function
 * read in particle initial scalar value, k, intergrate surface and lamb solution order
 */


void parts_read_input_scalar_restart(void);
/* 
 * FUNCTION
 * read in part_scalar.config for scalar field after restart
 *
 */

void parts_scalar_show_config(void);
/*
 * FUNCTION
 * show particle scalar information
 *
 */


void parts_init_scalar(void);
/*
 * function
 * initialize the particle scalar field value
 */

void parts_scalar_clean(void);
/*
 * FUNCTION
 * clean variables
 */

/*************************FUNCTION IN CUDA_SCALAR.CU********************/

void cuda_part_scalar_malloc(void);
/*
 * Function
 * if scalar_on == 1, allocate and init variables
 */

void cuda_part_scalar_push(void);

void cuda_part_scalar_pull(void);

void cuda_part_scalar_free(void);


void cuda_scalar_malloc(void);

void cuda_scalar_push(void);

void cuda_scalar_pull(void);

void cuda_scalar_free(void);


void cuda_scalar_BC_s0(void);
/*
 * function
 * applying the boundary condition for scalar field before calulating
 */

void cuda_scalar_BC_s(void);
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


void cuda_compute_boussinesq(void);

void cuda_part_heat_flux(void);

void cuda_store_coeffs_scalar(void);

void cuda_update_part_scalar(void);
#endif
