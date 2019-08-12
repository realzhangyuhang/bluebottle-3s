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
real s_alpha;
real lamb_cut_scalar;
int scalar_on;
int coeff_stride_scalar;

part_struct_scalar *parts_s;
real *s0;
real *s;
real *conv0_s;
real *conv_s;
real *diff0_s;
real *diff_s;

real *anm_re;
real *anm_im;
real *anm_re0;
real *anm_im0;
real *anm_re00;
real *anm_im00;

real *anm_re_perturb;
real *anm_im_perturb;

part_struct_scalar **_parts_s;
real **_s0;
real **_s;
real **_conv0_s;
real **_conv_s;
real **_diff0_s;
real **_diff_s;

real **_anm_re;
real **_anm_im;
real **_anm_re0;
real **_anm_im0;
real **_anm_re00;
real **_anm_im00;

real **_anm_re_perturb;
real **_anm_im_perturb;

int *_nn_scalar;
int *_mm_scalar;

void scalar_read_input(void)
{

  int fret = 0;
  fret = fret; // prevent compiler warning

  cpumem = 0;
  gpumem = 0;

  // open configuration file for reading
  char fname[FILE_NAME_SIZE] = "";
  sprintf(fname, "%s/input/scalar.config", ROOT_DIR);
  FILE *infile = fopen(fname, "r");
  if(infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    fprintf(stderr, "No scalar field will be calculated.\n");
    scalar_on = 0;
  }
  else {
    scalar_on = 1;
    char buf[CHAR_BUF_SIZE] = "";  // character read buffer
    fret = fscanf(infile, "scalar_on %d\n", &scalar_on);
    if (scalar_on == 1) {    
      // read domain
#ifdef DOUBLE
      fret = fscanf(infile, "diffusivity %lf\n", &s_D);
      fret = fscanf(infile, "conductivity %lf\n", &s_k);
      fret = fscanf(infile, "perturbation solution %lf\n", &s_perturbation);
      fret = fscanf(infile, "lamb_cut %lf\n", &lamb_cut_scalar);
      fret = fscanf(infile, "initial_scalar %lf\n", &s_init);
      fret = fscanf(infile, "alpha %lf\n", &s_alpha);
#else
      fret = fscanf(infile, "diffusivity %f\n", &s_D);
      fret = fscanf(infile, "conductivity %f\n", &s_k);
      fret = fscanf(infile, "perturbation solution %f\n", &s_perturbation);
      fret = fscanf(infile, "lamb_cut %f\n", &lamb_cut_scalar);
      fret = fscanf(infile, "initial_scalar %f\n", &s_init);
      fret = fscanf(infile, "alpha %f\n", &s_alpha);
#endif
      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "BOUNDARY CONDITIONS\n");
      fret = fscanf(infile, "bc_s.sW %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sW = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sW = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sWD);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sW = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sWN);
      } else {
        fprintf(stderr, "flow.config read error in W boundary condition.\n");
        exit(EXIT_FAILURE);
      }

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sE %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sE = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sE = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sED);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sE = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sEN); 
      } else {
        fprintf(stderr, "flow.config read error in E boundary condition.\n");
        exit(EXIT_FAILURE);
      }      

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sN %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sN = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sN = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sND);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sN = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sNN);
      } else {
        fprintf(stderr, "flow.config read error in N boundary condition.\n");
        exit(EXIT_FAILURE);
      }

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sS %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sS = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sS = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sSD);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sS = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sSN);
      } else {
        fprintf(stderr, "flow.config read error in S boundary condition.\n");
        exit(EXIT_FAILURE);
      }             

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sB %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sB = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sB = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sBD);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sB = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sBN);
      } else {
        fprintf(stderr, "flow.config read error in B boundary condition.\n");
        exit(EXIT_FAILURE);
      }      

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sT %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sT = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sT = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sTD);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sT = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sTN);
      } else {
        fprintf(stderr, "flow.config read error in T boundary condition.\n");
        exit(EXIT_FAILURE);
      }

    }
  }
}

void show_scalar_config(void)
{
  if(scalar_on == 1) {
    printf("Show scalar.config...\n");
    printf("scalar_on is %d\n", scalar_on);
    printf("diffusivity is %f\n", s_D);
    printf("conductivity is %f\n", s_k);
    printf("perturbation_solution is %f\n", s_perturbation);
    printf("lamb_cut is %f\n", lamb_cut_scalar);
    printf("initial_scalar is %f\n", s_init);
    printf("alpha is %f\n", s_alpha);
    printf("Boundary condition is:\n");
    printf("  On W ");
    if(bc_s.sW == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sWD);
    else if(bc_s.sW == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sWN);
    else if(bc_s.sW == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
    else printf(" bc_s.sW is wrong with value %d\n", bc_s.sW);
  
    printf("  On E ");
    if(bc_s.sE == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sED);
    else if(bc_s.sE == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sEN);
    else if(bc_s.sE == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
    else printf(" bc_s.sE is wrong with value %d\n", bc_s.sE);

    printf("  On N ");
    if(bc_s.sN == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sND);
    else if(bc_s.sN == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sNN);
    else if(bc_s.sN == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
    else printf(" bc_s.sN is wrong with value %d\n", bc_s.sN);

    printf("  On S ");
    if(bc_s.sS == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sSD);
    else if(bc_s.sS == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sSN);
    else if(bc_s.sS == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
    else printf(" bc_s.sS is wrong with value %d\n", bc_s.sS);

    printf("  On B ");
    if(bc_s.sB == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sBD);
    else if(bc_s.sB == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sBN);
    else if(bc_s.sB == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
    else printf(" bc_s.sB is wrong with value %d\n", bc_s.sB);

    printf("  On T ");
    if(bc_s.sT == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sTD);
    else if(bc_s.sT == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sTN);
    else if(bc_s.sT == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
    else printf(" bc_s.sT is wrong with value %d\n", bc_s.sT); 
  }
}

void parts_scalar_show_config(void)
{
  if(scalar_on == 1) {
    printf("Show part_scalar.config...\n");
    for(int i = 0; i < nparts; i++) {
      printf("  Particle %d:\n", i);
      printf("    s = %f\n", parts_s[i].s0);
      printf("    update = %d\n", parts_s[i].update);
      printf("    cp = %f\n", parts_s[i].cp);
      printf("    rs = %f\n", parts_s[i].rs);
      printf("    order = %d\n", parts_s[i].order);
    }
  }
}

void scalar_clean(void)
{
  if(scalar_on == 1) {
    free(s0);
    free(s);
    free(conv0_s);
    free(conv_s);
    free(diff0_s);
    free(diff_s);
  }
}

void scalar_out_restart(void)
{
  if(scalar_on == 1) {
    // create the file
    char path[FILE_NAME_SIZE] = "";
    sprintf(path, "%s/input/restart_scalar.config", ROOT_DIR);
    FILE *rest = fopen(path, "w");
    if(rest == NULL) {
      fprintf(stderr, "Could not open file restart.input.\n");
      exit(EXIT_FAILURE);
    }

    // flow field variable 
    fwrite(s, sizeof(real), DOM.Gcc.s3b, rest);     
    fwrite(s0, sizeof(real), DOM.Gcc.s3b, rest);
    fwrite(conv_s, sizeof(real), DOM.Gcc.s3b, rest);
    fwrite(conv0_s, sizeof(real), DOM.Gcc.s3b, rest);
    fwrite(diff_s, sizeof(real), DOM.Gcc.s3b, rest);
    fwrite(diff0_s, sizeof(real), DOM.Gcc.s3b, rest);

    // particle related variable
    fwrite(parts_s, sizeof(part_struct_scalar), nparts, rest);
    fwrite(&coeff_stride_scalar, sizeof(int), 1, rest);

    fwrite(anm_re, sizeof(real), nparts*coeff_stride_scalar, rest);
    fwrite(anm_im, sizeof(real), nparts*coeff_stride_scalar, rest);  
    fwrite(anm_re0, sizeof(real), nparts*coeff_stride_scalar, rest);
    fwrite(anm_im0, sizeof(real), nparts*coeff_stride_scalar, rest);
    fwrite(anm_re00, sizeof(real), nparts*coeff_stride_scalar, rest);
    fwrite(anm_im00, sizeof(real), nparts*coeff_stride_scalar, rest);
    // close the file
    fclose(rest);
  }
}

void scalar_in_restart(void)
{
  if(scalar_on == 1) {
    int fret = 0;
    fret = fret; // prevent compiler warning
    // open configuration file for reading
    char fname[FILE_NAME_SIZE] = "";
    sprintf(fname, "%s/input/restart_scalar.config", ROOT_DIR);
    FILE *infile = fopen(fname, "r");
    if(infile == NULL) {
      fprintf(stderr, "Could not open file %s\n", fname);
      exit(EXIT_FAILURE);
    }  

    // flow field variable  
    fret = fread(s, sizeof(real), DOM.Gcc.s3b, infile);
    fret = fread(s0, sizeof(real), DOM.Gcc.s3b, infile);
    fret = fread(conv_s, sizeof(real), DOM.Gcc.s3b, infile);
    fret = fread(conv0_s, sizeof(real), DOM.Gcc.s3b, infile);
    fret = fread(diff_s, sizeof(real), DOM.Gcc.s3b, infile);
    fret = fread(diff0_s, sizeof(real), DOM.Gcc.s3b, infile);

    // particle related variable
    fret = fread(parts_s, sizeof(part_struct_scalar), nparts, infile);
    fret = fread(&coeff_stride_scalar, sizeof(int), 1, infile);

    fret = fread(anm_re, sizeof(real), nparts*coeff_stride_scalar, infile);
    fret = fread(anm_im, sizeof(real), nparts*coeff_stride_scalar, infile);
    fret = fread(anm_re0, sizeof(real), nparts*coeff_stride_scalar, infile);
    fret = fread(anm_im0, sizeof(real), nparts*coeff_stride_scalar, infile);
    fret = fread(anm_re00, sizeof(real), nparts*coeff_stride_scalar, infile);
    fret = fread(anm_im00, sizeof(real), nparts*coeff_stride_scalar, infile);
 
    // close file
    fclose(infile);
  }
}

void parts_read_input_scalar(void)
{
  if(scalar_on == 1) {
    int i;
    int fret = 0;
    fret = fret; // prevent compiler warning

    // open configuration file for reading
    char fname[FILE_NAME_SIZE] = "";
    sprintf(fname, "%s/input/part_scalar.config", ROOT_DIR);
    FILE *infile = fopen(fname, "r");
    if(infile == NULL) {
      printf("no part_scalar.config for scalar field\n");
    }  
   
    // read particle list
    parts_s = (part_struct_scalar*) malloc(nparts * sizeof(part_struct_scalar));
    cpumem += nparts * sizeof(part_struct_scalar);

    // read nparts particles
    for(i = 0; i < nparts; i++) {
#ifdef DOUBLE
      fret = fscanf(infile, "s %lf\n", &parts_s[i].s0);
      fret = fscanf(infile, "update %d\n", &parts_s[i].update);
      fret = fscanf(infile, "cp %lf\n", &parts_s[i].cp);
      fret = fscanf(infile, "rs %lf\n", &parts_s[i].rs);
#else
      fret = fscanf(infile, "s %f\n", &parts_s[i].s0);
      fret = fscanf(infile, "update %d\n", &parts_s[i].update);
      fret = fscanf(infile, "cp %f\n", &parts_s[i].cp);
      fret = fscanf(infile, "rs %f\n", &parts_s[i].rs);
#endif
      fret = fscanf(infile, "order %d\n", &parts_s[i].order);
      fret = fscanf(infile, "\n");
    }
    fclose(infile);
  }
} 

void parts_read_input_scalar_restart(void)
{
  if(scalar_on == 1) {
    int i;
    int fret = 0;
    fret = fret; // prevent compiler warning
    
    real tmp1 = 0.0; // temporal value
    int tmp2 = 0; // temporal value
    int update = 0; // if anything is changed update = 1 

    // open configuration file for reading
    char fname[FILE_NAME_SIZE] = "";
    sprintf(fname, "%s/input/part_scalar.config", ROOT_DIR);
    FILE *infile = fopen(fname, "r");
    if(infile == NULL) {
      printf("no part_scalar.config for scalar field\n");
    }

    printf("\n   Reading part_scalar.config for any updated input ...\n");
    // read nparts particles
    for(i = 0; i < nparts; i++) {
#ifdef DOUBLE
      fret = fscanf(infile, "s %lf\n", &tmp1);
      fret = fscanf(infile, "update %d\n", &tmp2);
      if(tmp2 != parts_s[i].update) {
        printf("    particle[%d].update has been updated!\n", i);
        parts_s[i].update = tmp2;
        update = 1;
      }
      fret = fscanf(infile, "cp %lf\n", &tmp1);
      if(tmp1 != parts_s[i].cp){
        printf("    particle[%d].cp  has been updated!\n", i);
        parts_s[i].cp = tmp1;
        update = 1; 
      }
      fret = fscanf(infile, "rs %lf\n", &tmp1);
      if(tmp1 != parts_s[i].rs){
        printf("    particle[%d].rs  has been updated!\n", i);
        parts_s[i].cp = tmp1;
        update = 1;
      }
#else
      fret = fscanf(infile, "s %f\n", &tmp1);
      fret = fscanf(infile, "update %d\n", &parts_s[i].update);
      if(tmp2 != parts_s[i].update) {
        printf("    particle[%d].update has been updated!\n", i);
        parts_s[i].update = tmp2;
        update = 1;
      }
      fret = fscanf(infile, "cp %f\n", &parts_s[i].cp);
      if(tmp1 != parts_s[i].cp){
        printf("    particle[%d].cp  has been updated!\n", i);
        parts_s[i].cp = tmp1;
        update = 1;
      }
      fret = fscanf(infile, "rs %f\n", &parts_s[i].rs);
      if(tmp1 != parts_s[i].rs){
        printf("    particle[%d].rs  has been updated!\n", i);
        parts_s[i].cp = tmp1;
        update = 1;
      }
#endif
      fret = fscanf(infile, "order %d\n", &tmp2);
      if(tmp2 != parts_s[i].order) {
        printf("    particle[%d].order cann't be changed!\n", i);
      }
      fret = fscanf(infile, "\n");
    }
    fclose(infile);

    if(update == 1)parts_scalar_show_config();
  }
}



void parts_init_scalar(void)
{
  coeff_stride_scalar = 0;
  if(scalar_on == 1){
    for(int i = 0; i < nparts; i++) {
      parts_s[i].s = parts_s[i].s0;
      parts_s[i].q = 0.0;
      parts_s[i].ncoeff = 0;
      // for each n, -n<=m<=n
      for(int j = 0; j <= parts_s[i].order; j++) {
        parts_s[i].ncoeff += 2*j + 1;
      }
      if(parts_s[i].ncoeff > coeff_stride_scalar) {
        coeff_stride_scalar = parts_s[i].ncoeff;
      }
      for(int j = 0; j < NNODES; j++) {
        parts_s[i].dsdr[j] = 0.0;
      }
    }
    // allocate lamb's coefficients on host
    anm_re = (real*) malloc(coeff_stride_scalar * nparts * sizeof(real));
    cpumem += coeff_stride_scalar * nparts * sizeof(real);
    anm_im = (real*) malloc(coeff_stride_scalar * nparts * sizeof(real));
    cpumem += coeff_stride_scalar * nparts * sizeof(real);
    anm_re0 = (real*) malloc(coeff_stride_scalar * nparts * sizeof(real));
    cpumem += coeff_stride_scalar * nparts * sizeof(real);
    anm_im0 = (real*) malloc(coeff_stride_scalar * nparts * sizeof(real));
    cpumem += coeff_stride_scalar * nparts * sizeof(real);
    anm_re00 = (real*) malloc(coeff_stride_scalar * nparts * sizeof(real));
    cpumem += coeff_stride_scalar * nparts * sizeof(real);
    anm_im00 = (real*) malloc(coeff_stride_scalar * nparts * sizeof(real));
    cpumem += coeff_stride_scalar * nparts * sizeof(real);
 
    anm_re_perturb = (real*) malloc(coeff_stride_scalar * nparts * sizeof(real));
    cpumem += coeff_stride_scalar * nparts * sizeof(real);
    anm_im_perturb = (real*) malloc(coeff_stride_scalar * nparts * sizeof(real));
    cpumem += coeff_stride_scalar * nparts * sizeof(real);
 
    // initialize lamb's coefficents
    for(int i = 0; i < coeff_stride_scalar * nparts; i++) {
      anm_re[i] = 0.0;
      anm_im[i] = 0.0;
      anm_re0[i] = 0.0;
      anm_im0[i] = 0.0;
      anm_re00[i] = 0.0;
      anm_im00[i] = 0.0;
      
      anm_re_perturb[i] = 0.0;
      anm_im_perturb[i] = 0.0;
    }

/*
    // allocate the lebsque coefficents table and lebsque nodes infor
    nn_scalar = (real*) malloc(25 * sizeof(int));
    cpumem += 25 * sizeof(int);
    mm_scalar = (real*) malloc(25 * sizeof(int));
    cpumem += 25 * sizeof(int);

    // initialize coefficients table
    int nn_scalar[25] = {0,
                  1, 1, 1,
                  2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4, 4, 4, 4, 4};
    int mm_scalar[25] = {0,
                  -1, 0, 1,
                  -2, -1, 0, 1, 2,
                  -3, -2, -1, 0, 1, 2, 3,
                  -4, -3, -2, -1, 0, 1, 2, 3, 4};

    nn_scalar = nn_scalar;
    mm_scalar = mm_scalar;
*/
  }
}

void parts_scalar_clean(void)
{
  if(scalar_on == 1) {
    free(parts_s);
    free(anm_re);
    free(anm_im);
    free(anm_re0);
    free(anm_im0);
    free(anm_re00);
    free(anm_im00);

    free(anm_re_perturb);
    free(anm_im_perturb);
  }
}


