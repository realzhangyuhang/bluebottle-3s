#include "cuda_physalis.h"
#include "cuda_scalar.h"

__global__ void pack_s_parts_e(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    cbin = GFX_LOC(_bins.Gcc._ie, tj, tk, s1b, s2b);
    c2b = tj + tk * _bins.Gcc.jnb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_s_parts_w(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    cbin = GFX_LOC(_bins.Gcc._is, tj, tk, s1b, s2b);
    c2b = tj + tk * _bins.Gcc.jnb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_s_parts_n(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    cbin = GFY_LOC(ti, _bins.Gcc._je, tk, s1b, s2b);
    c2b = tk + ti * _bins.Gcc.knb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_s_parts_s(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    cbin = GFY_LOC(ti, _bins.Gcc._js, tk, s1b, s2b);
    c2b = tk + ti * _bins.Gcc.knb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_s_parts_t(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    cbin = GFZ_LOC(ti, tj, _bins.Gcc._ke, s1b, s2b);
    c2b = ti + tj * _bins.Gcc.inb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_s_parts_b(part_struct_scalar *send_parts, part_struct_scalar *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    cbin = GFZ_LOC(ti, tj, _bins.Gcc._ks, s1b, s2b);
    c2b = ti + tj * _bins.Gcc.inb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void copy_central_bin_s_parts_i(part_struct_scalar *tmp_parts,
  part_struct_scalar *parts, int *bin_start, int *bin_count, int *part_ind,
  int *offset)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;             // bin index
  int pp;               // particle index
  int dest;             // destination in tmp_parts

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    // Loop over i-planes
    for (int i = _bins.Gcc._is; i <= _bins.Gcc._ie; i++) {
      cbin = GFX_LOC(i, tj, tk, s1b, s2b);


      for (int n = 0; n < bin_count[cbin]; n++) {
        pp = part_ind[bin_start[cbin] + n];
        dest = offset[cbin] + n;

        tmp_parts[dest] = parts[pp];
      }
    }
  }
}

__global__ void copy_central_bin_s_parts_j(part_struct_scalar *tmp_parts,
  part_struct_scalar *parts, int *bin_start, int *bin_count, int *part_ind,
  int *offset)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;             // bin index
  int pp;               // particle index
  int dest;             // destination in tmp_parts

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    // Loop over j-planes
    for (int j = _bins.Gcc._js; j <= _bins.Gcc._je; j++) {
      cbin = GFY_LOC(ti, j, tk, s1b, s2b);

      for (int n = 0; n < bin_count[cbin]; n++) {
        pp = part_ind[bin_start[cbin] + n];
        dest = offset[cbin] + n;

        tmp_parts[dest] = parts[pp];
      }
    }
  }
}

__global__ void copy_central_bin_s_parts_k(part_struct_scalar *tmp_parts,
  part_struct_scalar *parts, int *bin_start, int *bin_count, int *part_ind,
  int *offset)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;             // bin index
  int pp;               // particle index
  int dest;             // destination in tmp_parts

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    // Loop over j-planes
    for (int k = _bins.Gcc._ks; k <= _bins.Gcc._ke; k++) {
      cbin = GFZ_LOC(ti, tj, k, s1b, s2b);

      for (int n = 0; n < bin_count[cbin]; n++) {
        pp = part_ind[bin_start[cbin] + n];
        dest = offset[cbin] + n;

        tmp_parts[dest] = parts[pp];
      }
    }
  }
}

__global__ void copy_ghost_bin_s_parts(part_struct_scalar *tmp_parts,
  part_struct_scalar *recv_parts, int nparts_recv, int offset, int plane, dom_struct *DOM)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index
  int dest;

  if (pp < nparts_recv) {
    dest = offset + pp;
    tmp_parts[dest] = recv_parts[pp];
  }
}

__global__ void BC_s_W_D(real *array, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    array[GCC_LOC(_dom.Gcc._isb, tj + 1, tk + 1, s1b, s2b)] = bc_s;
}

__global__ void BC_s_W_N(real *array, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    array[GCC_LOC(_dom.Gcc._isb, tj + 1, tk + 1, s1b, s2b)] = 
      array[GCC_LOC(_dom.Gcc._is, tj + 1, tk + 1, s1b, s2b)] - bc_s*_dom.dx;
}

__global__ void BC_s_E_D(real *array, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    array[GCC_LOC(_dom.Gcc._ieb, tj + 1, tk + 1, s1b, s2b)] = bc_s;
}

__global__ void BC_s_E_N(real *array, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    array[GCC_LOC(_dom.Gcc._ieb, tj + 1, tk + 1, s1b, s2b)] = 
      array[GCC_LOC(_dom.Gcc._ie, tj + 1, tk + 1, s1b, s2b)] + _dom.dx*bc_s;
}

__global__ void BC_s_N_D(real *array, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    array[GCC_LOC(ti + 1, _dom.Gcc._jeb, tk + 1, s1b, s2b)] = bc_s;
}

__global__ void BC_s_N_N(real *array, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    array[GCC_LOC(ti + 1, _dom.Gcc._jeb, tk + 1, s1b, s2b)] = 
      array[GCC_LOC(ti + 1, _dom.Gcc._je, tk + 1, s1b, s2b)] + _dom.dy*bc_s;
}

__global__ void BC_s_S_D(real *array, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    array[GCC_LOC(ti + 1, _dom.Gcc._jsb, tk + 1, s1b, s2b)] = bc_s;
}

__global__ void BC_s_S_N(real *array, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    array[GCC_LOC(ti + 1, _dom.Gcc._jsb, tk + 1, s1b, s2b)] = 
      array[GCC_LOC(ti + 1, _dom.Gcc._js, tk + 1, s1b, s2b)] - _dom.dy*bc_s;
}

__global__ void BC_s_B_D(real *array, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    array[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ksb, s1b, s2b)] = bc_s;
}

__global__ void BC_s_B_N(real *array, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    array[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ksb, s1b, s2b)] = 
      array[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ks, s1b, s2b)] - _dom.dz*bc_s;
}

__global__ void BC_s_T_D(real *array, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    array[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._keb, s1b, s2b)] = bc_s;
}

__global__ void BC_s_T_N(real *array, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    array[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._keb, s1b, s2b)] = 
      array[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ke, s1b, s2b)] + _dom.dz*bc_s;
}

__global__ void forcing_boussinesq_x(real alpha, real gx, real s_init, real *s, real *fx)
{
  int i, C0, C1;
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  if(tj < _dom.Gfx.jnb && tk < _dom.Gfx.knb) {
    for(i = _dom.Gfx._isb + 1; i <= _dom.Gfx._ieb - 1; i++) {
	  C0 = GCC_LOC(i-1, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
	  C1 = GCC_LOC(i  , tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      fx[GFX_LOC(i, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)]
        += - gx * alpha * (0.5*(s[C0]+s[C1]) - s_init);
    }
  }
}

__global__ void forcing_boussinesq_y(real alpha, real gy, real s_init, real *s, real *fy)
{
  int j, C0, C1;
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  if(tk < _dom.Gfy.knb && ti < _dom.Gfy.inb) {
    for(j = _dom.Gfy._jsb + 1; j <= _dom.Gfy._jeb - 1; j++) {
	  C0 = GCC_LOC(ti, j-1, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
	  C1 = GCC_LOC(ti, j  , tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      fy[GFY_LOC(ti, j, tk, _dom.Gfy.s1b, _dom.Gfy.s2b)]
        += - gy * alpha * (0.5*(s[C0]+s[C1]) - s_init);
    }
  }
}

__global__ void forcing_boussinesq_z(real alpha, real gz, real s_init, real *s, real *fz)
{
  int k, C0, C1;
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  if(ti < _dom.Gfz.inb && tj < _dom.Gfz.jnb) {
    for(k = _dom.Gfz._ksb + 1; k <= _dom.Gfz._keb - 1; k++) {
	  C0 = GCC_LOC(ti, tj, k-1, _dom.Gcc.s1b, _dom.Gcc.s2b);
	  C1 = GCC_LOC(ti, tj, k  , _dom.Gcc.s1b, _dom.Gcc.s2b);
      fz[GFZ_LOC(ti, tj, k, _dom.Gfz.s1b, _dom.Gfz.s2b)]
        += - gz * alpha * (0.5*(s[C0]+s[C1]) - s_init);
    }
  }
}

__global__ void copy_subdom_parts_with_scalar(part_struct *tmp_parts, part_struct *parts,
  part_struct_scalar *tmp_s_parts, part_struct_scalar *s_parts,
  int *bin_start, int *bin_count, int *part_ind, int *bin_offset)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;             // bin index
  int pp;               // particle index
  int dest;             // destination in tmp_parts

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.in && tj < _bins.Gcc.jn) {
    for (int k = _bins.Gcc._ks; k <= _bins.Gcc._ke; k++) {
      cbin = GFZ_LOC(ti + 1, tj + 1, k, s1b, s2b);

      for (int n = 0; n < bin_count[cbin]; n++) {
        pp = part_ind[bin_start[cbin] + n];
        dest = bin_offset[cbin] + n;

        tmp_parts[dest] = parts[pp];
        tmp_s_parts[dest] = s_parts[pp];
      }
    }
  }
}

__global__ void scalar_solve(int *phase, real *s0, real *s,
  real *conv, real *diff, real *conv0, real *diff0,
  real *u0, real *v0, real *w0, real D, real dt, real dt0)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  // working constants
  real ab0 = 0.5 * dt / dt0;
  real ab = 1. + ab0;
  int C, Cx0, Cx1, Cy0, Cy1, Cz0, Cz1;
  int fx0, fx1, fy0, fy1, fz0, fz1;
  real conv_x, conv_y, conv_z;
  real diff_x, diff_y, diff_z;

  // loop over x-plane
  if(tj <= _dom.Gcc._je && tk <= _dom.Gcc._ke) {
    for(int i = _dom.Gcc._is; i <= _dom.Gcc._ie; i++) {
      C   = GCC_LOC(i,   tj,   tk,   _dom.Gcc.s1b, _dom.Gcc.s2b);
      Cx0 = GCC_LOC(i-1, tj,   tk,   _dom.Gcc.s1b, _dom.Gcc.s2b);
      Cx1 = GCC_LOC(i+1, tj,   tk,   _dom.Gcc.s1b, _dom.Gcc.s2b);
      Cy0 = GCC_LOC(i,   tj-1, tk,   _dom.Gcc.s1b, _dom.Gcc.s2b);
      Cy1 = GCC_LOC(i,   tj+1, tk,   _dom.Gcc.s1b, _dom.Gcc.s2b);
      Cz0 = GCC_LOC(i,   tj,   tk-1, _dom.Gcc.s1b, _dom.Gcc.s2b);
      Cz1 = GCC_LOC(i,   tj,   tk+1, _dom.Gcc.s1b, _dom.Gcc.s2b);
      fx0 = GFX_LOC(i,   tj,   tk,   _dom.Gfx.s1b, _dom.Gfx.s2b);
      fx1 = GFX_LOC(i+1, tj,   tk,   _dom.Gfx.s1b, _dom.Gfx.s2b);
      fy0 = GFY_LOC(i,   tj,   tk,   _dom.Gfy.s1b, _dom.Gfy.s2b);
      fy1 = GFY_LOC(i,   tj+1, tk,   _dom.Gfy.s1b, _dom.Gfy.s2b);
      fz0 = GFZ_LOC(i,   tj,   tk,   _dom.Gfz.s1b, _dom.Gfz.s2b);
      fz1 = GFZ_LOC(i,   tj,   tk+1, _dom.Gfz.s1b, _dom.Gfz.s2b);

      // calculate the convection term
      conv_x = u0[fx1] * 0.5 * (s0[Cx1] + s0[C]) - u0[fx0] * 0.5 * (s0[C] + s0[Cx0]);
      conv_x = conv_x / _dom.dx;
      conv_y = v0[fy1] * 0.5 * (s0[Cy1] + s0[C]) - v0[fy0] * 0.5 * (s0[C] + s0[Cy0]);
      conv_y = conv_y / _dom.dy;
      conv_z = w0[fz1] * 0.5 * (s0[Cz1] + s0[C]) - w0[fz0] * 0.5 * (s0[C] + s0[Cz0]);
      conv_z = conv_z / _dom.dz;  
      conv[C] = conv_x + conv_y + conv_z;

      // calculate the diffusion term
      diff_x = D * (s0[Cx0] - 2.*s0[C] + s0[Cx1]) / _dom.dx / _dom.dx;
      diff_y = D * (s0[Cy0] - 2.*s0[C] + s0[Cy1]) / _dom.dy / _dom.dy;
      diff_z = D * (s0[Cz0] - 2.*s0[C] + s0[Cz1]) / _dom.dz / _dom.dz;
      diff[C] = diff_x + diff_y + diff_z;

      // Adams-Bashforth
      if(phase[C] == -1) {
        if(dt0 > 0) {
          s[C] = s0[C] + dt * (ab * diff[C] - ab0 * diff0[C] - (ab * conv[C] - ab0 * conv0[C]));
        } else {
          s[C] = s0[C] + dt * (diff[C] - conv[C]);
        }
      }
    }
  }
}

__global__ void scalar_check_nodes(part_struct *parts,
  part_struct_scalar *s_parts, BC_s *bc_s, dom_struct *DOM)
{
  int node = threadIdx.x;
  int part = blockIdx.x;

  /* Convert node (r, theta, phi) to (x, y, z) */
  real xp, yp, zp;  // Cartesian radial vector
  real x, y, z;     // Cartesian location of node
  rtp2xyz(s_parts[part].rs, _node_t[node], _node_p[node], &xp, &yp, &zp);

  /* shift from particle center */
  x = xp + parts[part].x;
  y = yp + parts[part].y;
  z = zp + parts[part].z;

  // start off with all -1's
  parts[part].nodes[node] = -1;

  /* check if the node is interfered with by a wall */
  // compute distance between node and walls
  // set equal to some number to identify which wall is interfering

  // We use <= for E,N,T and > for W,S,B -- allows us to do [start,end) on all 
  // subdomains regardless of bc
  parts[part].nodes[node] += (WEST_WALL + 1) *    // set equal to WEST_WALL...
              (x - _dom.xs < 0) *                 // if outside domain &
              (_dom.I == DOM->Is) *                // if edge domain & DIRICHLET
              (bc_s->sW == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (EAST_WALL + 1) * 
              (x - _dom.xe >= 0) *
              (_dom.I == DOM->Ie) *
              (bc_s->sE == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (SOUTH_WALL + 1) *
              (y - _dom.ys < 0) *
              (_dom.J == DOM->Js) *
              (bc_s->sS == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (NORTH_WALL + 1) *
              (y - _dom.ye >= 0) *
              (_dom.J == DOM->Je) *
              (bc_s->sN == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (BOTTOM_WALL + 1) *
              (z - _dom.zs < 0) *
              (_dom.K == DOM->Ks) *
              (bc_s->sB == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (TOP_WALL + 1) *
              (z - _dom.ze >= 0) *
              (_dom.K == DOM->Ke) *
              (bc_s->sT == DIRICHLET)*
              (parts[part].nodes[node] == -1);
}

__global__ void scalar_interpolate_nodes(real *s, real *ss,
  part_struct *parts, part_struct_scalar *s_parts, BC_s *bc_s)
{
  int node = threadIdx.x;
  int part = blockIdx.x;

  real ddx = 1. / _dom.dx;
  real ddy = 1. / _dom.dy;
  real ddz = 1. / _dom.dz;

  real sswall;

  int i, j, k;  // index of cells containing node
  int oob;      // out of bounds indicator, 1 if out of bounds else 0
  int C, Ce, Cw, Cn, Cs, Ct, Cb;  // cell indices
  real xx, yy, zz;  // Cartesian location of s

  // convert node (r, theta, phi) to (x, y, z)
  real xp, yp, zp;  // Cartesian radial vector
  real x, y, z;     // Cartesian location of node
  rtp2xyz(s_parts[part].rs, _node_t[node], _node_p[node], &xp, &yp, &zp);

  // shift from particle center
  x = xp + parts[part].x;
  y = yp + parts[part].y;
  z = zp + parts[part].z;

  /* Find index of cell containing node. */
  // Do this in GLOBAL coordinates so that magnitude of floating point error is
  // the same on each subdomain.
  real arg_x = (x - (_dom.xs - _dom.dx)) * ddx + _dom.Gcc.isb;
  real arg_y = (y - (_dom.ys - _dom.dy)) * ddy + _dom.Gcc.jsb;
  real arg_z = (z - (_dom.zs - _dom.dz)) * ddz + _dom.Gcc.ksb;

  /* Deal with floating point errors in position so we don't lose nodes */
  // Similar to bin_fill_{i,j,k}. If floor != round and round is "close enough"
  // to the nearest integer, use round instead. this ensures that all nodes are
  // accounted for between subdomains
  // Using global indices makes sure that the floating point representation
  // error is the same for each subdomain, since the magnitude of the index will
  // be the similar/the same.

  i = floor(arg_x);
  j = floor(arg_y);
  k = floor(arg_z);

  int round_x = lrint(arg_x);
  int round_y = lrint(arg_y);
  int round_z = lrint(arg_z);

  // Better way to do this? no if-statement... abs?
  if ((round_x != i) && (abs(round_x - arg_x) <= DBL_EPSILON)) {
    i = round_x;
  }
  if ((round_y != j) && (abs(round_y - arg_y) <= DBL_EPSILON)) {
    j = round_y;
  }
  if ((round_z != k) && (abs(round_z - arg_z) <= DBL_EPSILON)) {
    k = round_z;
  }

  // Convert back to LOCAL coodrinates
  i -= _dom.Gcc.isb;
  j -= _dom.Gcc.jsb;
  k -= _dom.Gcc.ksb;

  /* Interpolate Scalar */
  // Find if out-of-bounds -- 1 if oob, 0 if in bounds
  oob = i < _dom.Gcc._is || i >= _dom.Gcc._ie ||
        j < _dom.Gcc._js || j >= _dom.Gcc._je ||
        k < _dom.Gcc._ks || k >= _dom.Gcc._ke;

  // Correct indices so we don't have out-of-bounds reads
  // If out out bounds, we'll read good info but trash the results
  i += (_dom.Gcc._is - i) * (i < _dom.Gcc._is);
  j += (_dom.Gcc._js - j) * (j < _dom.Gcc._js);
  k += (_dom.Gcc._ks - k) * (k < _dom.Gcc._is);
  i += (_dom.Gcc._ie - i) * (i > _dom.Gcc._ie);
  j += (_dom.Gcc._je - j) * (j > _dom.Gcc._je);
  k += (_dom.Gcc._ke - k) * (k > _dom.Gcc._ke);

  // Cell-centered indices
  C = GCC_LOC(i, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Ce = GCC_LOC(i + 1, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Cw = GCC_LOC(i - 1, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Cn = GCC_LOC(i, j + 1, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Cs = GCC_LOC(i, j - 1, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Ct = GCC_LOC(i, j, k + 1, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Cb = GCC_LOC(i, j, k - 1, _dom.Gcc.s1b, _dom.Gcc.s2b);

  // Cartesian location of center of cell
  xx = (i - 0.5) * _dom.dx + _dom.xs;
  yy = (j - 0.5) * _dom.dy + _dom.ys;
  zz = (k - 0.5) * _dom.dz + _dom.zs;

  // perform tri-linear interpolation
  real dsdx = 0.5*(s[Ce] - s[Cw]) * ddx;
  real dsdy = 0.5*(s[Cn] - s[Cs]) * ddy;
  real dsdz = 0.5*(s[Ct] - s[Cb]) * ddz;
  ss[node + NNODES*part] = s[C] + dsdx*(x - xx) + dsdy*(y - yy) + dsdz*(z - zz)
    - s_parts[part].s;

  // set sswall equal to interfering wall s
  sswall = (parts[part].nodes[node] == WEST_WALL)  *bc_s->sWD
         + (parts[part].nodes[node] == EAST_WALL)  *bc_s->sED
         + (parts[part].nodes[node] == SOUTH_WALL) *bc_s->sSD
         + (parts[part].nodes[node] == NORTH_WALL) *bc_s->sND
         + (parts[part].nodes[node] == BOTTOM_WALL)*bc_s->sBD
         + (parts[part].nodes[node] == TOP_WALL)   *bc_s->sTD
         - s_parts[part].s;

  // set actual node value based on whether it is interfered with
  ss[node + NNODES*part] =
         (1 - oob) * ((parts[part].nodes[node] == -1) * ss[node + NNODES*part]
                    + (parts[part].nodes[node] <  -1) * sswall);
}

__global__ void scalar_lebedev_quadrature(part_struct *parts,
  part_struct_scalar *s_parts, int s_ncoeffs_max,
  real *ss, real *int_Ys_re, real *int_Ys_im)
{
  int part = blockIdx.x;
  int coeff = blockIdx.y;
  int node = threadIdx.x;

  if (coeff < s_parts[part].ncoeff) {
    /* Calculate integrand at each node */
    int j = part*NNODES*s_ncoeffs_max + coeff*NNODES + node;

    int n = _s_nn[coeff];
    int m = _s_mm[coeff];
    real theta = _node_t[node];
    real phi = _node_p[node];
    real N_nm = nnm(n, m);
    real P_nm = pnm(n, m, theta);

    // Precalculate things we use more than once
    real cmphi = cos(m * phi);
    real smphi = sin(m * phi);

    int stride = node + part*NNODES;

    int_Ys_re[j] = N_nm*P_nm*ss[stride]*cmphi;
    int_Ys_im[j] = -N_nm*P_nm*ss[stride]*smphi;

    __syncthreads();

    /* Compute partial sum of Lebedev quadrature (scalar product) */
    // put sum into first node position for each coeff for each particle
    if (node == 0) {
      int_Ys_re[j] *= _A1;
      int_Ys_im[j] *= _A1;
      for (int i = 1; i < 6; i++) {
        int_Ys_re[j] += _A1 * int_Ys_re[j+i];
        int_Ys_im[j] += _A1 * int_Ys_im[j+i];
      }
      for (int i = 6; i < 18; i++) {
        int_Ys_re[j] += _A2 * int_Ys_re[j+i];
        int_Ys_im[j] += _A2 * int_Ys_im[j+i];
      }
      for (int i = 18; i < 26; i++) {
        int_Ys_re[j] += _A3 * int_Ys_re[j+i];
        int_Ys_im[j] += _A3 * int_Ys_im[j+i];
      }
    }
  }
}

__global__ void pack_s_sums_e(real *sum_send_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    for (int ti = _bins.Gcc._ie; ti <= _bins.Gcc._ieb; ti++) {
      cbin = GFX_LOC(ti, tj, tk, s1b, s2b);
      c2b = tj + tk * s1b + (ti - _bins.Gcc._ie) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          sum_send_e[sp0] = int_Ys_re[psum_ind];
          sum_send_e[sp1] = int_Ys_im[psum_ind];
        }
      }
    } // loop over ti planes
  }
}

__global__ void pack_s_sums_w(real *sum_send_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    for (int ti = _bins.Gcc._isb; ti <= _bins.Gcc._is; ti++) {
      cbin = GFX_LOC(ti, tj, tk, s1b, s2b);
      c2b = tj + tk * s1b + (ti - _bins.Gcc._isb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          sum_send_w[sp0] = int_Ys_re[psum_ind];
          sum_send_w[sp1] = int_Ys_im[psum_ind];
        }
      }
    } // loop over ti
  }
}

__global__ void pack_s_sums_n(real *sum_send_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    for (int tj = _bins.Gcc._je; tj <= _bins.Gcc._jeb; tj++) {
      cbin = GFY_LOC(ti, tj, tk, s1b, s2b);
      c2b = tk + ti * s1b + (tj - _bins.Gcc._je) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          sum_send_n[sp0] = int_Ys_re[psum_ind];
          sum_send_n[sp1] = int_Ys_im[psum_ind];
        }
      }
    } // loop over tj planes
  }
}

__global__ void pack_s_sums_s(real *sum_send_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    for (int tj = _bins.Gcc._jsb; tj <= _bins.Gcc._js; tj++) {
      cbin = GFY_LOC(ti, tj, tk, s1b, s2b);
      c2b = tk + ti * s1b + (tj - _bins.Gcc._jsb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          sum_send_s[sp0] = int_Ys_re[psum_ind];
          sum_send_s[sp1] = int_Ys_im[psum_ind];
        }
      }
    } // loop over tj planes
  }
}

__global__ void pack_s_sums_t(real *sum_send_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    for (int tk = _bins.Gcc._ke; tk <= _bins.Gcc._keb; tk++) {
      cbin = GFZ_LOC(ti, tj, tk, s1b, s2b);
      c2b = ti + tj * s1b + (tk - _bins.Gcc._ke) * s2b;

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          sum_send_t[sp0] = int_Ys_re[psum_ind];
          sum_send_t[sp1] = int_Ys_im[psum_ind];
        }
      }
    } // loop over tk planes
  }
}

__global__ void pack_s_sums_b(real *sum_send_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    for (int tk = _bins.Gcc._ksb; tk <= _bins.Gcc._ks; tk++) {
      cbin = GFZ_LOC(ti, tj, tk, s1b, s2b);
      c2b = ti + tj * s1b + (tk - _bins.Gcc._ksb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          sum_send_b[sp0] = int_Ys_re[psum_ind];
          sum_send_b[sp1] = int_Ys_im[psum_ind];
        }
      }
    } // loop over tk planes
  }
}

__global__ void unpack_s_sums_e(real *sum_recv_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    for (int ti = _bins.Gcc._ie; ti <= _bins.Gcc._ieb; ti++) {
      cbin = GFX_LOC(ti, tj, tk, s1b, s2b);
      c2b = tj + tk * s1b + (ti - _bins.Gcc._ie) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          int_Ys_re[psum_ind] += sum_recv_e[sp0];
          int_Ys_im[psum_ind] += sum_recv_e[sp1];
        }
      }
    } // loop over ti
  }
}

__global__ void unpack_s_sums_w(real *sum_recv_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    for (int ti = _bins.Gcc._isb; ti <= _bins.Gcc._is; ti++) {
      cbin = GFX_LOC(ti, tj, tk, s1b, s2b);
      c2b = tj + tk * s1b + (ti - _bins.Gcc._isb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          int_Ys_re[psum_ind] += sum_recv_w[sp0];
          int_Ys_im[psum_ind] += sum_recv_w[sp1];
        }
      }
    } // loop over ti
  }
}

__global__ void unpack_s_sums_n(real *sum_recv_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    for (int tj = _bins.Gcc._je; tj <= _bins.Gcc._jeb; tj++) {
      cbin = GFY_LOC(ti, tj, tk, s1b, s2b);
      c2b = tk + ti * s1b + (tj - _bins.Gcc._je) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          int_Ys_re[psum_ind] += sum_recv_n[sp0];
          int_Ys_im[psum_ind] += sum_recv_n[sp1];
        }
      }
    } // loop over tj
  }
}

__global__ void unpack_s_sums_s(real *sum_recv_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    for (int tj = _bins.Gcc._jsb; tj <= _bins.Gcc._js; tj++) {
      cbin = GFY_LOC(ti, tj, tk, s1b, s2b);
      c2b = tk + ti * s1b + (tj - _bins.Gcc._jsb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          int_Ys_re[psum_ind] += sum_recv_s[sp0];
          int_Ys_im[psum_ind] += sum_recv_s[sp1];
        }
      }
    } // loop over tj
  }
}

__global__ void unpack_s_sums_t(real *sum_recv_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    for (int tk = _bins.Gcc._ke; tk <= _bins.Gcc._keb; tk++) {
      cbin = GFZ_LOC(ti, tj, tk, s1b, s2b);
      c2b = ti + tj * s1b + (tk - _bins.Gcc._ke) * s2b;

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          int_Ys_re[psum_ind] += sum_recv_t[sp0];
          int_Ys_im[psum_ind] += sum_recv_t[sp1];
        }
      }
    } // loop over tk
  }
}

__global__ void unpack_s_sums_b(real *sum_recv_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int s_ncoeffs_max,
  real *int_Ys_re, real *int_Ys_im)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, s)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    for (int tk = _bins.Gcc._ksb; tk <= _bins.Gcc._ks; tk++) {
      cbin = GFZ_LOC(ti, tj, tk, s1b, s2b);
      c2b = ti + tj * s1b + (tk - _bins.Gcc._ksb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < s_ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + s_ncoeffs_max*SP_YS_RE + s_ncoeffs_max*SNSP*dest;    // Ys_re
          sp1 = coeff + s_ncoeffs_max*SP_YS_IM + s_ncoeffs_max*SNSP*dest;    // Ys_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*s_ncoeffs_max + coeff*NNODES;

          int_Ys_re[psum_ind] += sum_recv_b[sp0];
          int_Ys_im[psum_ind] += sum_recv_b[sp1];
        }
      }
    } // loop over tk
  }
}

__device__ real X_sn(int n, real theta, real phi,
  int pp, part_struct_scalar *s_parts)
{
  int coeff = 0;
  for(int j = 0; j < n; j++) coeff += 2*j+1;

  real sum = 0.;
  for(int m = -n; m <= n; m++) {
    sum += Nnm(n,m)*Pnm(n,m,theta)
      *(s_parts[pp].anm_re[coeff]*cos(m*phi)
      - s_parts[pp].anm_im[coeff]*sin(m*phi));
    coeff++;
  }
  return sum;
}

__global__ void scalar_part_BC(real *s, int *phase, int *phase_shell,
  part_struct *parts, part_struct_scalar *s_parts)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int tj = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;
  int CC;
  real x, y, z;         // scalar node location Cartesian
  real Xp, Yp, Zp;      // particle position
  real r, theta, phi;   // velocity node location spherical
  real ss_tmp;          // temporary scalar
  int P, PS;            // phase, phase_shell
  real a;               // particle radius
  int order;            // particle scalar order
  real sp;              // particle scalar

  if (ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for (int k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      CC = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Position of current thread
      x = (ti-0.5) * _dom.dx + _dom.xs;
      y = (tj-0.5) * _dom.dy + _dom.ys;
      z = (k-0.5) * _dom.dz + _dom.zs;

      // get particle number and phase_shell number
      P = phase[CC];
      PS = phase_shell[CC];

      if(P > -1) {
        a = parts[P].r;
        Xp = parts[P].x;
        Yp = parts[P].y;
        Zp = parts[P].z;
        order = s_parts[P].order;
        sp = s_parts[P].s;
      } else {
        a = (_dom.dx + _dom.dy + _dom.dz) / 3.;
        Xp = (ti-0.5) * _dom.dx + _dom.xs + a;
        Yp = (tj-0.5) * _dom.dy + _dom.ys + a;
        Zp = (k-0.5) * _dom.dz + _dom.zs + a;
        // set order = -1, so it won't enter the "for loop"
        order = -1;
        sp = 0.;
      }

      // Position in particle frame
      x -= Xp;
      y -= Yp;
      z -= Zp;
      xyz2rtp(x, y, z, &r, &theta, &phi);

      // calculate analytic solution
      real ar = a / r;
      real ra = r / a;
      ss_tmp = sp;
      for(int n = 0; n <= order; n++) {
        ss_tmp += (pow(ra,n) - pow(ar,n+1)) * X_sn(n, theta, phi, P, s_parts);
      }

      // phase_shell = 1 means normal nodes, phase_shell = 0 means pressure nodes
      // phase shell cells: ss_tmp
      //       inner cells: sp, particle scalar
      //       fluid cells: s[CC]
      s[CC] = ss_tmp * (P > -1 && PS < 1)
            + sp     * (P > -1 && PS > 0)
            + s[CC]  * (P <= -1);
    }
  }
}

__global__ void scalar_compute_coeffs(part_struct *parts,
  part_struct_scalar *s_parts, int s_ncoeffs_max, int nparts,
  real *int_Ys_re, real *int_Ys_im)
{

  int coeff = threadIdx.x;
  int part = blockIdx.x;

  // precalculate constants
  real ars = parts[part].r / s_parts[part].rs;
  real rsa = s_parts[part].rs / parts[part].r;

  if (coeff < s_parts[part].ncoeff && part < nparts) {
    int j = part * NNODES * s_ncoeffs_max + coeff * NNODES + 0;
    int n = _s_nn[coeff];
    real A = pow(rsa, n) - pow(ars, n+1.);
    s_parts[part].anm_re[coeff] = int_Ys_re[j] / A;
    s_parts[part].anm_im[coeff] = int_Ys_im[j] / A;

    __syncthreads();

    // calculate heat flux for each particle
    if (coeff == 0) {
      s_parts[part].q = 2. * sqrt(PI) * parts[part].r * s_parts[part].anm_re[0];
    }
  }
}

__global__ void scalar_compute_error(real lamb_cut_scalar, int s_ncoeffs_max, int nparts,
  part_struct_scalar *s_parts, real *s_part_errors)
{
  int part = blockIdx.x;
  int coeff = threadIdx.x;

  real div = 0.;
  real max = DBL_MIN;

  __shared__ real s_coeffs[S_MAX_COEFFS * SNSP];
  __shared__ real s_coeffs0[S_MAX_COEFFS * SNSP];
  __shared__ real s_max[S_MAX_COEFFS];

  if (part < nparts && coeff < s_ncoeffs_max) {

    s_coeffs[coeff + s_ncoeffs_max * 0] = s_parts[part].anm_re[coeff];
    s_coeffs[coeff + s_ncoeffs_max * 1] = s_parts[part].anm_im[coeff];

    s_coeffs0[coeff + s_ncoeffs_max * 0] = s_parts[part].anm_re0[coeff];
    s_coeffs0[coeff + s_ncoeffs_max * 1] = s_parts[part].anm_im0[coeff];

    s_max[coeff] = DBL_MIN;

    __syncthreads();
    
    // If coefficient has a large enough magnitude (relative to 0th order coeff)
    //  calculate the error
    for (int i = 0; i < SNSP; i++) {
      int c = coeff + s_ncoeffs_max * i;

      // Determine if current coefficient has large enough value compared to 0th
      // (also, make sure it's large enough so we don't get issues with close-to-zero
      //  errors)
      // (also, if zeroth order is 0, ignore)
      real curr_val = s_coeffs[c];
      real zeroth_val = s_coeffs[0 + s_ncoeffs_max * i];
      int flag = (fabs(curr_val) > fabs(lamb_cut_scalar*zeroth_val)) *
                  (fabs(curr_val) > 1.e-16) *
                  (fabs(zeroth_val) > DBL_MIN);

      // If flag == 1, set scoeff equal to error value
      // If flag == 0, set scoeff equal to zero (no error)
      div = fabs(curr_val);
      div += (1.e-16 - div) * (div < 1.e-16);
      real curr_val0 = s_coeffs0[c];

      s_coeffs[c] = (real) flag * fabs(curr_val - curr_val0) / div;

      // See if current error is the max we've seen so far over all the
      // coefficients of a given order, set if so
      s_max[coeff] += (s_coeffs[c] - s_max[coeff]) * (s_coeffs[c] > s_max[coeff]);
    }

    __syncthreads();

    // We've now calculated the error for each "large enough" coefficients and
    //  found the maximum over all coefficients of a given order. Now, each
    //  order has a maximum, and we need to find the max over these
    if (coeff == 0) {
      for (int i = 0; i < s_ncoeffs_max; i++) {
        max += (s_max[i] - max) * (s_max[i] > max);  
      }
      s_part_errors[part] = max;
    }
  }
}

__global__ void scalar_store_coeffs(part_struct_scalar *s_parts, int nparts,
  int s_ncoeffs_max)
{
  int part = blockIdx.x;
  int coeff = threadIdx.x;
  if (part < nparts && coeff < s_ncoeffs_max) {
   s_parts[part].anm_re0[coeff] = s_parts[part].anm_re[coeff];
   s_parts[part].anm_im0[coeff] = s_parts[part].anm_im[coeff];
  }
}

__global__ void update_part_scalar(part_struct *parts,
  part_struct_scalar *s_parts, real time, real dt, real s_k)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index

  real vol = 4./3. * PI * parts[pp].r*parts[pp].r*parts[pp].r;
  real m = vol * parts[pp].rho;
  // prepare s for next timestep
  s_parts[pp].s += (float)s_parts[pp].update * s_parts[pp].q * s_k * dt / m /s_parts[pp].cp;
}

__global__ void scalar_part_fill(real *s, int *phase,
  part_struct_scalar *s_parts)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int tj = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;
  int CC, P;
  real sp = 0.;

  if (ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for (int k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      CC = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      P = phase[CC];
      if(P > -1) sp = s_parts[P].s;
      s[CC] = sp * (P > -1) + s[CC] * (P == -1);
    }
  }
}
