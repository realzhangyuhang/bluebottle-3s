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

__global__ void BC_s_W_D(real *s, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    s[GCC_LOC(_dom.Gcc._isb, tj + 1, tk + 1, s1b, s2b)] = bc_s;
}

__global__ void BC_s_W_N(real *s, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    s[GCC_LOC(_dom.Gcc._isb, tj + 1, tk + 1, s1b, s2b)] = 
      s[GCC_LOC(_dom.Gcc._is, tj + 1, tk + 1, s1b, s2b)] - bc_s*_dom.dx;
}

__global__ void BC_s_E_D(real *s, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    s[GCC_LOC(_dom.Gcc._ieb, tj + 1, tk + 1, s1b, s2b)] = bc_s;
}

__global__ void BC_s_E_N(real *s, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    s[GCC_LOC(_dom.Gcc._ieb, tj + 1, tk + 1, s1b, s2b)] = 
      s[GCC_LOC(_dom.Gcc._ie, tj + 1, tk + 1, s1b, s2b)] + _dom.dx*bc_s;
}

__global__ void BC_s_N_D(real *s, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    s[GCC_LOC(ti + 1, _dom.Gcc._jeb, tk + 1, s1b, s2b)] = bc_s;
}

__global__ void BC_s_N_N(real *s, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    s[GCC_LOC(ti + 1, _dom.Gcc._jeb, tk + 1, s1b, s2b)] = 
      s[GCC_LOC(ti + 1, _dom.Gcc._je, tk + 1, s1b, s2b)] + _dom.dy*bc_s;
}

__global__ void BC_s_S_D(real *s, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    s[GCC_LOC(ti + 1, _dom.Gcc._jsb, tk + 1, s1b, s2b)] = bc_s;
}

__global__ void BC_s_S_N(real *s, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    s[GCC_LOC(ti + 1, _dom.Gcc._jsb, tk + 1, s1b, s2b)] = 
      s[GCC_LOC(ti + 1, _dom.Gcc._js, tk + 1, s1b, s2b)] - _dom.dy*bc_s;
}

__global__ void BC_s_B_D(real *s, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    s[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ksb, s1b, s2b)] = bc_s;
}

__global__ void BC_s_B_N(real *s, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    s[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ksb, s1b, s2b)] = 
      s[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ks, s1b, s2b)] - _dom.dz*bc_s;
}

__global__ void BC_s_T_D(real *s, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    s[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._keb, s1b, s2b)] = bc_s;
}

__global__ void BC_s_T_N(real *s, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    s[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._keb, s1b, s2b)] = 
      s[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ke, s1b, s2b)] + _dom.dz*bc_s;
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
