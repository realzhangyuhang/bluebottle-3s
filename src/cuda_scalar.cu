#include <cuda.h>
#include <thrust/sort.h>

#include "cuda_particle.h"
#include "scalar.h"
#include "cuda_scalar.h"

#include <helper_cuda.h>
//#include <thrust/scan.h>
//#include <thrust/device_ptr.h>

__constant__ int _s_mm[64];
__constant__ int _s_nn[64];
real *_int_Ys_re;
real *_int_Ys_im;

extern "C"
void cuda_scalar_transfer_parts_i(void)
{
  //printf("N%d >> Transfering parts in i, nparts = %d\n", rank, nparts);
  /* Transfer particles east and west
   *  * Bin the particles, indexing with `i` varying slowest
   *  * Sort particles by their bin
   *  * Find start and end of each bin's particles
   *  * Find number of particles in each bin
   *  * Find number of particles in _is & _ie planes. These need to be sent W/E
   *  * Communicate these number east and west. Each process now knows how many
   *    to send and recv
   *  * Allocate memory for particle send and recv
   *  * Copy particles into sending arrays. Each bin can find the offset target
   *    index for its particles by performing a prefix scan.
   *  * Communicate particles east and west, send -> recv
   *  * Recv'd parts exist in the ghost bins and replace whatever existed there
   *    at the last time step. Sum the particles in _isb & _ieb and subtract
   *    from nparts. This, plus the number of particle recv'd from E/W, is the
   *    number of new particles
   *  * Allocate temp part structure to hold all new particles.
   *  * Reduce bin_count from _is->_ie to find nparts that we will keep
   *  * Prefix scan from _ie -> _ie to find offset index for particle copy to
   *    temp struct
   *  * Backfill recv'd particles to the end of the temp array
   *  * Repeat process for j, k to take care of edge, corner. Indexing will be
   *    different to take advantage of memory coalescence and the prefix scan
   *    two steps back
   */

  /* NOTE
   *  cuda-memcheck occasionally produces the error "bulk_kernel_by_value: an
   *  illegal memory address was encountered" error on a (thrust) call to
   *  cudaDeviceSynchronize. This doesn't seem to be reliably reproducible
   *  (occurs on any of the several thrust calls in this function). This does
   *  not seem to affect the results in any way, but should be further
   *  investigated. See bug id 008.
   */

  /* Init execution config -- thread over east/west faces */
  int ty = bins.Gcc.jnb * (bins.Gcc.jnb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.jnb >= MAX_THREADS_DIM);
  int tz = bins.Gcc.knb * (bins.Gcc.knb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.knb >= MAX_THREADS_DIM);

  int by = (int) ceil((real) bins.Gcc.jnb / (real) ty);
  int bz = (int) ceil((real) bins.Gcc.knb / (real) tz);

  dim3 bin_num_inb(by, bz);
  dim3 bin_dim_inb(ty, tz);

  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Declare things we might need */
  int s1b, s2b; // custom strides
  int offset;

  /* Allocate memory */
  // These are realloc'd every time
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_e;
  int *_offset_w;
  checkCudaErrors(cudaMalloc(&_offset_e, bins.Gcc.s2b_i * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_w, bins.Gcc.s2b_i * sizeof(int)));
  thrust::device_ptr<int> t_offset_e(_offset_e);
  thrust::device_ptr<int> t_offset_w(_offset_w);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  // If we have parts...
  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_i<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_i<<<bin_num_inb, bin_dim_inb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send, and packing offsets */
    s1b = bins.Gcc.jnb;
    s2b = s1b * bins.Gcc.knb;

    // East
    offset = GFX_LOC(bins.Gcc._ie, 0, 0, s1b, s2b);
    if (dom[rank].e != MPI_PROC_NULL) {
      // _bin_count is indexed with i varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _ie plane
      nparts_send[EAST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[EAST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_i, t_offset_e);
      } else {
        cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
      }

    } else { // no parts to send
      nparts_send[EAST] = 0;
      cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
    }

    // West
    offset = GFX_LOC(bins.Gcc._is, 0, 0, s1b, s2b);
    if (dom[rank].w != MPI_PROC_NULL) {
      nparts_send[WEST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());
      if (nparts_send[WEST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_i, t_offset_w);
      } else {
        cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
      }

    } else {
      nparts_send[WEST] = 0;
      cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
    }

  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[EAST] = 0;
    nparts_send[WEST] = 0;
    cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
    cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
  }

  /* Send number of parts to east/west */
  //    origin                target
  // nparts_send[WEST] -> nparts_recv[EAST]
  // nparts_recv[WEST] <- nparts_send[EAST]
  nparts_recv[WEST] = 0; // init
  nparts_recv[EAST] = 0;
  mpi_send_nparts_i();

  /* Allocate memory for send and receiving particles */
  // NOTE: If no particles need to be sent/received in a given direction, this
  //  allocates a memory location with size zero which returns a null device
  //  pointer. If this is passed to MPI_Win_create(base, ...) as the base in
  //  CUDA 9.0, it causes MPI to hang. This was not an issue in CUDA 7.5
  //
  // The fix involves fooling MPI by allocating a very small amount of dummy
  // information if no particles are to be sent. This gives the location a valid
  // memory pointer, than than a null pointer. The MPI communication still knows
  // that the allocated window size and info to be sent is zero, and nothing is
  // unpacked because that is wrapped in an if-statement already. This doesn't 
  // affect most cases where particles are communicated every direction at every
  // time; this will only affect extremely dilute cases.

  int send_alloc_e = nparts_send[EAST]*(nparts_send[EAST] > 0) + (nparts_send[EAST] == 0);
  int send_alloc_w = nparts_send[WEST]*(nparts_send[WEST] > 0) + (nparts_send[WEST] == 0);
  int recv_alloc_e = nparts_recv[EAST]*(nparts_recv[EAST] > 0) + (nparts_recv[EAST] == 0);
  int recv_alloc_w = nparts_recv[WEST]*(nparts_recv[WEST] > 0) + (nparts_recv[WEST] == 0);

  checkCudaErrors(cudaMalloc(&_send_parts_e, send_alloc_e * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_send_parts_w, send_alloc_w * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_e, recv_alloc_e * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_w, recv_alloc_w * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_send_s_parts_e, send_alloc_e * sizeof(part_struct_scalar)));
  checkCudaErrors(cudaMalloc(&_send_s_parts_w, send_alloc_w * sizeof(part_struct_scalar)));
  checkCudaErrors(cudaMalloc(&_recv_s_parts_e, recv_alloc_e * sizeof(part_struct_scalar)));
  checkCudaErrors(cudaMalloc(&_recv_s_parts_w, recv_alloc_w * sizeof(part_struct_scalar)));

  /* Pack particles into _send_parts */
  if (nparts_send[EAST] > 0) {
    pack_parts_e<<<bin_num_inb, bin_dim_inb>>>(_send_parts_e, _parts, _offset_e,
      _bin_start, _bin_count, _part_ind);
  } else {  // fill dummy data
    //cudaMemset(_send_parts_e, 0., send_alloc_e * sizeof(part_struct));
  }

  if (nparts_send[WEST] > 0) {
    pack_parts_w<<<bin_num_inb, bin_dim_inb>>>(_send_parts_w, _parts, _offset_w,
      _bin_start, _bin_count, _part_ind);
  } else {  // fill dummy data
    //cudaMemset(_send_parts_w, 0., send_alloc_w * sizeof(part_struct));
  }
  cudaDeviceSynchronize(); // To ensure packing is complete before sending

  /* Communicate particles with MPI */
  mpi_send_parts_i();

  // send particle scalar information
  if (nparts_send[EAST] > 0) {
    pack_s_parts_e<<<bin_num_inb, bin_dim_inb>>>(_send_s_parts_e, _s_parts, _offset_e,
      _bin_start, _bin_count, _part_ind);
  }
  if (nparts_send[WEST] > 0) {
    pack_s_parts_w<<<bin_num_inb, bin_dim_inb>>>(_send_s_parts_w, _s_parts, _offset_w,
      _bin_start, _bin_count, _part_ind);
  }
  cudaDeviceSynchronize();
  mpi_send_s_parts_i();

  /* Find number of particles currently in the EAST/WEST ghost bins */
  int nparts_ghost[6];

  if (nparts > 0) {
    // East
    offset = GFX_LOC(bins.Gcc._ieb, 0, 0, s1b, s2b);
    if (dom[rank].e != MPI_PROC_NULL) {
      nparts_ghost[EAST] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_i,
                                          0., thrust::plus<int>());
    } else {
      nparts_ghost[EAST] = 0;
    }

    // West
    offset = GFX_LOC(bins.Gcc._isb, 0, 0, s1b, s2b);
    if (dom[rank].w != MPI_PROC_NULL) {
      nparts_ghost[WEST] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_i,
                                          0., thrust::plus<int>());
    } else {
      nparts_ghost[WEST] = 0;
    }
  } else { // no parts
    nparts_ghost[EAST] = 0;
    nparts_ghost[WEST] = 0;
  }

  /* Calculate new number of particles */
  int nparts_old = nparts;
  nparts += nparts_recv[EAST] + nparts_recv[WEST] 
          - nparts_ghost[EAST] - nparts_ghost[WEST];

  /* allocate temporary part struct */
  part_struct *_tmp_parts;
  checkCudaErrors(cudaMalloc(&_tmp_parts, nparts * sizeof(part_struct)));
  part_struct_scalar *_tmp_s_parts;
  checkCudaErrors(cudaMalloc(&_tmp_s_parts, nparts * sizeof(part_struct_scalar)));

  if (nparts_old > 0) {
    /* parallel prefix scan of [_is, _ie] of _bin_count */
    int *_offset_all;
    checkCudaErrors(cudaMalloc(&_offset_all, bins.Gcc.s3b * sizeof(int)));
    thrust::device_ptr<int> t_offset_all(_offset_all);

    // Scan over bin_count[_is->_ie, j, k]
    int size = bins.Gcc.s3b - 2*bins.Gcc.s2b_i;
    
    thrust::exclusive_scan(t_bin_count + bins.Gcc.s2b_i,
                           t_bin_count + bins.Gcc.s2b_i + size,
                           t_offset_all + bins.Gcc.s2b_i);

    /* copy bins of particles to tmp_parts */
    copy_central_bin_parts_i<<<bin_num_inb, bin_dim_inb>>>(_tmp_parts, _parts,
      _bin_start, _bin_count, _part_ind, _offset_all);
	copy_central_bin_s_parts_i<<<bin_num_inb, bin_dim_inb>>>(_tmp_s_parts, _s_parts,
      _bin_start, _bin_count, _part_ind, _offset_all);

    cudaFree(_offset_all);

  } else { // no (old) parts
    // Do not need to copy or prefix scan
  }

  /* Copy ghost particles received from WEST */
  if (nparts_recv[WEST] > 0) {
    t_nparts = nparts_recv[WEST] * (nparts_recv[WEST] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[WEST] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[WEST] / (real) t_nparts);

    dim3 dim_nparts_w(t_nparts);
    dim3 num_nparts_w(b_nparts);

    offset = nparts_old - nparts_ghost[WEST] - nparts_ghost[EAST];
    copy_ghost_bin_parts<<<num_nparts_w, dim_nparts_w>>>(_tmp_parts, _recv_parts_w,
      nparts_recv[WEST], offset, WEST, _DOM);
	copy_ghost_bin_s_parts<<<num_nparts_w, dim_nparts_w>>>(_tmp_s_parts, _recv_s_parts_w,
      nparts_recv[WEST], offset, WEST, _DOM);
  } else { // nparts_recv[WEST] <= 0
    // Do nothing
  }

  /* Copy ghost particles received from EAST */
  if (nparts_recv[EAST] > 0) {
    t_nparts = nparts_recv[EAST] * (nparts_recv[EAST] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[EAST] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[EAST] / (real) t_nparts);

    dim3 dim_nparts_e(t_nparts);
    dim3 num_nparts_e(b_nparts);

    offset = nparts_old - nparts_ghost[WEST] - nparts_ghost[EAST] 
            + nparts_recv[WEST];
    copy_ghost_bin_parts<<<num_nparts_e, dim_nparts_e>>>(_tmp_parts, _recv_parts_e,
      nparts_recv[EAST], offset, EAST, _DOM);
    copy_ghost_bin_s_parts<<<num_nparts_e, dim_nparts_e>>>(_tmp_s_parts, _recv_s_parts_e,
      nparts_recv[EAST], offset, EAST, _DOM);
  } else { // npats_recv[EAST] <= 0
    // Do nothing
  }

  /* Swap pointers to _parts and _tmp_parts */
  part_struct *tmp = _parts;
  _parts = _tmp_parts;
  _tmp_parts = tmp;
  part_struct_scalar *s_tmp = _s_parts;
  _s_parts = _tmp_s_parts;
  _tmp_s_parts = s_tmp;

//  /* Correct ghost particle position for periodic boundaries */
//  int nparts_added = nparts_recv[EAST] + nparts_recv[WEST];
//  if (nparts_added > 0) {
//    t_nparts = nparts_added * (nparts_added < MAX_THREADS_1D)
//              + MAX_THREADS_1D * (nparts_added >= MAX_THREADS_1D);
//    b_nparts = (int) ceil((real) nparts_added / (real) t_nparts);
//
//    dim3 dim_nparts_a(t_nparts);
//    dim3 num_nparts_a(b_nparts);
//
//    offset = nparts_old - nparts_ghost[WEST] - nparts_ghost[EAST];
//    correct_periodic_boundaries_i<<<num_nparts_a, dim_nparts_a>>>(_parts, 
//      offset, nparts_added, _bc, _DOM);
//  }

  // Free memory
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_e);
  cudaFree(_offset_w);
  cudaFree(_send_parts_e);
  cudaFree(_send_parts_w);
  cudaFree(_recv_parts_e);
  cudaFree(_recv_parts_w);
  cudaFree(_send_s_parts_e);
  cudaFree(_send_s_parts_w);
  cudaFree(_recv_s_parts_e);
  cudaFree(_recv_s_parts_w);
  cudaFree(_tmp_parts);
  cudaFree(_tmp_s_parts);
}

extern "C"
void cuda_scalar_transfer_parts_j(void)
{
  // Steps are the same as in cuda_transfer_part_i, except we index with 'j'
  // varying the slowest

  /* Init execution config */

  // thread over north/south faces 
  int tz = bins.Gcc.knb * (bins.Gcc.knb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.knb >= MAX_THREADS_DIM);
  int tx = bins.Gcc.inb * (bins.Gcc.inb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.inb >= MAX_THREADS_DIM);

  int bz = (int) ceil((real) bins.Gcc.knb / (real) tz);
  int bx = (int) ceil((real) bins.Gcc.inb / (real) tx);

  dim3 bin_num_jnb(bz, bx);
  dim3 bin_dim_jnb(tz, tx);

  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Declare things we might need */
  int s1b, s2b; // custom strides
  int offset;

  /* Allocate memory */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_n;
  int *_offset_s;
  checkCudaErrors(cudaMalloc(&_offset_n, bins.Gcc.s2b_j * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_s, bins.Gcc.s2b_j * sizeof(int)));
  thrust::device_ptr<int> t_offset_n(_offset_n);
  thrust::device_ptr<int> t_offset_s(_offset_s);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  // If we have parts...
  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_j<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_j<<<bin_num_jnb, bin_dim_jnb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send, and packing offsets */
    s1b = bins.Gcc.knb;
    s2b = s1b * bins.Gcc.inb;
  
    // North
    offset = GFY_LOC(0, bins.Gcc._je, 0, s1b, s2b);
    if (dom[rank].n != MPI_PROC_NULL) {
      // _bin_count is indexed with j varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _je plane
      nparts_send[NORTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[NORTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_j, t_offset_n);
      } else {
        cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
      }
      
    } else {
      nparts_send[NORTH] = 0;
      cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
    }
  
    // South
    offset = GFY_LOC(0, bins.Gcc._js, 0, s1b, s2b);
    if (dom[rank].s != MPI_PROC_NULL) {
      nparts_send[SOUTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());

      if (nparts_send[SOUTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_j, t_offset_s);
      } else {
        cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
      }

    } else {
      nparts_send[SOUTH] = 0;
      cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
    }
  
  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[NORTH] = 0;
    nparts_send[SOUTH] = 0;
    cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
    cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
  }

  /* Send number of parts to north/south */
  nparts_recv[SOUTH] = 0; // init
  nparts_recv[NORTH] = 0;
  mpi_send_nparts_j();

  /* Allocate memory for send and receiving particles */
  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_n = nparts_send[NORTH]*(nparts_send[NORTH] > 0) + (nparts_send[NORTH] == 0);
  int send_alloc_s = nparts_send[SOUTH]*(nparts_send[SOUTH] > 0) + (nparts_send[SOUTH] == 0);
  int recv_alloc_n = nparts_recv[NORTH]*(nparts_recv[NORTH] > 0) + (nparts_recv[NORTH] == 0);
  int recv_alloc_s = nparts_recv[SOUTH]*(nparts_recv[SOUTH] > 0) + (nparts_recv[SOUTH] == 0);

  checkCudaErrors(cudaMalloc(&_send_parts_n, send_alloc_n * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_send_parts_s, send_alloc_s * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_n, recv_alloc_n * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_s, recv_alloc_s * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_send_s_parts_n, send_alloc_n * sizeof(part_struct_scalar)));
  checkCudaErrors(cudaMalloc(&_send_s_parts_s, send_alloc_s * sizeof(part_struct_scalar)));
  checkCudaErrors(cudaMalloc(&_recv_s_parts_n, recv_alloc_n * sizeof(part_struct_scalar)));
  checkCudaErrors(cudaMalloc(&_recv_s_parts_s, recv_alloc_s * sizeof(part_struct_scalar)));

  /* Pack particles into _send_parts */
  if (nparts_send[NORTH] > 0)  {
    pack_parts_n<<<bin_num_jnb, bin_dim_jnb>>>(_send_parts_n, _parts, _offset_n,
      _bin_start, _bin_count, _part_ind);
  } else { // fill dummy data
    //cudaMemset(_send_parts_n, 0., send_alloc_n * sizeof(part_struct));
  }

  if (nparts_send[SOUTH] > 0)  {
    pack_parts_s<<<bin_num_jnb, bin_dim_jnb>>>(_send_parts_s, _parts, _offset_s,
      _bin_start, _bin_count, _part_ind);
  } else { // fill dummy data
    //cudaMemset(_send_parts_s, 0., send_alloc_s * sizeof(part_struct));
  }
  cudaDeviceSynchronize(); // To ensure packing is complete before sending

  /* Communicate particles with MPI */
  mpi_send_parts_j();

  // send particle scalar information
  if (nparts_send[NORTH] > 0) {
    pack_s_parts_n<<<bin_num_jnb, bin_dim_jnb>>>(_send_s_parts_n, _s_parts, _offset_n,
      _bin_start, _bin_count, _part_ind);
  }
  if (nparts_send[SOUTH] > 0) {
    pack_s_parts_s<<<bin_num_jnb, bin_dim_jnb>>>(_send_s_parts_s, _s_parts, _offset_s,
      _bin_start, _bin_count, _part_ind);
  }
  cudaDeviceSynchronize();
  mpi_send_s_parts_j();

  /* Find number of particles currently in the NORTH/SOUTH ghost bins */
  int nparts_ghost[6];

  if (nparts > 0) {
    // North
    offset = GFY_LOC(0, bins.Gcc._jeb, 0, s1b, s2b);
    if (dom[rank].n != MPI_PROC_NULL) {
      nparts_ghost[NORTH] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + bins.Gcc.s2b_j,
                                           0., thrust::plus<int>());
    } else {
      nparts_ghost[NORTH] = 0;
    }

    // South
    offset = GFY_LOC(0, bins.Gcc._jsb, 0, s1b, s2b);
    if (dom[rank].s != MPI_PROC_NULL) {
      nparts_ghost[SOUTH] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + bins.Gcc.s2b_j,
                                           0., thrust::plus<int>());
    } else {
      nparts_ghost[SOUTH] = 0;
    }
  } else { // no parts
    nparts_ghost[NORTH] = 0;
    nparts_ghost[SOUTH] = 0;
  }

  /* Calculate new number of particles */
  int nparts_old = nparts;
  nparts += nparts_recv[NORTH] + nparts_recv[SOUTH] 
          - nparts_ghost[NORTH] - nparts_ghost[SOUTH];

  /* allocate temporary part struct */
  part_struct *_tmp_parts;
  checkCudaErrors(cudaMalloc(&_tmp_parts, nparts * sizeof(part_struct)));
  part_struct_scalar *_tmp_s_parts;
  checkCudaErrors(cudaMalloc(&_tmp_s_parts, nparts * sizeof(part_struct_scalar)));

  if (nparts_old > 0) {
    /* parallel prefix scan of ALL of _bin_count */
    int *_offset_all;
    checkCudaErrors(cudaMalloc(&_offset_all, bins.Gcc.s3b * sizeof(int)));

    // Scan over bin_count[i, _js->_je, k]
    int size = bins.Gcc.s3b - 2*bins.Gcc.s2b_j;
    thrust::device_ptr<int> t_offset_all(_offset_all);
    thrust::exclusive_scan(t_bin_count + bins.Gcc.s2b_j,
                           t_bin_count + bins.Gcc.s2b_j + size,
                           t_offset_all + bins.Gcc.s2b_j);


    /* copy bins of particles to tmp_parts */
    copy_central_bin_parts_j<<<bin_num_jnb, bin_dim_jnb>>>(_tmp_parts, _parts,
      _bin_start, _bin_count, _part_ind, _offset_all);
    copy_central_bin_s_parts_j<<<bin_num_jnb, bin_dim_jnb>>>(_tmp_s_parts, _s_parts,
      _bin_start, _bin_count, _part_ind, _offset_all);

    cudaFree(_offset_all);

  } else { // no (old) parts
    // Do nothing
  }

  /* Copy ghost particles recieved from SOUTH */
  if (nparts_recv[SOUTH] > 0) {
    t_nparts = nparts_recv[SOUTH] * (nparts_recv[SOUTH] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[SOUTH] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[SOUTH] / (real) t_nparts);

    dim3 dim_nparts_s(t_nparts);
    dim3 num_nparts_s(b_nparts);

    offset = nparts_old - nparts_ghost[SOUTH] - nparts_ghost[NORTH];
    copy_ghost_bin_parts<<<num_nparts_s, dim_nparts_s>>>(_tmp_parts, _recv_parts_s,
      nparts_recv[SOUTH], offset, SOUTH, _DOM);
    copy_ghost_bin_s_parts<<<num_nparts_s, dim_nparts_s>>>(_tmp_s_parts, _recv_s_parts_s,
      nparts_recv[SOUTH], offset, SOUTH, _DOM);
  } else { // nparts_recv[SOUTH] <= 0
    // Do nothing
  }

  /* Copy ghost particles received from NORTH */
  if (nparts_recv[NORTH] > 0) {
    t_nparts = nparts_recv[NORTH] * (nparts_recv[NORTH] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[NORTH] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[NORTH] / (real) t_nparts);

    dim3 dim_nparts_n(t_nparts);
    dim3 num_nparts_n(b_nparts);

    offset = nparts_old - nparts_ghost[SOUTH] - nparts_ghost[NORTH]
            + nparts_recv[SOUTH];
    copy_ghost_bin_parts<<<num_nparts_n, dim_nparts_n>>>(_tmp_parts, _recv_parts_n,
      nparts_recv[NORTH], offset, NORTH, _DOM);
    copy_ghost_bin_s_parts<<<num_nparts_n, dim_nparts_n>>>(_tmp_s_parts, _recv_s_parts_n,
      nparts_recv[NORTH], offset, NORTH, _DOM);
  } else { // nparts_recv[NORTH] <= 0
    // Do nothing
  }

  /* Swap pointers to _parts and _tmp_parts */
  part_struct *tmp = _parts;
  _parts = _tmp_parts;
  _tmp_parts = tmp;
  part_struct_scalar *s_tmp = _s_parts;
  _s_parts = _tmp_s_parts;
  _tmp_s_parts = s_tmp;

//  /* Correct ghost particle position for periodic boundaries */
//  int nparts_added = nparts_recv[NORTH] + nparts_recv[SOUTH];
//  if (nparts_added > 0) {
//    t_nparts = nparts_added * (nparts_added < MAX_THREADS_1D)
//              + MAX_THREADS_1D * (nparts_added >= MAX_THREADS_1D);
//    b_nparts = (int) ceil((real) nparts_added / (real) t_nparts);
//
//    dim3 dim_nparts_a(t_nparts);
//    dim3 num_nparts_a(b_nparts);
//
//    offset = nparts_old - nparts_ghost[SOUTH] - nparts_ghost[NORTH];
//    correct_periodic_boundaries_j<<<num_nparts_a, dim_nparts_a>>>(_parts, 
//      offset, nparts_added, _bc, _DOM);
//  }

  // Free memory
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_n);
  cudaFree(_offset_s);
  cudaFree(_send_parts_n);
  cudaFree(_send_parts_s);
  cudaFree(_recv_parts_n);
  cudaFree(_recv_parts_s);
  cudaFree(_send_s_parts_n);
  cudaFree(_send_s_parts_s);
  cudaFree(_recv_s_parts_n);
  cudaFree(_recv_s_parts_s);
  cudaFree(_tmp_parts);
  cudaFree(_tmp_s_parts);
}

extern "C"
void cuda_scalar_transfer_parts_k(void)
{
  // Steps are the same as in cuda_transfer_part_i, except we index with 'k'
  // varying the slowest

  /* Init execution config */

  // thread over top/bottom faces 
  int tx = bins.Gcc.inb * (bins.Gcc.inb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.inb >= MAX_THREADS_DIM);
  int ty = bins.Gcc.jnb * (bins.Gcc.jnb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.jnb >= MAX_THREADS_DIM);

  int bx = (int) ceil((real) bins.Gcc.inb / (real) tx);
  int by = (int) ceil((real) bins.Gcc.jnb / (real) ty);

  dim3 bin_num_knb(bx, by);
  dim3 bin_dim_knb(tx, ty);

  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Declare things we might need */
  int s1b = bins.Gcc.inb;
  int s2b = s1b * bins.Gcc.jnb;
  int offset;

  /* Allocate memory */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_t;
  int *_offset_b;
  checkCudaErrors(cudaMalloc(&_offset_t, bins.Gcc.s2b_k * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_b, bins.Gcc.s2b_k * sizeof(int)));
  thrust::device_ptr<int> t_offset_t(_offset_t);
  thrust::device_ptr<int> t_offset_b(_offset_b);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  // If we have parts...
  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }
    //_part_bin = thrust::raw_pointer_cast(t_part_bin);
    //_part_ind = thrust::raw_pointer_cast(t_part_ind);

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send, and packing offsets */
    // Top
    offset = GFZ_LOC(0, 0, bins.Gcc._ke, s1b, s2b);
    if (dom[rank].t != MPI_PROC_NULL) {
      // _bin_count is indexed with k varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _ke plane
      nparts_send[TOP] = thrust::reduce(t_bin_count + offset,
                                        t_bin_count + offset + bins.Gcc.s2b_k,
                                          0., thrust::plus<int>());
    
      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[TOP] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_k, t_offset_t);
      } else {
        cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
      }

    } else {
      nparts_send[TOP] = 0;
      cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
    }

    // Bottom
    offset = GFZ_LOC(0, 0, bins.Gcc._ks, s1b, s2b);
    if (dom[rank].b != MPI_PROC_NULL) {
      nparts_send[BOTTOM] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + bins.Gcc.s2b_k,
                                           0., thrust::plus<int>());

      if (nparts_send[BOTTOM] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_k, t_offset_b);
      } else {
        cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
      }

    } else {
      nparts_send[BOTTOM] = 0;
      cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
    }
    
  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[TOP] = 0;
    nparts_send[BOTTOM] = 0;
    cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
    cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
  }

  /* Send number of parts to top/bottom */
  nparts_recv[TOP] = 0; // init
  nparts_recv[BOTTOM] = 0;
  mpi_send_nparts_k();

  /* Allocate memory for send and receiving particles */
  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_t = nparts_send[TOP]*(nparts_send[TOP] > 0) + (nparts_send[TOP] == 0);
  int send_alloc_b = nparts_send[BOTTOM]*(nparts_send[BOTTOM] > 0) + (nparts_send[BOTTOM] == 0);
  int recv_alloc_t = nparts_recv[TOP]*(nparts_recv[TOP] > 0) + (nparts_recv[TOP] == 0);
  int recv_alloc_b = nparts_recv[BOTTOM]*(nparts_recv[BOTTOM] > 0) + (nparts_recv[BOTTOM] == 0);

  checkCudaErrors(cudaMalloc(&_send_parts_t, send_alloc_t * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_send_parts_b, send_alloc_b * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_t, recv_alloc_t * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_b, recv_alloc_b * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_send_s_parts_t, send_alloc_t * sizeof(part_struct_scalar)));
  checkCudaErrors(cudaMalloc(&_send_s_parts_b, send_alloc_b * sizeof(part_struct_scalar)));
  checkCudaErrors(cudaMalloc(&_recv_s_parts_t, recv_alloc_t * sizeof(part_struct_scalar)));
  checkCudaErrors(cudaMalloc(&_recv_s_parts_b, recv_alloc_b * sizeof(part_struct_scalar)));

  /* Pack particles into _send_parts */
  if (nparts_send[TOP] > 0) {
    pack_parts_t<<<bin_num_knb, bin_dim_knb>>>(_send_parts_t, _parts, _offset_t,
      _bin_start, _bin_count, _part_ind);
  } else {  // fill dummy data
    //cudaMemset(_send_parts_t, 0., send_alloc_t * sizeof(part_struct));
  }

  if (nparts_send[BOTTOM] > 0) {
    pack_parts_b<<<bin_num_knb, bin_dim_knb>>>(_send_parts_b, _parts, _offset_b,
      _bin_start, _bin_count, _part_ind);
  } else {  // fill dummy data
    //cudaMemset(_send_parts_b, 0., send_alloc_b * sizeof(part_struct));
  }
  cudaDeviceSynchronize(); // To ensure packing is complete before sending

  /* Communicate particles with MPI */
  mpi_send_parts_k();

  // send particle scalar information
  if (nparts_send[TOP] > 0) {
    pack_s_parts_t<<<bin_num_knb, bin_dim_knb>>>(_send_s_parts_t, _s_parts, _offset_t,
      _bin_start, _bin_count, _part_ind);
  }
  if (nparts_send[BOTTOM] > 0) {
    pack_s_parts_b<<<bin_num_knb, bin_dim_knb>>>(_send_s_parts_b, _s_parts, _offset_b,
      _bin_start, _bin_count, _part_ind);
  }
  cudaDeviceSynchronize();
  mpi_send_s_parts_k();

  /* Find number of particles currently in the TOP/BOTTOM ghost bins */
  int nparts_ghost[6];
  
  if (nparts > 0) {
    // TOP
    offset = GFZ_LOC(0, 0, bins.Gcc._keb, s1b, s2b);
    if (dom[rank].t != MPI_PROC_NULL) {
      nparts_ghost[TOP] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_k,
                                         0., thrust::plus<int>());
    } else {
      nparts_ghost[TOP] = 0;
    }

    // BOTTOM
    offset = GFZ_LOC(0, 0, bins.Gcc._ksb, s1b, s2b);
    if (dom[rank].b != MPI_PROC_NULL) {
      nparts_ghost[BOTTOM] = thrust::reduce(t_bin_count + offset,
                                            t_bin_count + offset + bins.Gcc.s2b_k,
                                            0., thrust::plus<int>());
    } else {
      nparts_ghost[BOTTOM] = 0;
    }
  } else { // no parts
    nparts_ghost[TOP] = 0;
    nparts_ghost[BOTTOM] = 0;
  }

  /* Calculate new number of particles */
  int nparts_old = nparts;
  nparts += nparts_recv[TOP] + nparts_recv[BOTTOM] 
          - nparts_ghost[TOP] - nparts_ghost[BOTTOM];

  /* allocate temporary part struct */
  part_struct *_tmp_parts;
  checkCudaErrors(cudaMalloc(&_tmp_parts, nparts * sizeof(part_struct)));
  part_struct_scalar *_tmp_s_parts;
  checkCudaErrors(cudaMalloc(&_tmp_s_parts, nparts * sizeof(part_struct_scalar)));

  if (nparts_old > 0) {
    /* parallel prefix scan of ALL of _bin_count */
    int *_offset_all;
    checkCudaErrors(cudaMalloc(&_offset_all, bins.Gcc.s3b * sizeof(int)));

    // Scan over bin_count[i, m, _ks->_ke]
    int size = bins.Gcc.s3b - 2*bins.Gcc.s2b_k;
    thrust::device_ptr<int> t_offset_all(_offset_all);
    thrust::exclusive_scan(t_bin_count + bins.Gcc.s2b_k,
                           t_bin_count + bins.Gcc.s2b_k + size,
                           t_offset_all + bins.Gcc.s2b_k);


    /* copy bins of particles to tmp_parts */
    copy_central_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_tmp_parts, _parts,
      _bin_start, _bin_count, _part_ind, _offset_all);
    copy_central_bin_s_parts_k<<<bin_num_knb, bin_dim_knb>>>(_tmp_s_parts, _s_parts,
      _bin_start, _bin_count, _part_ind, _offset_all);

    cudaFree(_offset_all);

  } else { // no (old) parts
    // Do nothing
  }

  /* Copy ghost particles recieved from BOTTOM */
  if (nparts_recv[BOTTOM] > 0) {
    t_nparts = nparts_recv[BOTTOM] * (nparts_recv[BOTTOM] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[BOTTOM] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[BOTTOM] / (real) t_nparts);

    dim3 dim_nparts_b(t_nparts);
    dim3 num_nparts_b(b_nparts);

    offset = nparts_old - nparts_ghost[BOTTOM] - nparts_ghost[TOP];
    copy_ghost_bin_parts<<<num_nparts_b, dim_nparts_b>>>(_tmp_parts, _recv_parts_b,
      nparts_recv[BOTTOM], offset, BOTTOM, _DOM);
    copy_ghost_bin_s_parts<<<num_nparts_b, dim_nparts_b>>>(_tmp_s_parts, _recv_s_parts_b,
      nparts_recv[BOTTOM], offset, BOTTOM, _DOM);
  } else { // nparts_recv[BOTTOM] <= 0
    // Do nothing
  }

  /* Copy ghost particles received from TOP */
  if (nparts_recv[TOP] > 0) {
    t_nparts = nparts_recv[TOP] * (nparts_recv[TOP] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[TOP] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[TOP] / (real) t_nparts);

    dim3 dim_nparts_t(t_nparts);
    dim3 num_nparts_t(b_nparts);

    offset = nparts_old - nparts_ghost[BOTTOM] - nparts_ghost[TOP]
            + nparts_recv[BOTTOM];
    copy_ghost_bin_parts<<<num_nparts_t, dim_nparts_t>>>(_tmp_parts, _recv_parts_t,
      nparts_recv[TOP], offset, TOP, _DOM);
    copy_ghost_bin_s_parts<<<num_nparts_t, dim_nparts_t>>>(_tmp_s_parts, _recv_s_parts_t,
      nparts_recv[TOP], offset, TOP, _DOM);
  } else { // nparts_recv[TOP] <= 0
    // Do nothing
  }

  /* Swap pointers to _parts and _tmp_parts */
  part_struct *tmp = _parts;
  _parts = _tmp_parts;
  _tmp_parts = tmp;
  part_struct_scalar *s_tmp = _s_parts;
  _s_parts = _tmp_s_parts;
  _tmp_s_parts = s_tmp;

//  /* Correct ghost particle position for periodic boundaries */
//  int nparts_added = nparts_recv[TOP] + nparts_recv[BOTTOM];
//  if (nparts_added > 0) {
//    t_nparts = nparts_added * (nparts_added < MAX_THREADS_1D)
//              + MAX_THREADS_1D * (nparts_added >= MAX_THREADS_1D);
//    b_nparts = (int) ceil((real) nparts_added / (real) t_nparts);
//
//    dim3 dim_nparts_a(t_nparts);
//    dim3 num_nparts_a(b_nparts);
//
//    offset = nparts_old - nparts_ghost[BOTTOM] - nparts_ghost[TOP];
//    correct_periodic_boundaries_k<<<num_nparts_a, dim_nparts_a>>>(_parts, 
//      offset, nparts_added, _bc, _DOM);
//   
//  }

  // Free memory
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_t);
  cudaFree(_offset_b);
  cudaFree(_send_parts_t);
  cudaFree(_send_parts_b);
  cudaFree(_recv_parts_t);
  cudaFree(_recv_parts_b);
  cudaFree(_send_s_parts_t);
  cudaFree(_send_s_parts_b);
  cudaFree(_recv_s_parts_t);
  cudaFree(_recv_s_parts_b);
  cudaFree(_tmp_parts);
  cudaFree(_tmp_s_parts);
}

extern "C"
void cuda_scalar_malloc_host(void)
{
  checkCudaErrors(cudaMallocHost(&s, dom[rank].Gcc.s3b * sizeof(real)));
    cpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&s0, dom[rank].Gcc.s3b * sizeof(real)));
    cpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&s_conv, dom[rank].Gcc.s3b * sizeof(real)));
    cpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&s_conv0, dom[rank].Gcc.s3b * sizeof(real)));
    cpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&s_diff, dom[rank].Gcc.s3b * sizeof(real)));
    cpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&s_diff0, dom[rank].Gcc.s3b * sizeof(real)));
    cpumem += dom[rank].Gcc.s3b * sizeof(real);
}

extern "C"
void cuda_scalar_malloc_dev(void)
{
  checkCudaErrors(cudaMalloc(&_bc_s, sizeof(BC_s)));
    gpumem += sizeof(BC_s);
  checkCudaErrors(cudaMemcpy(_bc_s, &bc_s, sizeof(BC_s), 
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&_s, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_s0, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_s_conv, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_s_conv0, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_s_diff, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_s_diff0, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);

  checkCudaErrors(cudaMemset(_s, 0., dom[rank].Gcc.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_s0, 0., dom[rank].Gcc.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_s_conv, 0., dom[rank].Gcc.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_s_conv0, 0., dom[rank].Gcc.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_s_diff, 0., dom[rank].Gcc.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_s_diff0, 0., dom[rank].Gcc.s3b * sizeof(real)));
}

extern "C"
void cuda_scalar_push(void)
{
  checkCudaErrors(cudaMemcpy(_s, s, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_s0, s0, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_s_conv, s_conv, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_s_conv0, s_conv0, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_s_diff, s_diff, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_s_diff0, s_diff0, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
}

extern "C"
void cuda_scalar_pull(void)
{
  checkCudaErrors(cudaMemcpy(s, _s, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_scalar_pull_debug(void)
{
  checkCudaErrors(cudaMemcpy(s_conv, _s_conv, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(s_diff, _s_diff, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_scalar_pull_restart(void)
{
  checkCudaErrors(cudaMemcpy(s0, _s0, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(s_conv0, _s_conv0, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(s_diff0, _s_diff0, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_scalar_free(void)
{
    checkCudaErrors(cudaFreeHost(s));
    checkCudaErrors(cudaFreeHost(s0));
    checkCudaErrors(cudaFreeHost(s_conv));
    checkCudaErrors(cudaFreeHost(s_conv0));
    checkCudaErrors(cudaFreeHost(s_diff));
    checkCudaErrors(cudaFreeHost(s_diff0));

    checkCudaErrors(cudaFree(_bc_s));

    checkCudaErrors(cudaFree(_s));
    checkCudaErrors(cudaFree(_s0));
    checkCudaErrors(cudaFree(_s_conv));
    checkCudaErrors(cudaFree(_s_conv0));
    checkCudaErrors(cudaFree(_s_diff));
    checkCudaErrors(cudaFree(_s_diff0));
}

extern "C"
void cuda_scalar_part_malloc_dev(void)
{
  if (NPARTS > 0 && SCALAR >= 1) {
    checkCudaErrors(cudaMalloc(&_s_parts, nparts * sizeof(part_struct_scalar)));
    gpumem += nparts * sizeof(part_struct_scalar);
  }
}

extern "C"
void cuda_scalar_part_push(void)
{
  if (NPARTS > 0 && SCALAR >= 1) {
    checkCudaErrors(cudaMemcpy(_s_parts, s_parts, nparts * sizeof(part_struct_scalar),
      cudaMemcpyHostToDevice));

    int s_nn[64] = {0,
                  1, 1, 1,
                  2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4, 4, 4, 4, 4,
                  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

    int s_mm[64] = {0,
                  -1, 0, 1,
                  -2, -1, 0, 1, 2,
                  -3, -2, -1, 0, 1, 2, 3,
                  -4, -3, -2, -1, 0, 1, 2, 3, 4,
                  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                  -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
                  -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

    checkCudaErrors(cudaMemcpyToSymbol(_s_mm, s_mm, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(_s_nn, s_nn, 64 * sizeof(int)));
  }
}

extern "C"
void cuda_scalar_part_pull_with_scalar(void)
{
  /* Declare temporary part structure and nparts_subdom */
  part_struct *_tmp_parts;
  part_struct_scalar *_tmp_s_parts;
  nparts_subdom = 0;

  /* Re-allocate memory */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  if (nparts > 0) {
    // Thread over nparts
    int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                  + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
    int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

    dim3 dim_nparts(t_nparts);
    dim3 num_nparts(b_nparts);

    // thread over top/bottom faces 
    int tx = bins.Gcc.inb * (bins.Gcc.inb < MAX_THREADS_DIM)
         + MAX_THREADS_DIM * (bins.Gcc.inb >= MAX_THREADS_DIM);
    int ty = bins.Gcc.jnb * (bins.Gcc.jnb < MAX_THREADS_DIM)
         + MAX_THREADS_DIM * (bins.Gcc.jnb >= MAX_THREADS_DIM);
    int tz = bins.Gcc.knb * (bins.Gcc.knb < MAX_THREADS_DIM)
         + MAX_THREADS_DIM * (bins.Gcc.knb >= MAX_THREADS_DIM);

    int bx = (int) ceil((real) bins.Gcc.inb / (real) tx);
    int by = (int) ceil((real) bins.Gcc.jnb / (real) ty);
    int bz = (int) ceil((real) bins.Gcc.knb / (real) tz);

    dim3 bin_num_inb(by, bz);
    dim3 bin_dim_inb(ty, tz);
    dim3 bin_num_jnb(bz, bx);
    dim3 bin_dim_jnb(tz, tx);
    dim3 bin_num_knb(bx, by);
    dim3 bin_dim_knb(tx, ty);

    /* Find each particle's bin */
    bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Set ghost bin count to zero (GFZ indexed) */
    zero_ghost_bins_i<<<bin_num_inb, bin_dim_inb>>>(_bin_count);
    zero_ghost_bins_j<<<bin_num_jnb, bin_dim_jnb>>>(_bin_count);
    zero_ghost_bins_k<<<bin_num_knb, bin_dim_knb>>>(_bin_count);

    /* Allocate memory to find bin offset target indices in tmp part_struct */
    int *_bin_offset;
    checkCudaErrors(cudaMalloc(&_bin_offset, bins.Gcc.s3b * sizeof(int)));

    /* Prefix scan _bin_count to find target indices in tmp part_struct */
    thrust::device_ptr<int> t_bin_count(_bin_count);
    thrust::device_ptr<int> t_bin_offset(_bin_offset);
    thrust::exclusive_scan(t_bin_count, t_bin_count + bins.Gcc.s3b, t_bin_offset);

    /* Reduce bin_count to find nparts in subdomain (ghost bins are zero'd) */
    nparts_subdom = thrust::reduce(t_bin_count, t_bin_count + bins.Gcc.s3b,
                                        0., thrust::plus<int>());

    /* Allocate new device part struct (no ghost particles) */
    checkCudaErrors(cudaMalloc(&_tmp_parts, nparts_subdom * sizeof(part_struct)));
    checkCudaErrors(cudaMalloc(&_tmp_s_parts, nparts_subdom * sizeof(part_struct_scalar)));

    /* Copy subdom parts to tmp part_struct (only in subdom, so [in, jn]) */
    // thread over inner bins (no ghost bins)
    tx = bins.Gcc.in * (bins.Gcc.in < MAX_THREADS_DIM)
     + MAX_THREADS_DIM * (bins.Gcc.in >= MAX_THREADS_DIM);
    ty = bins.Gcc.jn * (bins.Gcc.jn < MAX_THREADS_DIM)
     + MAX_THREADS_DIM * (bins.Gcc.jn >= MAX_THREADS_DIM);
    bx = (int) ceil((real) bins.Gcc.in / (real) tx);
    by = (int) ceil((real) bins.Gcc.jn / (real) ty);
    dim3 bin_num_kn(bx, by);
    dim3 bin_dim_kn(tx, ty);

    copy_subdom_parts_with_scalar<<<bin_num_kn, bin_dim_kn>>>(_tmp_parts, _parts,
      _tmp_s_parts, _s_parts, _bin_start, _bin_count, _part_ind, _bin_offset);

    cudaFree(_bin_offset);

  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_subdom = 0;
    checkCudaErrors(cudaMalloc(&_tmp_parts, nparts_subdom * sizeof(part_struct)));
    checkCudaErrors(cudaMalloc(&_tmp_s_parts, nparts_subdom * sizeof(part_struct_scalar)));
  }

  /* Allocate new host parts with nparts in subdom */
  free(parts);
  free(s_parts);
  parts = (part_struct*) malloc(nparts_subdom * sizeof(part_struct));
  s_parts = (part_struct_scalar*) malloc(nparts_subdom * sizeof(part_struct_scalar));

  // Pull from device
  checkCudaErrors(cudaMemcpy(parts, _tmp_parts, nparts_subdom * sizeof(part_struct),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(s_parts, _tmp_s_parts, nparts_subdom * sizeof(part_struct_scalar),
    cudaMemcpyDeviceToHost));

  // Free
  cudaFree(_tmp_parts);
  cudaFree(_tmp_s_parts);
  cudaFree(_part_ind);
  cudaFree(_part_bin);

  // Double check the number of particles is correct
  int reduce_parts = 0;
  MPI_Allreduce(&nparts_subdom, &reduce_parts, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (reduce_parts != NPARTS) {
    printf("N%d >> Something went wrong. NPARTS = %d, but %d exist\n",
      rank, NPARTS, reduce_parts);
    printf("N%d >> Has %d parts\n", rank, nparts_subdom);
    exit(EXIT_FAILURE);
  }
}

extern "C"
void cuda_scalar_part_free(void)
{
  if(SCALAR >= 1 && NPARTS > 0) {
    checkCudaErrors(cudaFree(_s_parts));
  }
}

extern "C"
void cuda_compute_boussinesq(void)
{
  forcing_boussinesq_x<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(s_alpha, g.x, s_init, _s, _f_x);
  forcing_boussinesq_y<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(s_alpha, g.y, s_init, _s, _f_y);
  forcing_boussinesq_z<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(s_alpha, g.z, s_init, _s, _f_z);
}

extern "C"
void cuda_scalar_BC(real *array)
{
  // Check whether each subdom boundary is an external boundary, then
  // apply the correct boundary conditions to all fields on that face

  // Only apply boundary conditions on the inner [*n x *n] plane, not the
  //  [*nb x *nb] -- this ensures we don't set the points that don't contain
  //  any solution, and we also don't set points twice

  /* WEST */
  if (dom[rank].w == MPI_PROC_NULL) {
    switch (bc_s.sW) {
      case DIRICHLET:
        BC_s_W_D<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array, bc_s.sWD);
        break;
      case NEUMANN:
        BC_s_W_N<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array, bc_s.sWN);
        break;
    }
  }

  /* EAST */
  if (dom[rank].e == MPI_PROC_NULL) {
    switch (bc_s.sE) {
      case DIRICHLET:
        BC_s_E_D<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array, bc_s.sED);
        break;
      case NEUMANN:
        BC_s_E_N<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array, bc_s.sEN);
        break;
    }
  }

  /* SOUTH */
  if (dom[rank].s == MPI_PROC_NULL) {
    switch (bc_s.sS) {
      case DIRICHLET:
        BC_s_S_D<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array, bc_s.sSD);
        break;
      case NEUMANN:
        BC_s_S_N<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array, bc_s.sSN);
        break;
    }
  }

  /* NORTH */
  if (dom[rank].n == MPI_PROC_NULL) {
    switch (bc_s.sN) {
      case DIRICHLET:
        BC_s_N_D<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array, bc_s.sND);
        break;
      case NEUMANN:
        BC_s_N_N<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array, bc_s.sNN);
        break;
    }
  }

  /* BOTTOM */
  if (dom[rank].b == MPI_PROC_NULL) {
    switch (bc_s.sB) {
      case DIRICHLET:
        BC_s_B_D<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array, bc_s.sBD);
        break;
      case NEUMANN:
        BC_s_B_N<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array, bc_s.sBN);
        break;
    }
  }

  /* TOP */
  if (dom[rank].t == MPI_PROC_NULL) {
    switch (bc_s.sT) {
      case DIRICHLET:
        BC_s_T_D<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array, bc_s.sTD);
        break;
      case NEUMANN:
        BC_s_T_N<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array, bc_s.sTN);
        break;
    }
  }
}

extern "C"
void cuda_scalar_part_BC(real *array)
{
  scalar_part_BC<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array,
    _phase, _phase_shell, _parts, _s_parts);
}

extern "C"
void cuda_scalar_part_fill(void)
{
  scalar_part_fill<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_s, _phase, _s_parts);
}

extern "C"
void cuda_scalar_solve(void)
{
  scalar_solve<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(_phase, _s0, _s,
    _s_conv, _s_diff, _s_conv0, _s_diff0, _u0, _v0, _w0, s_D, dt, dt0);
}

extern "C"
void cuda_scalar_partial_sum_i(void)
{
  //printf("N%d >> Communicating partial sums in i (nparts %d)\n", rank, nparts);
  /* Outline of communication of partial sums for Lebedev integration
   * 1) Finish local Lebedev integration in lebedev_quad<<<>>>. For a given
   *    scalar product, the partial sum for the jth coefficient of the nth
   *    particle is stored in: _int_someint[0 + NNODES*j + nparts*NNODES*n]
   * 2) All particles at the outermost two bin planes need their sums
   *    accumulated (e.g., (j,k) planes at _bins.Gcc.{_isb->_is,_ie->_ieb})
   * 3) Bin the particles using i indexing (find _bin_{start,end,count})
   * 4) Reduce _bin_count at _isb:_is, _ie:_ieb to find nparts_send_{e,w}
   * 5) Communicate nparts_send_{e,w} with adjacent subdomains to find
   *    nparts_recv_{w,e}
   * 6) Excl. prefix scan _bin_count over the _isb:_is, _ie:_ieb planes to find
   *    destination index for particle data packed into sending aray
   * 7) Allocate send array, int_send_{e,w} * 6 * sizeof(real). 6 comes from
   *    the number of integrals
   * 8) Allocate recv array, int_recv_{e,w} * 6 * sizeof(real).
   * 9) Communicate int_send_{e,w} to int_recv_{e,w}
   * 10)  Excl. prefix scan _bin_count over _isb:_is, _ie:_ieb planes to find unpacking
   *      incides - this already exists from earlier
   * 11)  Unpack and accumulate
   * 12)  Repeat for j, k
   */

  /* Initialize execution config */
  // Thread over east/west faces
  int ty = bins.Gcc.jnb * (bins.Gcc.jnb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.jnb >= MAX_THREADS_DIM);
  int tz = bins.Gcc.knb * (bins.Gcc.knb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.knb >= MAX_THREADS_DIM);

  int by = (int) ceil((real) bins.Gcc.jnb / (real) ty);
  int bz = (int) ceil((real) bins.Gcc.knb / (real) tz);

  dim3 bin_num_inb(by, bz);
  dim3 bin_dim_inb(ty, tz);

  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Declare things we might need */
  int s1b, s2b; // custom strides
  int offset;

  /* Allocate */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_e;
  int *_offset_w;
  checkCudaErrors(cudaMalloc(&_offset_e, 2 * bins.Gcc.s2b_i * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_w, 2 * bins.Gcc.s2b_i * sizeof(int)));
  thrust::device_ptr<int> t_offset_e(_offset_e);
  thrust::device_ptr<int> t_offset_w(_offset_w);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_i<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_i<<<bin_num_inb, bin_dim_inb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send and packing offsets */
    s1b = bins.Gcc.jnb;
    s2b = s1b * bins.Gcc.knb;

    // East: _ie and _ieb planes
    if (dom[rank].e != MPI_PROC_NULL) {
      // _bin_count is indexed with i varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _ie plane
      offset = GFX_LOC(bins.Gcc._ie, 0, 0, s1b, s2b);
      nparts_send[EAST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + 2 * bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[EAST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_i, t_offset_e);
      } else {
        cudaMemset(_offset_e, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
      }

    } else { // no parts to send
      nparts_send[EAST] = 0;
      cudaMemset(_offset_e, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
    }

    // West: _isb and _is planes
    if (dom[rank].w != MPI_PROC_NULL) {
      offset = GFX_LOC(bins.Gcc._isb, 0, 0, s1b, s2b);
      nparts_send[WEST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + 2 * bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());
      if (nparts_send[WEST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_i, t_offset_w);
      } else {
        cudaMemset(_offset_w, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
      }

    } else {
      nparts_send[WEST] = 0;
      cudaMemset(_offset_w, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
    }
  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[EAST] = 0;
    nparts_send[WEST] = 0;
    cudaMemset(_offset_e, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
    cudaMemset(_offset_w, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
  }

  // Sending and receiving is the same since the outer two bin planes are shared
  nparts_recv[EAST] = nparts_send[EAST];
  nparts_recv[WEST] = nparts_send[WEST];

  /* Send number of parts to east/west */
  //    origin                target
  // nparts_send[WEST] -> nparts_recv[EAST]
  // nparts_recv[WEST] <- nparts_send[EAST]
  //nparts_recv[WEST] = 0; // init
  //nparts_recv[EAST] = 0;
  //mpi_send_nparts_i();

  /* Allocate memory for send and recv partial sums */
  int npsums = SNSP * s_ncoeffs_max;  // 2 scalar products * ncoeffs
  // Indexing is, for example:
  //  _sum_send_e[coeff + ncoeffs_max*sp + ncoeffs_max*nsp*part_id]
  // where
  //  part_id = [0, nparts) and sp = [0, 2)
  //    0:  Ys_re     1:  Ys_im

  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_e = nparts_send[EAST]*(nparts_send[EAST] > 0) + (nparts_send[EAST] == 0);
  int send_alloc_w = nparts_send[WEST]*(nparts_send[WEST] > 0) + (nparts_send[WEST] == 0);
  int recv_alloc_e = nparts_recv[EAST]*(nparts_recv[EAST] > 0) + (nparts_recv[EAST] == 0);
  int recv_alloc_w = nparts_recv[WEST]*(nparts_recv[WEST] > 0) + (nparts_recv[WEST] == 0);

  checkCudaErrors(cudaMalloc(&_sum_send_e, send_alloc_e*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_send_w, send_alloc_w*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_e, recv_alloc_e*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_w, recv_alloc_w*npsums*sizeof(real)));

  /* Pack partial sums */
  if (nparts_send[EAST] > 0) {
    pack_s_sums_e<<<bin_num_inb, bin_dim_inb>>>(_sum_send_e, _offset_e,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  } else {
    //cudaMemset(_sum_send_e, 0., send_alloc_e * npsums * sizeof(real));
  }

  if (nparts_send[WEST] > 0) {
    pack_s_sums_w<<<bin_num_inb, bin_dim_inb>>>(_sum_send_w, _offset_w,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  } else {
    //cudaMemset(_sum_send_w, 0., send_alloc_w * npsums * sizeof(real));
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Communicate partial sums with MPI */
  mpi_send_s_psums_i();

  // Offsets are the same since they're over both ghost bins and edge bins
  /* Unpack and complete partial sums */
  if (nparts_recv[EAST] > 0) {
    unpack_s_sums_e<<<bin_num_inb, bin_dim_inb>>>(_sum_recv_e, _offset_e,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  }
  if (nparts_recv[WEST] > 0) {
    unpack_s_sums_w<<<bin_num_inb, bin_dim_inb>>>(_sum_recv_w, _offset_w,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Free */
  cudaFree(_sum_send_e);
  cudaFree(_sum_send_w);
  cudaFree(_sum_recv_e);
  cudaFree(_sum_recv_w);
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_e);
  cudaFree(_offset_w);
}

extern "C"
void cuda_scalar_partial_sum_j(void)
{
  //printf("N%d >> Communicating partial sums in j\n", rank);
  /* Initialize execution config */
  // Thread over north/south faces
  int tz = bins.Gcc.knb * (bins.Gcc.knb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.knb >= MAX_THREADS_DIM);
  int tx = bins.Gcc.inb * (bins.Gcc.inb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.inb >= MAX_THREADS_DIM);

  int bz = (int) ceil((real) bins.Gcc.knb / (real) tz);
  int bx = (int) ceil((real) bins.Gcc.inb / (real) tx);

  dim3 bin_num_jnb(bz, bx);
  dim3 bin_dim_jnb(tz, tx);

  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Declare things we might need */
  int s1b, s2b; // custom strides
  int offset;

  /* Allocate */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_n;
  int *_offset_s;
  checkCudaErrors(cudaMalloc(&_offset_n, 2 * bins.Gcc.s2b_j * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_s, 2 * bins.Gcc.s2b_j * sizeof(int)));
  thrust::device_ptr<int> t_offset_n(_offset_n);
  thrust::device_ptr<int> t_offset_s(_offset_s);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_j<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_j<<<bin_num_jnb, bin_dim_jnb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send and packing offsets */
    s1b = bins.Gcc.knb;
    s2b = s1b * bins.Gcc.inb;

    // North: _je and _jeb planes
    if (dom[rank].n != MPI_PROC_NULL) {
      // _bin_count is indexed with i varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _je plane
      offset = GFY_LOC(0, bins.Gcc._je, 0, s1b, s2b);
      nparts_send[NORTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + 2 * bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[NORTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_j, t_offset_n);
      } else {
        cudaMemset(_offset_n, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
      }

    } else { // no parts to send
      nparts_send[NORTH] = 0;
      cudaMemset(_offset_n, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
    }

    // South: _jsb and _js planes
    if (dom[rank].s != MPI_PROC_NULL) {
      offset = GFY_LOC(0, bins.Gcc._jsb, 0, s1b, s2b);
      nparts_send[SOUTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + 2 * bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());
      if (nparts_send[SOUTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_j, t_offset_s);
      } else {
        cudaMemset(_offset_s, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
      }

    } else {
      nparts_send[SOUTH] = 0;
      cudaMemset(_offset_s, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
    }
  } else { // nparts == 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[NORTH] = 0;
    nparts_send[SOUTH] = 0;
    cudaMemset(_offset_n, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
    cudaMemset(_offset_s, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
  }

  // Sending and receiving is the same since the outer two bin planes are shared
  nparts_recv[NORTH] = nparts_send[NORTH];
  nparts_recv[SOUTH] = nparts_send[SOUTH];

  /* Send number of parts to north/south */
  //    origin                target
  // nparts_send[SOUTH] -> nparts_recv[NORTH]
  // nparts_recv[SOUTH] <- nparts_send[NORTH]
  //nparts_recv[SOUTH] = 0; // init
  //nparts_recv[NORTH] = 0;
  //mpi_send_nparts_j();

  /* Allocate memory for send and recv partial sums */
  int npsums = SNSP * s_ncoeffs_max;  // 2 scalar products * ncoeffs
  // Indexing is, for example:
  //  _sum_send_n[coeff + ncoeffs_max*sp + ncoeffs_max*nsp*part_id]
  // where
  //  part_id = [0, nparts) and sp = [0, 2)
  //    0:  Ys_re     1:  Ys_im

  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_n = nparts_send[NORTH]*(nparts_send[NORTH] > 0) + (nparts_send[NORTH] == 0);
  int send_alloc_s = nparts_send[SOUTH]*(nparts_send[SOUTH] > 0) + (nparts_send[SOUTH] == 0);
  int recv_alloc_n = nparts_recv[NORTH]*(nparts_recv[NORTH] > 0) + (nparts_recv[NORTH] == 0);
  int recv_alloc_s = nparts_recv[SOUTH]*(nparts_recv[SOUTH] > 0) + (nparts_recv[SOUTH] == 0);

  checkCudaErrors(cudaMalloc(&_sum_send_n, send_alloc_n*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_send_s, send_alloc_s*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_n, recv_alloc_n*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_s, recv_alloc_s*npsums*sizeof(real)));

  /* Pack partial sums */
  if (nparts_send[NORTH] > 0) {
    pack_s_sums_n<<<bin_num_jnb, bin_dim_jnb>>>(_sum_send_n, _offset_n,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  } else {
    //cudaMemset(_sum_send_n, 0., send_alloc_n * npsums * sizeof(real));
  }

  if (nparts_send[SOUTH] > 0) {
    pack_s_sums_s<<<bin_num_jnb, bin_dim_jnb>>>(_sum_send_s, _offset_s,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  } else {
    //cudaMemset(_sum_send_s, 0., send_alloc_s * npsums * sizeof(real));
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Communicate partial sums with MPI */
  mpi_send_s_psums_j();

  // Offsets are the same since they're over both ghost bins and edge bins
  /* Unpack and complete partial sums */
  if (nparts_recv[NORTH] > 0) {
    unpack_s_sums_n<<<bin_num_jnb, bin_dim_jnb>>>(_sum_recv_n, _offset_n,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  }
  if (nparts_recv[SOUTH] > 0) {
    unpack_s_sums_s<<<bin_num_jnb, bin_dim_jnb>>>(_sum_recv_s, _offset_s,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Free */
  cudaFree(_sum_send_n);
  cudaFree(_sum_send_s);
  cudaFree(_sum_recv_n);
  cudaFree(_sum_recv_s);
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_n);
  cudaFree(_offset_s);
}

extern "C"
void cuda_scalar_partial_sum_k(void)
{
  //printf("N%d >> Communicating partial sums in k\n", rank);
  /* Initialize execution config */
  // Thread over top/bottom faces
  int tx = bins.Gcc.inb * (bins.Gcc.inb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.inb >= MAX_THREADS_DIM);
  int ty = bins.Gcc.jnb * (bins.Gcc.jnb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.jnb >= MAX_THREADS_DIM);

  int bx = (int) ceil((real) bins.Gcc.inb / (real) tx);
  int by = (int) ceil((real) bins.Gcc.jnb / (real) ty);

  dim3 bin_num_knb(bx, by);
  dim3 bin_dim_knb(tx, ty);

  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Declare things we might need */
  int s1b, s2b; // custom strides
  int offset;

  /* Allocate */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_t;
  int *_offset_b;
  checkCudaErrors(cudaMalloc(&_offset_t, 2 * bins.Gcc.s2b_k * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_b, 2 * bins.Gcc.s2b_k * sizeof(int)));
  thrust::device_ptr<int> t_offset_t(_offset_t);
  thrust::device_ptr<int> t_offset_b(_offset_b);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send and packing offsets */
    s1b = bins.Gcc.inb;
    s2b = s1b * bins.Gcc.jnb;

    // North: _ke and _keb planes
    if (dom[rank].t != MPI_PROC_NULL) {
      // _bin_count is indexed with k varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _ke plane
      offset = GFZ_LOC(0, 0, bins.Gcc._ke, s1b, s2b);
      nparts_send[TOP] = thrust::reduce(t_bin_count + offset,
                                        t_bin_count + offset + 2 * bins.Gcc.s2b_k,
                                        0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[TOP] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_k, t_offset_t);
      } else {
        cudaMemset(_offset_t, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
      }

    } else { // no parts to send
      nparts_send[TOP] = 0;
      cudaMemset(_offset_t, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
    }

    // South: _ksb and _ks planes
    if (dom[rank].b != MPI_PROC_NULL) {
      offset = GFZ_LOC(0, 0, bins.Gcc._ksb, s1b, s2b);
      nparts_send[BOTTOM] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + 2 * bins.Gcc.s2b_k,
                                           0., thrust::plus<int>());
      if (nparts_send[BOTTOM] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_k, t_offset_b);
      } else {
        cudaMemset(_offset_b, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
      }

    } else {
      nparts_send[BOTTOM] = 0;
      cudaMemset(_offset_b, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
    }
  } else { // nparts = 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[TOP] = 0;
    nparts_send[BOTTOM] = 0;
    cudaMemset(_offset_t, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
    cudaMemset(_offset_b, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
  }

  // Sending and receiving is the same since the outer two bin planes are shared
  nparts_recv[TOP] = nparts_send[TOP];
  nparts_recv[BOTTOM] = nparts_send[BOTTOM];

  /* Send number of parts to top/bottom */
  //    origin                target
  // nparts_send[BOTTOM] -> nparts_recv[TOP]
  // nparts_recv[BOTTOM] <- nparts_send[TOP]
  //nparts_recv[BOTTOM] = 0; // init
  //nparts_recv[TOP] = 0;
  //mpi_send_nparts_k();

  /* Allocate memory for send and recv partial sums */
  int npsums = SNSP * s_ncoeffs_max;  // 2 scalar products * ncoeffs
  // Indexing is, for example:
  //  _sum_send_t[coeff + ncoeffs_max*sp + ncoeffs_max*nsp*part_id]
  // where
  //  part_id = [0, nparts) and sp = [0, 2)
  //    0:  Ys_re     1:  Ys_im

  int send_alloc_t = nparts_send[TOP]*(nparts_send[TOP] > 0) + (nparts_send[TOP] == 0);
  int send_alloc_b = nparts_send[BOTTOM]*(nparts_send[BOTTOM] > 0) + (nparts_send[BOTTOM] == 0);
  int recv_alloc_t = nparts_recv[TOP]*(nparts_recv[TOP] > 0) + (nparts_recv[TOP] == 0);
  int recv_alloc_b = nparts_recv[BOTTOM]*(nparts_recv[BOTTOM] > 0) + (nparts_recv[BOTTOM] == 0);

  checkCudaErrors(cudaMalloc(&_sum_send_t, send_alloc_t*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_send_b, send_alloc_b*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_t, recv_alloc_t*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_b, recv_alloc_b*npsums*sizeof(real)));

  /* Pack partial sums */
  if (nparts_send[TOP] > 0) {
    pack_s_sums_t<<<bin_num_knb, bin_dim_knb>>>(_sum_send_t, _offset_t,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  } else {
    //cudaMemset(_sum_send_t, 0., send_alloc_t * npsums * sizeof(real));
  }

  if (nparts_send[BOTTOM] > 0) {
    pack_s_sums_b<<<bin_num_knb, bin_dim_knb>>>(_sum_send_b, _offset_b,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  } else {
    //cudaMemset(_sum_send_b, 0., send_alloc_b * npsums * sizeof(real));
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Communicate partial sums with MPI */
  mpi_send_s_psums_k();

  // Offsets are the same since they're over both ghost bins and edge bins
  /* Unpack and complete partial sums */
  if (nparts_recv[TOP] > 0) {
    unpack_s_sums_t<<<bin_num_knb, bin_dim_knb>>>(_sum_recv_t, _offset_t,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  }
  if (nparts_recv[BOTTOM] > 0) {
    unpack_s_sums_b<<<bin_num_knb, bin_dim_knb>>>(_sum_recv_b, _offset_b,
      _bin_start, _bin_count, _part_ind, s_ncoeffs_max,
      _int_Ys_re, _int_Ys_im);
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Free */
  cudaFree(_sum_send_t);
  cudaFree(_sum_send_b);
  cudaFree(_sum_recv_t);
  cudaFree(_sum_recv_b);
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_t);
  cudaFree(_offset_b);
}

extern "C"
void cuda_scalar_lamb(void)
{
  /* CUDA exec config */
  dim3 num_parts(nparts); // nparts blocks with nnodes threads each
  dim3 dim_nodes(NNODES);
  dim3 num_partcoeff(nparts, s_ncoeffs_max);
  dim3 dim_coeff(s_ncoeffs_max);

  //printf("N%d >> Determining Lamb's coefficients (nparts = %d)\n", rank, nparts);
  if (nparts > 0) {
    /* Temp storage for field variables at quadrature nodes */
    real *_ss;    // scalar
    checkCudaErrors(cudaMalloc(&_ss, NNODES * nparts * sizeof(real)));

    /* Interpolate field varaibles to quadrature nodes */
    scalar_check_nodes<<<num_parts, dim_nodes>>>(_parts, _s_parts, _bc_s, _DOM);
    scalar_interpolate_nodes<<<num_parts, dim_nodes>>>(_s, _ss,
      _parts, _s_parts, _bc_s);

    /* Create scalar product storage using max particle coefficient size */
    int sp_size = nparts * NNODES * s_ncoeffs_max;
    checkCudaErrors(cudaMalloc(&_int_Ys_re, sp_size * sizeof(real)));
    checkCudaErrors(cudaMalloc(&_int_Ys_im, sp_size * sizeof(real)));

    /* Perform partial sums of lebedev quadrature */
    scalar_lebedev_quadrature<<<num_partcoeff, dim_nodes>>>(_parts,
      _s_parts, s_ncoeffs_max, _ss, _int_Ys_re, _int_Ys_im);

    checkCudaErrors(cudaFree(_ss));
  }

  /* Accumulate partial sums (all procs need to be involved) */
  cuda_scalar_partial_sum_i();  // 2a) Calculate partial sums over x face
  cuda_scalar_partial_sum_j();  // 2b) Calculate partial sums over y face
  cuda_scalar_partial_sum_k();  // 2c) Calculate partial sums over z face

  if (nparts > 0) {
    /* Compute lambs coefficients from partial sums */
    scalar_compute_coeffs<<<num_parts, dim_coeff>>>(_parts, _s_parts,
      s_ncoeffs_max, nparts, _int_Ys_re, _int_Ys_im);

    /* Free */
    checkCudaErrors(cudaFree(_int_Ys_re));
    checkCudaErrors(cudaFree(_int_Ys_im));
  }
}

extern "C"
real cuda_scalar_lamb_err(void)
{
  //printf("N%d >> Determining Lamb's error\n", rank);
  real error = DBL_MIN;
  if (nparts > 0) {
    // create a place to store errors
    real *_part_errors;
    cudaMalloc((void**) &_part_errors, nparts*sizeof(real));
    
    // sort the coefficients and calculate errors along the way
    dim3 numBlocks(nparts);
    dim3 dimBlocks(s_ncoeffs_max);

    scalar_compute_error<<<numBlocks, dimBlocks>>>(lamb_cut_scalar,
     s_ncoeffs_max, nparts, _s_parts, _part_errors);

    // find maximum error of all particles
    thrust::device_ptr<real> t_part_errors(_part_errors);
    error = thrust::reduce(t_part_errors,
                           t_part_errors + nparts,
                           0., thrust::maximum<real>());

    // clean up
    cudaFree(_part_errors);

    // store copy of coefficients for future calculation
    scalar_store_coeffs<<<numBlocks, dimBlocks>>>(_s_parts, nparts, s_ncoeffs_max);
  }

  // MPI reduce to find max error
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, mpi_real, MPI_MAX, MPI_COMM_WORLD);
  return error;
}

extern "C"
void cuda_store_s(void)
{
  checkCudaErrors(cudaMemcpy(_s0, _s, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(_s_conv0, _s_conv, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(_s_diff0, _s_diff, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToDevice));
}

void cuda_scalar_update_part(void)
{
  if(nparts > 0) {
    int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                  + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
    int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

    dim3 dim_nparts(t_nparts);
    dim3 num_nparts(b_nparts);
    update_part_scalar<<<num_nparts, dim_nparts>>>(_parts, _s_parts, ttime, dt, s_k);
  }
}
