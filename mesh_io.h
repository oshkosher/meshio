#ifndef __MPI_MESH_IO_H__
#define __MPI_MESH_IO_H__

/* Mesh IO - a small wrapper around MPI-IO to efficiently read and write mesh data.

   Ed Karrels, edk@illinois.edu, 2016
*/

#include <mpi.h>


#define MESH_IO_LITTLE_ENDIAN 0  /* Data file is little-endian.
                                    Swap bytes if CPU is not. */
#define MESH_IO_BIG_ENDIAN 1     /* Data file is big-endian.
                                    Swap bytes if CPU is not. */
#define MESH_IO_IGNORE_ENDIAN 2  /* Don't worry about endianness;
                                    do no swapping */
#define MESH_IO_SWAP_ENDIAN 3    /* Data file is opposite endian; 
                                    do swap bytes. */



/* Read an n-dimensional mesh from a file.
   This is a collective call. All the ranks that opened the file 'fh'
   must call this collectively.
   The data in the file is assumed to be in contiguous, canonical order.

   The data read may be a subset of the mesh encoded in the file, and it
   may be a subset of the mesh encoded in memory. 

   For example, let the file contain a 6x10 mesh of data. We wish to
   read the upper-right quadrant into an in-memory array with 1 halo
   cell on each side.

   file:
   . . . . . d d d d d
   . . . . . d d d d d
   . . . . . d d d d d
   . . . . . . . . . .
   . . . . . . . . . .
   . . . . . . . . . .

   memory:
   . . . . . . .
   . d d d d d .
   . d d d d d .
   . d d d d d .
   . . . . . . .

   mesh_size = {3, 5};
   file_mesh_sizes = {6, 10};
   file_mesh_starts = {0, 5};
   memory_mesh_sizes = {5, 7};
   memory_mesh_starts = {1, 1}

   fh - Handle of the file from which data will be read.
   offset - Offset (in bytes) from the beginning of the file where the first
     byte of the full mesh can be found.
   etype - datatype of each element in the mesh. If file_endian is
     MESH_IO_IGNORE_ENDIAN, then any datatype is allowed. Otherwise,
     only basic datatypes are supported.
   file_endian - one of four values:
     MESH_IO_LITTLE_ENDIAN - Data file is little-endian.
                             Swap bytes if CPU is not.
     MESH_IO_BIG_ENDIAN - Data file is big-endian.
                          Swap bytes if CPU is not.
     MESH_IO_IGNORE_ENDIAN - Don't worry about endianness; do no swapping.
     MESH_IO_SWAP_ENDIAN   - Data file is opposite endian; swap bytes.
   ndims - number of array dimensions (positive integer)
   buf - location of the mesh in memory
   mesh_sizes - number of elements in each dimension of the mesh being read
     (array of positive integers). These elements will be a subset of the
   file_mesh_sizes - number of elements in each dimension of the mesh
     as it is stored in the file.
   file_mesh_starts - number of elements in each dimension by which the
     submesh being read is inset from the origin of the file mesh.
   file_array_order - storage order of the mesh on disk.
     MPI_ORDER_C for row-major, MPI_ORDER_FORTRAN for column-major.
   memory_mesh_sizes - number of elements in each dimension of the full
     in-memory mesh
   memory_mesh_starts - number of elements in each dimension by which the
     submesh being written to memory is inset from the origin of the
     memory mesh.
   memory_array_order - storage order of the mesh in memory.
     MPI_ORDER_C for row-major, MPI_ORDER_FORTRAN for column-major.

  Returns:
    MPI_SUCCESS on success
    MPI_ERR_TYPE if etype is not a supported datatype
    MPI_ERR_BUFFER if buf is NULL
    MPI_ERR_DIMS if ndims is nonpositive
    MPI_ERR_ARG if any of the array bounds are invalid or the datatype
      has a nonpositive size or odd size (other than 1).
    MPI_ERR_TRUNCATE if no data is read from the file
    If any MPI call fails, this returns the error code from that call.
*/
int Mesh_IO_read
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 void *buf,
 int ndims,
 const int *mesh_sizes,
 const int *file_mesh_sizes,
 const int *file_mesh_starts,
 int file_array_order,
 const int *memory_mesh_sizes,
 const int *memory_mesh_starts,
 int memory_array_order);


/* Write an n-dimensional mesh to a file.
   The arguments have the same meanings as those of Mesh_IO_read(),
   except the data will be written to the file rather than read.

   'buf' is not const because when byte-swapping (to correct an endian
   mismatch) a large amount of data, it may need to be byte-swapped
   in place rather than by allocating a secondary buffer. The data will
   be byte-swapped again before the function returns, so the data is left
   unchanged.
*/
int Mesh_IO_write
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 void *buf,
 int ndims,
 const int *mesh_sizes,
 const int *file_mesh_sizes,
 const int *file_mesh_starts,
 int file_array_order,
 const int *memory_mesh_sizes,
 const int *memory_mesh_starts,
 int memory_array_order);


typedef void (*Mesh_IO_traverse_fn)(void *p, size_t count, void *param);

/* Traverse the mesh defined by {ndims, full_mesh_sizes,
   sub_mesh_sizes, sub_mesh_starts, array_order}.

   rowFunction() will be called on each contiguous submesh, and it will
   be given a pointer to the first element of the contiguous submesh
   and the value rowFunctionParam.

   Returns the number of elements processed.
*/
size_t Mesh_IO_traverse
(void *buf, size_t element_size, int ndims,
 const int *full_mesh_sizes, const int *sub_mesh_sizes,
 const int *sub_mesh_starts, int array_order,
 Mesh_IO_traverse_fn rowFunction, void *rowFunctionParam);

size_t Mesh_IO_endian_swap_in_place
(void *buf, size_t element_size, int ndims,
 const int *full_mesh_sizes, const int *sub_mesh_sizes,
 const int *sub_mesh_starts, int array_order);

size_t Mesh_IO_copy_to_linear_array
(void *dest, const void *src, size_t element_size, int ndims,
 const int *full_mesh_sizes, const int *sub_mesh_sizes,
 const int *sub_mesh_starts, int array_order);



#endif /* __MPI_MESH_IO_H__ */
