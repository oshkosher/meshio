/* Mesh IO - a smaller wrapper around MPI-IO to efficiently read and
   write mesh data.

   Ed Karrels, edk@illinois.edu, 2016
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "mesh_io.h"
#include "reverse_bytes.h"

/* to cut down argument length, package up the arguments for
   Mesh_IO_traverseOneDim */
typedef struct {
  int ndims;

  /* for all i > first_full_dim, sub_mesh_sizes[i] == full_mesh_sizes[i] */
  int first_full_dim;

  size_t element_size;
  const int *sub_mesh_sizes, *full_mesh_sizes, *sub_mesh_starts;

  /* {full,sub}_mesh_sizes[i] = product of {full,sub}_mesh_sizes[i+1..ndim-1] */
  const int *full_cum_sizes, *sub_cum_sizes;

  Mesh_IO_traverse_fn rowFunction;
  void *rowFunctionParam;

} MeshTraverseParams;


static int *createReversedArrayCopy(int len, const int *array);


/* Returns nonzero iff t is a type supported by Mesh_IO_read() */
static int isSupportedType(MPI_Datatype t);

/* Returns 1 iff the current architecture is big-endian. */
static int isBigEndian();

/* Common initialization routine for Mesh_IO_read() and Mesh_IO_write().
   Checks for argument errors.  Sets file_type, memory_type, and
   element_size. */
static int readWriteInit
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 int ndims,
 const int *mesh_sizes,
 const int *file_mesh_sizes,
 const int *file_mesh_starts,
 const int *memory_mesh_sizes,
 const int *memory_mesh_starts,
 int array_order,
 MPI_Datatype *file_type,
 MPI_Datatype *memory_type,
 size_t *element_size);

static int isEndianSwapNeeded(int file_endian);
 

/* Read an n-dimensional mesh from a file.

   fh - Handle of the file from which data will be read.
   offset - Offset (in bytes) from the beginning of the file where the first
     byte of the full mesh can be found.
   etype - datatype of each element in the mesh. If file_endian is
     MESH_IO_IGNORE_ENDIAN, then any datatype is allowed. Otherwise,
     only basic datatypes are supported.
   file_is_big_endian - one of four values:
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
   memory_mesh_sizes - number of elements in each dimension of the full
     in-memory mesh
   memory_mesh_starts - number of elements in each dimension by which the
     submesh being written to memory is inset from the origin of the
     memory mesh.
   order - storage order of the mesh in memory.
     MPI_ORDER_C for row-major, MPI_ORDER_FORTRAN for column-major.
   comm - the MPI communicator use to open the file fh

  Returns:
    MPI_SUCCESS on success
    MPI_ERR_TYPE if etype is not a supported datatype
    MPI_ERR_BUFFER if buf is NULL
    MPI_ERR_DIMS if ndims is nonpositive
    MPI_ERR_ARG if:
      any of the array bounds are invalid, or
      the datatype has a nonpositive size or odd size (other than 1), or
      if file_endian is not one of the expected values.
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
 const int *memory_mesh_sizes,
 const int *memory_mesh_starts,
 int order,
 MPI_Comm comm) {

  int i, err = MPI_SUCCESS, rank;
  MPI_Datatype file_type, memory_type;
  size_t element_size;
  MPI_Status status;

  MPI_Comm_rank(comm, &rank);

  /* check for argument errors and initialize datatypes */
  err = readWriteInit(fh, offset, etype, file_endian, ndims, mesh_sizes,
                      file_mesh_sizes, file_mesh_starts, 
                      memory_mesh_sizes, memory_mesh_starts, order,
                      &file_type, &memory_type, &element_size);
  if (err != MPI_SUCCESS) return err;

  /* read the mesh */
  err = MPI_File_read_at_all(fh, 0, buf, 1, memory_type, &status);
  if (err != MPI_SUCCESS) goto fail;

  /* no data was read */
  MPI_Get_count(&status, memory_type, &i);
  if (i != 1) {
    err = MPI_ERR_TRUNCATE;
    goto fail;
  }

  /* Fix the endianness of the data */
  if (isEndianSwapNeeded(file_endian)) {
    Mesh_IO_endian_swap_in_place
      (buf, element_size, ndims,
       memory_mesh_sizes, mesh_sizes,
       memory_mesh_starts, order);
  }
  
  /* free my datatypes */
 fail:
  MPI_Type_free(&memory_type);
  MPI_Type_free(&file_type);

  return err;
}


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
 const int *memory_mesh_sizes,
 const int *memory_mesh_starts,
 int order,
 MPI_Comm comm) {

  int i, err = MPI_SUCCESS, rank, doEndianSwap;
  MPI_Datatype file_type, memory_type;
  size_t element_size, element_count;
  MPI_Status status;
  void *temp_buffer = NULL;

  MPI_Comm_rank(comm, &rank);

  /* compute the total number of elements */
  element_count = 1;
  for (i=0; i < ndims; i++) element_count *= mesh_sizes[i];

  doEndianSwap = isEndianSwapNeeded(file_endian);

  /* Check for argument errors and initialize datatypes.
     If bytes need to be swapped; don't create the memory datatype. */
  err = readWriteInit(fh, offset, etype, file_endian, ndims, mesh_sizes,
                      file_mesh_sizes, file_mesh_starts,
                      memory_mesh_sizes, memory_mesh_starts, order,
                      &file_type,
                      doEndianSwap ? NULL : &memory_type,
                      &element_size);
  if (err != MPI_SUCCESS) {
    /* printf("readWriteInit error %d\n", err); */
    return err;
  }


  if (doEndianSwap) {
    /* copy the data into a temp buffer and swap the bytes there. */
    /* XXX if the data gets large, rather then copying the data this 
       routine could swap the data in place */
    temp_buffer = malloc(element_count * element_size);
    if (!temp_buffer) {
      err = MPI_ERR_NO_MEM;
      goto fail;
    }

    /* XXX optimization: the byte swapping could be done at the same
       time the data is copied to the buffer. */
    
    Mesh_IO_copy_to_linear_array
      (temp_buffer, buf, element_size, ndims, memory_mesh_sizes,
       mesh_sizes, memory_mesh_starts, order);

    reverseBytes(temp_buffer, element_size, element_count);
    
    MPI_Type_contiguous(element_count, etype, &memory_type);
    MPI_Type_commit(&memory_type);

    buf = temp_buffer;
  }

  /* write the mesh */
  err = MPI_File_write_at_all(fh, 0, buf, 1, memory_type, &status);
  /*if (err != MPI_SUCCESS) goto fail; */
  if (err != MPI_SUCCESS) {
    printf("write_at_all error %d\n", err);
    goto fail;
  }

  /* no data was read */
  MPI_Get_count(&status, memory_type, &i);
  if (i != 1) {
    err = MPI_ERR_TRUNCATE;
    goto fail;
  }

 fail:
  free(temp_buffer);
  MPI_Type_free(&memory_type);
  MPI_Type_free(&file_type);

  return err;
}

/* Common initialization routine for Mesh_IO_read() and Mesh_IO_write().
   Checks for argument errors.  Sets file_type, memory_type, and
   element_size. */
static int readWriteInit
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 int ndims,
 const int *mesh_sizes,
 const int *file_mesh_sizes,
 const int *file_mesh_starts,
 const int *memory_mesh_sizes,
 const int *memory_mesh_starts,
 int order,
 MPI_Datatype *file_type,
 MPI_Datatype *memory_type,
 size_t *element_size) {

  int i, err = MPI_SUCCESS;
  
  /* check for argument errors */
  if (ndims < 1) return MPI_ERR_DIMS;

  if (file_endian < MESH_IO_LITTLE_ENDIAN ||
      file_endian > MESH_IO_SWAP_ENDIAN)
    return MPI_ERR_ARG;

  /* if endianness is to be considered, check for a basic datatype */
  if (file_endian != MESH_IO_IGNORE_ENDIAN
      && !isSupportedType(etype)) return MPI_ERR_TYPE;

  /* check for mesh boundary errors */
  for (i=0; i < ndims; i++) {
    /*
    assert(!(file_mesh_starts[i] < 0));
    assert(!(file_mesh_starts[i] + mesh_sizes[i] > file_mesh_sizes[i]));
    assert(!(memory_mesh_starts[i] < 0));
    assert(!(memory_mesh_starts[i] + mesh_sizes[i] > memory_mesh_sizes[i]));
    assert(!(!(order == MPI_ORDER_FORTRAN ||
             order == MPI_ORDER_C)));
    */
    if (file_mesh_starts[i] < 0
        || file_mesh_starts[i] + mesh_sizes[i] > file_mesh_sizes[i]
        || memory_mesh_starts[i] < 0
        || memory_mesh_starts[i] + mesh_sizes[i] > memory_mesh_sizes[i]
        || !(order == MPI_ORDER_FORTRAN || order == MPI_ORDER_C)
        ) {
      return MPI_ERR_ARG;
    }
  }

  /* get the size of each element */
  MPI_Type_size(etype, &i);
  *element_size = i;
  /* complain if it's nonpositive */
  if (*element_size < 1)
    return MPI_ERR_ARG;
  /* if endian swapping is to be done, complain if the size is larger
     than 1 and odd */
  if (file_endian != MESH_IO_IGNORE_ENDIAN
      && *element_size > 1
      && (*element_size & 1))
    return MPI_ERR_ARG;    
  
  /* define the shape of the mesh on disk (with no offset) */
  /* printf("[%02d] on disk subarray {%d,%d,%d} {%d,%d,%d} {%d,%d,%d}\n", rank,
         file_mesh_sizes[0], file_mesh_sizes[1], file_mesh_sizes[2],
         mesh_sizes[0], mesh_sizes[1], mesh_sizes[2],
         file_mesh_starts[0], file_mesh_starts[1], file_mesh_starts[2]); */
  err = MPI_Type_create_subarray
    (ndims, file_mesh_sizes, mesh_sizes, file_mesh_starts, order,
     etype, file_type);
  if (err != MPI_SUCCESS) goto fail0;

  err = MPI_Type_commit(file_type);
  if (err != MPI_SUCCESS) goto fail1;

  /* set the file view */
  err = MPI_File_set_view
    (fh, offset, *file_type, *file_type, "native", MPI_INFO_NULL);
  if (err != MPI_SUCCESS) goto fail1;

  /* define the shape of the mesh in memory */
  /* printf("[%02d] in memory subarray {%d,%d,%d} {%d,%d,%d} {%d,%d,%d}\n", rank,
         memory_mesh_sizes[0], memory_mesh_sizes[1], memory_mesh_sizes[2],
         mesh_sizes[0], mesh_sizes[1], mesh_sizes[2],
         memory_mesh_starts[0], memory_mesh_starts[1], memory_mesh_starts[2]); */
  if (memory_type != NULL) {
    err = MPI_Type_create_subarray
      (ndims, memory_mesh_sizes, mesh_sizes, memory_mesh_starts,
       order, etype, memory_type);
    if (err != MPI_SUCCESS) goto fail1;
    
    err = MPI_Type_commit(memory_type);
    if (err != MPI_SUCCESS) goto fail2;
  }

  return MPI_SUCCESS;
  
  /* free my datatypes */
 fail2:
  MPI_Type_free(memory_type);
 fail1:
  MPI_Type_free(file_type);

 fail0:
  return err;
}  


/* Returns nonzero iff t is a type supported by Mesh_IO_read() */
static int isSupportedType(MPI_Datatype t) {
  return (t == MPI_CHAR
          || t == MPI_SHORT
          || t == MPI_INT
          || t == MPI_LONG
          || t == MPI_LONG_LONG_INT
          || t == MPI_LONG_LONG
          || t == MPI_SIGNED_CHAR
          || t == MPI_UNSIGNED_CHAR
          || t == MPI_UNSIGNED_SHORT
          || t == MPI_UNSIGNED
          || t == MPI_UNSIGNED_LONG
          || t == MPI_UNSIGNED_LONG_LONG
          || t == MPI_FLOAT
          || t == MPI_DOUBLE
          || t == MPI_LONG_DOUBLE
          || t == MPI_WCHAR
          || t == MPI_C_BOOL
          || t == MPI_INT8_T
          || t == MPI_INT16_T
          || t == MPI_INT32_T
          || t == MPI_INT64_T
          || t == MPI_UINT8_T
          || t == MPI_UINT16_T
          || t == MPI_UINT32_T
          || t == MPI_UINT64_T
          || t == MPI_AINT
          || t == MPI_COUNT  /* not supported on older versions? (Vulkan) */
          || t == MPI_OFFSET);
}

/* Returns 1 iff the current architecture is big-endian. */
static int isBigEndian() {
  int tmp = 1;
  return ! *(char*) &tmp;
}


static int isEndianSwapNeeded(int file_endian) {
  switch (file_endian) {
  case MESH_IO_LITTLE_ENDIAN: return isBigEndian();
  case MESH_IO_BIG_ENDIAN: return !isBigEndian();
  case MESH_IO_IGNORE_ENDIAN: return 0;
  case MESH_IO_SWAP_ENDIAN: return 1;
  default: /* shouldn't happen */ return 0;
  }
}


/* Helper function used inside Mesh_IO_traverse; calls itself recursively
   for each dimension. */
static void Mesh_IO_traverseOneDim(int dim_no, char *buf,
                                   const MeshTraverseParams *p) {
  int i, start=p->sub_mesh_starts[dim_no], count = p->sub_mesh_sizes[dim_no];
  buf += p->element_size * start * p->full_cum_sizes[dim_no];

  if (dim_no+1 >= p->first_full_dim) {

    /* call the user function on each contigous chunk */
    p->rowFunction(buf, count * p->sub_cum_sizes[dim_no], p->rowFunctionParam);

  } else {
    for (i = start; i < start + count; i++) {
      Mesh_IO_traverseOneDim(dim_no+1, buf, p);
      buf += p->element_size * p->full_cum_sizes[dim_no];
    }
  }
}
  

/* Traverse an n-dimensional array, calling the given function on
   each contiguous segment of data. */
size_t Mesh_IO_traverse
(void *buf, size_t element_size, int ndims,
 const int *full_mesh_sizes, const int *sub_mesh_sizes,
 const int *sub_mesh_starts, int array_order,
 Mesh_IO_traverse_fn rowFunction, void *rowFunctionParam) {

  int i, first_full_dim, *full_cum_sizes, *sub_cum_sizes;

  /* do nothing if the element size is less than 1 */
  if (element_size < 1) return 0;

  /* reverse the dimension order if it's column-major */
  if (array_order != MPI_ORDER_C) {
    sub_mesh_sizes = createReversedArrayCopy(ndims, sub_mesh_sizes);
    full_mesh_sizes = createReversedArrayCopy(ndims, full_mesh_sizes);
    sub_mesh_starts = createReversedArrayCopy(ndims, sub_mesh_starts);
  }

  /* Summarize the size of each dimension as the product of the remaining
     dimensions. For example, with a C array foo[2][3][5]:
       size[0] == 15, size[1] == 5, size[2] == 1
  */
  full_cum_sizes = (int*) malloc(sizeof(int) * ndims);
  sub_cum_sizes  = (int*) malloc(sizeof(int) * ndims);
  assert(full_cum_sizes);
  assert(sub_cum_sizes);
  full_cum_sizes[ndims-1] = sub_cum_sizes[ndims-1] = 1;
  for (i=ndims-2; i >= 0; i--) {
    full_cum_sizes[i] = full_cum_sizes[i+1] * full_mesh_sizes[i+1];
    sub_cum_sizes[i]  = sub_cum_sizes[i+1]  * sub_mesh_sizes[i+1];
  }

  /* Figure out the dimension after which the storage is contiguous.
     For example, with full_mesh_sizes={5,5,5,5,5,5} and
     sub_mesh_sizes={4,3,2,1,5,5}
     the last two dimensions are contiguous.  first_full_dim is the
     index after which all dimensions are full so it will be 3.
 */
  for (i = ndims; i > 0; i--)
    if (sub_mesh_sizes[i-1] != full_mesh_sizes[i-1]) break;
  first_full_dim = i;

  /* use a recursive function to traverse the dimensions */
  {
    MeshTraverseParams params = {
      ndims, first_full_dim, element_size, sub_mesh_sizes, full_mesh_sizes,
      sub_mesh_starts, full_cum_sizes, sub_cum_sizes, rowFunction,
      rowFunctionParam
    };
    Mesh_IO_traverseOneDim(0, (char*)buf, &params);
  }

  /* free the copies I made of the shape arrays */
  if (array_order != MPI_ORDER_C) {
    /* strip off 'const' */
    free((int*)sub_mesh_sizes);
    free((int*)full_mesh_sizes);
    free((int*)sub_mesh_starts);
  }
  free(full_cum_sizes);
  free(sub_cum_sizes);

  /* return the nubmer of elements processed */
  return sub_cum_sizes[0] * sub_mesh_sizes[0];
}


static void rowFnEndianSwapInPlace(void *p, size_t count, void *param) {
  size_t element_size = *(size_t*)param;
  reverseBytes(p, element_size, count);
}


size_t Mesh_IO_endian_swap_in_place
(void *buf, size_t element_size, int ndims,
 const int *full_mesh_sizes, const int *sub_mesh_sizes,
 const int *sub_mesh_starts, int array_order) {

  return Mesh_IO_traverse
    (buf, element_size, ndims, full_mesh_sizes, sub_mesh_sizes,
     sub_mesh_starts, array_order, rowFnEndianSwapInPlace,
     &element_size);
}


typedef struct {
  char *write_ptr;
  size_t element_size;
} CopyToLinearParam;

static void rowFnCopyToLinear(void *p, size_t count, void *param_v) {
  CopyToLinearParam *param = (CopyToLinearParam*)param_v;
  size_t len = param->element_size * count;
  memcpy(param->write_ptr, p, len);
  param->write_ptr += len;
}


size_t Mesh_IO_copy_to_linear_array
(void *dest, const void *src, size_t element_size, int ndims,
 const int *full_mesh_sizes, const int *sub_mesh_sizes,
 const int *sub_mesh_starts, int array_order) {

  CopyToLinearParam param = {(char*)dest, element_size};
  
  return Mesh_IO_traverse
    ((void*)src, element_size, ndims, full_mesh_sizes, sub_mesh_sizes,
     sub_mesh_starts, array_order, rowFnCopyToLinear, &param);
}


static int *createReversedArrayCopy(int len, const int *array) {
  int i, *copy = (int*) malloc(sizeof(int) * len);
  assert(copy);

  for (i=0; i < len; i++)
    copy[len - 1 - i] = array[i];

  return copy;
}
