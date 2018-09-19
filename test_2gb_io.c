/*
  Test MPI-IO with reads & writes over 2GB.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <inttypes.h>
#include "mpi.h"

int rank = -1, np = -1;
size_t size = (size_t)3000 * 1024 * 1024;

#define DEFAULT_FILENAME "test_2gb_io.tmp"

int checkArray(const double *array, int count);


int main(int argc, char **argv) {
  int count, count_succeed, i, result = 1;
  double *array = NULL;
  const char *filename = DEFAULT_FILENAME;
  MPI_File fh;
  MPI_Datatype array_type;
  MPI_Status status;
  int starts[2] = {0,0}, sizes[2];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc > 1) {
    if (argc > 2 || argv[1][0] == '-') {
      printf("\n  test_2gb_io [temp filename]\n\n");
      return 0;
    }
    filename = argv[1];
  }

  count = size / sizeof(double);
  
  array = (double*) malloc(sizeof(double) * count);
  if (!array) {
    printf("Failed to allocate %ld bytes for array\n", 
           (long)(sizeof(double) * count));
    goto fail;
  }
  
  MPI_File_open(MPI_COMM_WORLD, filename,
                MPI_MODE_RDWR | MPI_MODE_CREATE,
                MPI_INFO_NULL, &fh);
  
  /* initialize array */
  for (i=0; i < count; i++)
    array[i] = i;
  
  MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

  /* write 1-D array */
  MPI_File_write_at(fh, 0, array, count, MPI_DOUBLE, &status);
  MPI_Get_count(&status, MPI_DOUBLE, &count_succeed);
  printf("Wrote %d of %d elements\n", count_succeed, count);

  /* read 1-D array*/
  memset(array, 0, sizeof(double) * count);
  MPI_File_read_at(fh, 0, array, count, MPI_DOUBLE, &status);
  MPI_Get_count(&status, MPI_DOUBLE, &count_succeed);
  printf("Read %d of %d elements\n", count_succeed, count);
  
  checkArray(array, count);

  /* write 2-D array */
  sizes[0] = sqrt(count);
  sizes[1] = count / sizes[0];
  MPI_Type_create_subarray(2, sizes, sizes, starts,
                           MPI_ORDER_C, MPI_DOUBLE, &array_type);
  MPI_Type_commit(&array_type);

  MPI_File_write_at(fh, 0, array, 1, array_type, &status);
  MPI_Get_count(&status, array_type, &count_succeed);
  printf("Wrote %d of 1 %dx%d array\n", count_succeed, sizes[0], sizes[1]);

  /* read 2-D array */
  memset(array, 0, sizeof(double) * count);
  MPI_File_read_at(fh, 0, array, 1, array_type, &status);
  MPI_Get_count(&status, array_type, &count_succeed);
  printf("Read %d of 1 %dx%d array\n", count_succeed, sizes[0], sizes[1]);
  checkArray(array, sizes[0] * sizes[1]);
  
  MPI_File_close(&fh);
  
  if (rank == 0) remove(filename);
  result = 0;

 fail:
  free(array);
  MPI_Finalize();
  return result;
}
 

int checkArray(const double *array, int count) {
  int i;
  for (i=0; i < count; i++) {
    if (array[i] != i) {
      printf("Read error: array[%d] should be %e, but is %e\n",
             i, (double)i, array[i]);
      return 1;
    }
  }
  return 0;
}
  
