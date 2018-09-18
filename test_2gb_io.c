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
size_t size = 3000000000;

#define DEFAULT_FILENAME "test_2gb_io.tmp"


int main(int argc, char **argv) {
  int count, count_succeed, i, result = 1;
  double *array = NULL;
  const char *filename = DEFAULT_FILENAME;
  MPI_File fh;
  /* MPI_Datatype file_type, memory_type; */
  MPI_Status status;

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
  
  for (i=0; i < count; i++)
    array[i] = i;
  
  MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

  /* Try writing */
  MPI_File_write(fh, array, count, MPI_DOUBLE, &status);
  MPI_Get_count(&status, MPI_DOUBLE, &count_succeed);
  printf("Wrote %d of %d elements\n", count_succeed, count);

  /* Try reading */
  memset(array, 0, sizeof(double) * count);
  MPI_File_read_at(fh, 0, array, count, MPI_DOUBLE, &status);
  MPI_Get_count(&status, MPI_DOUBLE, &count_succeed);
  printf("Read %d of %d elements\n", count_succeed, count);
  
  for (i=0; i < count; i++) {
    if (array[i] != i) {
      printf("Read error: array[%d] should be %e, but is %e\n",
             i, (double)i, array[i]);
      break;
    }
  }

  MPI_File_close(&fh);
  
  if (rank == 0) remove(filename);

 fail:
  free(array);
  MPI_Finalize();
  return result;
}
 
