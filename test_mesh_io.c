#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include "reverse_bytes.h"
#include "mesh_io.h"
#include "mpi.h"

int rank, np;

#define FILENAME "test_mesh_io.tmp"
#define KB ((uint64_t)1024)
#define MB ((uint64_t)1024*1024)
#define GB ((uint64_t)1024*1024*1024)
int FILE_HEADER_LEN = 16;
int simple_file_size[3] = {10, 10, 10};


int isBigEndian();
int anyFailed(int fail);
void createFile();
void testReadSimple(MPI_File fh);
void testReadWithHalo(MPI_File fh);
void testCopyToLinear();
void testWriteSimple(MPI_File fh);
void testWriteOffset(MPI_File fh);
void testWritePartial(MPI_File fh);
void testWriteEndian(MPI_File fh);
void testLargeAccess(MPI_File fh);
const char *getErrorName(int mpi_err);
void reportError(const char *msg, int mpi_err);

int main(int argc, char **argv) {
  MPI_File fh;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) createFile();

  if (np < 2) {
    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    testReadSimple(fh);
    testReadWithHalo(fh);
    testCopyToLinear();
    MPI_File_close(&fh);
  }

  MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  if (np < 2) {
    testWriteSimple(fh);
    testWriteOffset(fh);
    testWritePartial(fh);
    testWriteEndian(fh);
  }
  testLargeAccess(fh);
  MPI_File_close(&fh);

  if (rank == 0) remove(FILENAME);
  MPI_Finalize();
  return 0;
}


int isBigEndian() {
  int tmp = 1;
  return ! *(char*) &tmp;
}


int anyFailed(int fail) {
  int any_fail;

  MPI_Allreduce(&fail, &any_fail, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  return any_fail;
}


const char *getErrorName(int mpi_err) {
  switch (mpi_err) {
  case MPI_SUCCESS: return "MPI_SUCCESS";
  case MPI_ERR_TYPE: return "MPI_ERR_TYPE";
  case MPI_ERR_BUFFER: return "MPI_ERR_BUFFER";
  case MPI_ERR_DIMS: return "MPI_ERR_DIMS";
  case MPI_ERR_ARG: return "MPI_ERR_ARG";
  case MPI_ERR_TRUNCATE: return "MPI_ERR_TRUNCATE";
  default: return NULL;
  }
}


void reportError(const char *msg, int mpi_err) {
  const char *description = getErrorName(mpi_err);
  if (description) {
    printf("[%d] %s: %s\n", rank, msg, description);
  } else {
    printf("[%d] %s: %d\n", rank, msg, mpi_err);
  }
}


/* Create a simple data file with a 16 byte header and a 10x10x10 cube
   of doubles who values are z*10000 + y*100 + x. */
void createFile() {
  double array[10][10][10];
  int i, j, k;
  FILE *outf;

  for (i=0; i < 10; i++)
    for (j=0; j < 10; j++)
      for (k=0; k < 10; k++)
        array[i][j][k] = i*10000 + j * 100 + k;

  outf = fopen(FILENAME, "wb");

  /* write 16 character header (FILE_HEADER_LEN) */
  fprintf(outf, "mesh test header");
  fwrite(array, sizeof(double), 10*10*10, outf);
  fclose(outf);
}


/* Have each process the first 5 elements of a row, and each process
   is reading a different row. */
void testReadSimple(MPI_File fh) {
  int layer = (rank/10) % 10;
  int row = rank % 10;
  int read_size[3] = {1, 1, 5}, read_starts[3] = {layer, row, 0};
  int write_starts[3] = {0, 0, 0};
  int err, i;
  double data[5] = {-1, -1, -1, -1, -1};
  
  err = Mesh_IO_read(fh, FILE_HEADER_LEN, MPI_DOUBLE, MESH_IO_IGNORE_ENDIAN,
                     data, 3, read_size,
                     simple_file_size, read_starts, MPI_ORDER_C,
                     read_size, write_starts, MPI_ORDER_C);

  if (err != MPI_SUCCESS) {
    reportError("Failed with error code", err);
    return;
  }

  /* printf("[%02d] %.0f %.0f %.0f %.0f %.0f\n",
     rank, data[0], data[1], data[2], data[3], data[4]); */

  err = 0;
  for (i=0; i < 5; i++) {
    double expected = layer*10000 + row*100 + i;
    if (data[i] != expected) {
      printf("[%d] error [%d] is %f, should be %f\n", rank, i, data[i],
             expected);
      err = 1;
    }
  }

  MPI_Reduce(&err, &i, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("testReadSimple: %s\n", i==0 ? "ok" : "FAIL");
  }
}


/* Read 3 elements into an array of 5: . X X X . */
void testReadWithHalo(MPI_File fh) {
  int layer = (rank/10) % 10;
  int row = rank % 10;
  int read_size[3] = {1, 1, 3}, read_starts[3] = {layer, row, 1};
  int write_buffer_size[3] = {1, 1, 5}, write_starts[3] = {0, 0, 1};
  int err, i;
  double data[5] = {-1, -1, -1, -1, -1};

  err = Mesh_IO_read(fh, FILE_HEADER_LEN, MPI_DOUBLE, MESH_IO_IGNORE_ENDIAN,
                     data, 3, read_size,
                     simple_file_size, read_starts, MPI_ORDER_C,
                     write_buffer_size, write_starts, MPI_ORDER_C);

  if (err != MPI_SUCCESS) {
    reportError("Failed with error code", err);
    err = 1;
    goto fail;
  }

  /* printf("[%02d] %.0f %.0f %.0f %.0f %.0f\n",
     rank, data[0], data[1], data[2], data[3], data[4]); */

  err = 0;
  if (data[0] != -1 || data[4] != -1) {
    printf("[%d] error halo cells modified\n", rank);
    err = 1;
  }
  
  for (i=1; i < 4; i++) {
    double expected = layer*10000 + row*100 + i;
    if (data[i] != expected) {
      printf("[%d] error [%d] is %f, should be %f\n", rank, i, data[i],
             expected);
      err = 1;
    }
  }

 fail:
  MPI_Reduce(&err, &i, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("testReadWithHalo: %s\n", i==0 ? "ok" : "FAIL");
  }

}


void testCopyToLinear() {
  int source[5][7], dest[12] = {0};
  int full_mesh_sizes[2] = {5, 7};
  int sub_mesh_sizes[2] = {4, 3};
  int sub_mesh_starts[2] = {1, 3};
  int i, j;

  if (rank != 0) return;

  for (j=0; j < 5; j++)
    for (i=0; i < 7; i++)
      source[j][i] = j*10 + i;
  
  /*for (i=0; i < 12; i++)
    printf("%2d ", dest[i]);
    printf("\n");*/

  /*
  for (j=0; j < 5; j++) {
    for (i=0; i < 7; i++) {
      printf("%2d ", source[j][i]);
    }
    printf("\n");
  }
  printf("\n");
  */

  Mesh_IO_copy_to_linear_array(dest, source, sizeof(int), 2, full_mesh_sizes,
                               sub_mesh_sizes, sub_mesh_starts, MPI_ORDER_C);

  /*
  for (j=0; j < 4; j++) {
    for (i=0; i < 3; i++) {
      printf("%2d ", dest[j*3 + i]);
    }
    printf("\n");
  }
  printf("\n");
  */

  for (j=0; j < 4; j++)
    for (i=0; i < 3; i++)
      assert(dest[j*3 + i] == (j+1)*10 + (i+3));


  printf("testCopyToLinear ok\n");
}


void testWriteSimple(MPI_File fh) {
  int array[4], array0[4] = {0, 0, 0, 0}, array1[4] = {1000, 2000, 3000, 4000};
  int err, start=0, length=4;
  FILE *f;
  size_t read_len;

  /* fill the file with zeros */
  f = fopen(FILENAME, "wb");
  fwrite(array0, sizeof(int), 4, f);
  fclose(f);

  /* write array1 */
  err = Mesh_IO_write(fh, 0, MPI_INT, MESH_IO_IGNORE_ENDIAN, array1, 1, 
                      &length, &length, &start, MPI_ORDER_C,
                      &length, &start, MPI_ORDER_C);
  assert(err == MPI_SUCCESS);

  /* check the file contents */
  f = fopen(FILENAME, "rb");
  memset(array, 0xff, sizeof(array));
  read_len = fread(array, sizeof(int), 4, f);
  assert(read_len == 4);
  fclose(f);
  assert(0 == memcmp(array, array1, sizeof(array)));

  printf("testWriteSimple ok\n");
}


void testWriteOffset(MPI_File fh) {
  int array[4], array0[4] = {0, 0, 0, 0}, array1[4] = {1000, 2000, 3000, 4000};
  int err, start=0, length=4;
  char chars[3], chars1[3] = {12, 13, 14};
  FILE *f;
  size_t read_len;

  /* fill the file with a header and zeros */
  f = fopen(FILENAME, "wb");
  fwrite(chars1, 1, 3, f);
  fwrite(array0, sizeof(int), 4, f);
  fclose(f);

  /* write array1 */
  err = Mesh_IO_write(fh, 3, MPI_INT, MESH_IO_IGNORE_ENDIAN, array1, 1, 
                      &length, &length, &start, MPI_ORDER_C,
                      &length, &start, MPI_ORDER_C);
  assert(err == MPI_SUCCESS);

  /* check the file contents */
  f = fopen(FILENAME, "rb");
  memset(chars, 0xff, sizeof(chars));
  memset(array, 0xff, sizeof(array));
  read_len = fread(chars, 1, 3, f);
  assert(read_len == 3);
  read_len = fread(array, sizeof(int), 4, f);
  assert(read_len == 4);
  fclose(f);
  assert(0 == memcmp(chars, chars1, sizeof(chars)));
  assert(0 == memcmp(array, array1, sizeof(array)));

  printf("testWriteOffset ok\n");
}


void testWritePartial(MPI_File fh) {
  double sub_mesh[5][5][5];
  double full_mesh[10][10][10];
  int mesh_sizes[3] = {3, 3, 3};
  int memory_mesh_sizes[3] = {5, 5, 5};
  int memory_mesh_starts[3] = {0, 0, 0};
  int file_mesh_sizes[3] = {10, 10, 10};
  int file_mesh_starts[3] = {2, 3, 5};
  int i, j, k, err;
  FILE *inf;
  size_t read_len;

  if (rank != 0) return;
  
  /* reset the disk file */
  createFile();

  /* test contents before */
  inf = fopen(FILENAME, "rb");
  fseek(inf, FILE_HEADER_LEN, SEEK_SET);
  read_len = fread(full_mesh, sizeof(double), 10*10*10, inf);
  assert(read_len == 10*10*10);
  fclose(inf);
  for (i=0; i < 10; i++)
    for (j=0; j < 10; j++)
      for (k=0; k < 10; k++)
        assert(full_mesh[i][j][k] == i*10000 + j * 100 + k);

  /* fill data array, doubling the values of interest */
  for (i=0; i < 5; i++)
    for (j=0; j < 5; j++)
      for (k=0; k < 5; k++)
        sub_mesh[i][j][k] = 2 * (i*10000 + j * 100 + k);

  err = Mesh_IO_write
    (fh, FILE_HEADER_LEN, MPI_DOUBLE, MESH_IO_IGNORE_ENDIAN,
     sub_mesh, 3,
     mesh_sizes, file_mesh_sizes, file_mesh_starts, MPI_ORDER_C,
     memory_mesh_sizes, memory_mesh_starts, MPI_ORDER_C);
  if (err != MPI_SUCCESS) {
    reportError("Failed with error code", err);
    return;
  }

  /* test contents after */
  inf = fopen(FILENAME, "rb");
  fseek(inf, FILE_HEADER_LEN, SEEK_SET);
  read_len = fread(full_mesh, sizeof(double), 10*10*10, inf);
  assert(read_len == 10*10*10);
  fclose(inf);
  for (i=0; i < 10; i++)
    for (j=0; j < 10; j++)
      for (k=0; k < 10; k++) {

        if (i >= file_mesh_starts[0]
            && i < file_mesh_starts[0] + mesh_sizes[0]
            && j >= file_mesh_starts[1]
            && j < file_mesh_starts[1] + mesh_sizes[1]
            && k >= file_mesh_starts[2]
            && k < file_mesh_starts[2] + mesh_sizes[2]) {
          assert(full_mesh[i][j][k] ==
                 2 * ((i-file_mesh_starts[0])*10000
                      + (j-file_mesh_starts[1]) * 100
                      + (k-file_mesh_starts[2])));
        } else {
          assert(full_mesh[i][j][k] == i*10000 + j * 100 + k);
        }

        /*
        if (full_mesh[i][j][k] != i*10000 + j * 100 + k) {
          printf("[%d][%d][%d] expected %d, got %.0f\n", i, j, k,
                 i*10000 + j * 100 + k, full_mesh[i][j][k]);
        }
        */
      }          

  printf("test_mesh_io ok\n");
}
  

/* write bit-reversed [. 1 2 3 4 . 6 7 8 9] */
void testWriteEndian(MPI_File fh) {
  double mesh[10];
  int mesh_sizes[2] = {2, 4};
  int memory_mesh_sizes[2] = {2, 5};
  int memory_mesh_starts[2] = {0, 1};
  int file_mesh_sizes[2] = {100, 5};
  int file_mesh_starts[2] = {0, 1};
  int i, err;
  FILE *inf;
  size_t read_len;


  /* fill the first row of the file with 12345.0 */
  inf = fopen(FILENAME, "wb");
  for (i=0; i < 10; i++) mesh[i] = 12345;
  fseek(inf, FILE_HEADER_LEN, SEEK_SET);
  fwrite(mesh, sizeof(double), 10, inf);
  fclose(inf);

  /* write 8 out of 10 elements of that row, and write them with their
     bytes reversed */
  for (i=0; i < 10; i++) mesh[i] = i;

  err = Mesh_IO_write
    (fh, FILE_HEADER_LEN, MPI_DOUBLE, MESH_IO_SWAP_ENDIAN,
     mesh, 2,
     mesh_sizes, file_mesh_sizes, file_mesh_starts, MPI_ORDER_C,
     memory_mesh_sizes, memory_mesh_starts, MPI_ORDER_C);
  if (err != MPI_SUCCESS) {
    reportError("Failed with error code", err);
    return;
  }

  /* test the results */
  inf = fopen(FILENAME, "rb");
  for (i=0; i < 10; i++) mesh[i] = 12345;
  fseek(inf, FILE_HEADER_LEN, SEEK_SET);
  read_len = fread(mesh, sizeof(double), 10, inf);
  assert(read_len == 10);
  fclose(inf);

  assert(mesh[0] == 12345.0);
  assert(mesh[5] == 12345.0);

  reverseBytes(&mesh[1], sizeof(double), 4);
  reverseBytes(&mesh[6], sizeof(double), 4);
  for (i=1; i < 5; i++) {
    assert(mesh[i] == i);
    assert(mesh[i+5] == i+5);
  }
  printf("testWriteEndian ok\n");
}


/* Test reads and writes of 1GB and 4GB */
void testLargeAccess(MPI_File fh) {
  int *buf;
  int fail = 0, result;
  int cols, rows;
  int64_t bytes_per_process;
  int mesh_sizes[2], file_mesh_sizes[2], file_mesh_starts[2];
  int memory_mesh_sizes[2], memory_mesh_starts[2] = {0, 0};

  /* write two rows of data, so it will need to be reordered
     2.001 works (2 * 268569673 = 2GiB + 1073736)
     but 2.01 doesnt (2 * 269777633 = 2GiB + 10737416)
  */
  bytes_per_process = 2.01 * GB;
  rows = 2;
  cols = bytes_per_process / (rows * sizeof(int));
  bytes_per_process = rows * cols * sizeof(int);
  
  if (rank==0)
    printf("Writing %d x %d array of ints: %" PRIu64 " bytes per process\n",
           rows, cols, bytes_per_process);

  /* allocate buffer */
  buf = (int*) calloc(1, bytes_per_process);
  if (!buf) {
    fprintf(stderr, "%d: Failed to allocate %" PRIu64 
            " bytes for testLargeAccess\n",
            rank, bytes_per_process);
    fail = 1;
  }
  
  if (anyFailed(fail)) return;

  mesh_sizes[0] = rows;
  mesh_sizes[1] = cols;

  file_mesh_sizes[0] = rows;
  file_mesh_sizes[1] = cols * np;

  file_mesh_starts[0] = 0;
  file_mesh_starts[1] = cols * rank;
  
  memory_mesh_sizes[0] = mesh_sizes[0];
  memory_mesh_sizes[1] = mesh_sizes[1];
  
  result = Mesh_IO_write(fh, 0, MPI_INT, MESH_IO_IGNORE_ENDIAN, buf,
                         2, mesh_sizes,
                         file_mesh_sizes, file_mesh_starts, MPI_ORDER_C, 
                         memory_mesh_sizes, memory_mesh_starts, MPI_ORDER_C);
  
  if (result != MPI_SUCCESS) {
    fprintf(stderr, "%d: Error writing %" PRIu64 " bytes: %s\n",
            rank, bytes_per_process, getErrorName(result));
  } else {
    fprintf(stderr, "%d: Successfully wrote %" PRIu64 " bytes\n",
            rank, bytes_per_process);
  }

  free(buf);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("testLargeAccess\n");
  }
}
