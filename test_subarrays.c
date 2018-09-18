/*
  Test MPI-IO with large subarrays. Some versions of MPI had problems
  handling more than 2GB in IO calls.

  Ed Karrels, edk@illinois.edu, September 2018
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

#define DEFAULT_FILENAME "test_subarrays.tmp"
#define SHAPE_TALL 0
#define SHAPE_WIDE 1
#define SHAPE_GRID 2

#define MPICHECK(CALL) do {                                          \
  int result = CALL;                                                 \
  if (result != MPI_SUCCESS) {                                       \
    fprintf(stderr, "rank=%d %s:%d MPI error %d (%s)\n", rank, __FILE__, __LINE__, \
            result, getErrorName(result));                           \
    assert(result == MPI_SUCCESS);                                   \
  }                                                                  \
} while (0)

typedef double element_type;

typedef struct {
  int tile_height, tile_width;
  int64_t tile_top, tile_left;  /* global location of my tile */
  int64_t global_height, global_width;
  const char *filename;
} Options;

void printHelp();
int parseArgs(Options *opt, int argc, char **argv);
void printArgs(Options *opt);
double *createArray(Options *opt);
int createTypes(MPI_Datatype *file_type, MPI_Datatype *memory_type,
                Options *opt);
int64_t tileBytes(Options *opt);
void testWrite(MPI_File *fh, Options *opt, double *array,
               MPI_Datatype memory_type);
void testRead(MPI_File *fh, Options *opt, double *array,
              MPI_Datatype memory_type);
int anyFailed(int fail);
const char *getErrorName(int mpi_err);

int main(int argc, char **argv) {
  Options opt;
  MPI_File fh;
  int result = 1;
  double *array = NULL;
  MPI_Datatype file_type, memory_type;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (parseArgs(&opt, argc, argv))
    goto fail;
  
  printArgs(&opt);

  MPI_File_open(MPI_COMM_WORLD, opt.filename,
                MPI_MODE_RDWR | MPI_MODE_CREATE,
                MPI_INFO_NULL, &fh);

  array = createArray(&opt);
  if (anyFailed(array == NULL))
    goto fail;

  if (anyFailed(createTypes(&file_type, &memory_type, &opt)))
    goto fail;

  MPI_File_set_view(fh, 0, file_type, file_type, "native", MPI_INFO_NULL);

  testWrite(&fh, &opt, array, memory_type);
  testRead(&fh, &opt, array, memory_type);

  if (rank == 0) remove(opt.filename);
  result = 0;
  
 fail:
  free(array);
  MPI_Finalize();
  return result;
}


void printHelp() {
  if (rank == 0) {
    fprintf(stderr, "\n"
            "  test_subarrays [options]\n"
            "    Use collective IO to write and read a grid of data,\n"
            "    checking for correctness and performance. Each process will\n"
            "    write one tile of data, then read it back.\n"
            "  options:\n"
            "   -size <# of bytes> or <rows>x<cols>\n"
            "     Set the amount of data written and read by each process\n"
            "     Size may be a large (64 bit) value, but rows and cols must be < 2^31.\n"
            "     Default = 2^32 or 4 GiB\n"
            "   -file <filename>\n"
            "     Set the name of the test file.\n"
            "     Default = test_subarrays.tmp\n"
            "   -shape: tall, wide, or grid.\n"
            "     tall: tiles are stacked vertically, so each process will\n"
            "       access one contiguous strip of data.\n"
            "     wide: tiles are stacked horizontally, so each row will contain\n"
            "       data from every process.\n"
            "     grid: tiles are arranged in a 2d grid\n"
            "     Default = wide.\n"
            "\n");
  }
  MPI_Finalize();
  exit(1);
}
  
  
int parseArgs(Options *opt, int argc, char **argv) {
  int argno;
  int shape = SHAPE_WIDE;
  int64_t size_bytes = (int64_t)4 * 1024 * 1024 * 1024;
  opt->tile_height = opt->tile_width = -1;
  opt->tile_top = opt->tile_left = -1;
  opt->filename = DEFAULT_FILENAME;
  
  for (argno=1; argno < argc; argno++) {
    const char *arg = argv[argno];

    if (!strcmp(arg, "-h")) {
      printHelp();
    }

    else if (!strcmp(arg, "-size")) {
      if (argc - argno < 1) printHelp();
      arg = argv[++argno];
      const char *x = strchr(arg, 'x');
      if (x) {
        if (2 != sscanf(arg, "%dx%d", &opt->tile_height, &opt->tile_width)) {
          printHelp();
        }
        if (opt->tile_height < 1 || opt->tile_width < 1) {
          if (rank == 0) {
            fprintf(stderr, "Invalid tile size: %s\n", arg);
          }
          return 1;
        }
        size_bytes = -1;
      } else {
        if (1 != sscanf(arg, "%" SCNi64, &size_bytes)) {
          printHelp();
        }
        if (size_bytes < 1) {
          if (rank == 0) {
            fprintf(stderr, "Invalid byte count: %s\n", arg);
          }
          return 1;
        }
      }
    }
    
    else if (!strcmp(arg, "-file")) {
      if (argc - argno < 1) printHelp();
      opt->filename = argv[++argno];
    }

    else if (!strcmp(arg, "-shape")) {
      if (argc - argno < 1) printHelp();
      arg = argv[++argno];
      
      if (!strcmp(arg, "tall")) {
        shape = SHAPE_TALL;
      } else if (!strcmp(arg, "wide")) {
        shape = SHAPE_WIDE;
      } else if (!strcmp(arg, "grid")) {
        shape = SHAPE_GRID;
      } else {
        printHelp();
      }
    }

    else {
      printHelp();
    }
  }
  
  /* if the size is specified in bytes, split that into square tiles of
     approximately the right size */
  if (size_bytes > 0) {
    int64_t elem_count = (size_bytes + sizeof(element_type) - 1)
      / sizeof(element_type);
    opt->tile_height = sqrt(elem_count);
    opt->tile_width = elem_count / opt->tile_height;
  }
  assert(opt->tile_height > 0 && opt->tile_width > 0);

  /* compute the total size and my position in it */
  switch (shape) {
  case SHAPE_WIDE:
    opt->global_height = opt->tile_height;
    opt->global_width = opt->tile_width * np;
    opt->tile_top = 0;
    opt->tile_left = opt->tile_width * rank;
    break;

  case SHAPE_TALL:
    opt->global_height = opt->tile_height * np;
    opt->global_width = opt->tile_width;
    opt->tile_top = opt->tile_height * rank;
    opt->tile_left = 0;
    break;
    
  case SHAPE_GRID:
    {
      int tiles_horiz, tiles_vert;  /* # of tiles in grid */
      int tile_x, tile_y;
      tiles_horiz = ceil(sqrt(np));
      tiles_vert = (np + tiles_horiz - 1) / tiles_horiz;
      tile_y = rank / tiles_horiz;
      tile_x = rank % tiles_horiz;
      opt->tile_top = tile_y * opt->tile_height;
      opt->tile_left = tile_x * opt->tile_width;
      opt->global_height = tiles_vert * opt->tile_height;
      opt->global_width = tiles_horiz * opt->tile_width;
    }
  }
  
  return 0;
}


void printArgs(Options *opt) {
  int i;

  if (rank == 0) {
    int64_t bytes = tileBytes(opt);
    printf("tiles %dx%d, %ld bytes, %.3f GiB\n",
           opt->tile_height, opt->tile_width,
           (long)bytes, bytes / (1024. * 1024 * 1024));
    printf("overall %ldx%ld\n", 
           (long)opt->global_height, (long)opt->global_width);
  }

  for (i=0; i < np; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (i == rank) {
      printf("%d: tile at %ld,%ld\n", rank,
             (long)opt->tile_top, (long)opt->tile_left);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}


double *createArray(Options *opt) {
  size_t array_size = tileBytes(opt);
  double *array = (double*) malloc(array_size);

  if (!array) {
    printf("%d: Failed to allocate array of %" PRIi64 " bytes\n",
           rank, (int64_t)array_size);
  }
  return array;
}


int createTypes(MPI_Datatype *file_type, MPI_Datatype *memory_type,
                Options *opt) {
  int mesh_sizes[2] = {opt->tile_height, opt->tile_width};
  int file_mesh_sizes[2] = {opt->global_height, opt->global_width};
  int file_mesh_starts[2] = {opt->tile_top, opt->tile_left};
  int memory_mesh_sizes[2] = {opt->tile_height, opt->tile_width};
  int memory_mesh_starts[2] = {0, 0};

  /*
printf("%d: MPI_Type_create_subarray(2, [%d,%d], [%d,%d], [%d,%d]...\n",
         rank, file_mesh_sizes[0], file_mesh_sizes[1],
         mesh_sizes[0], mesh_sizes[1],
         file_mesh_starts[0], file_mesh_starts[1]);
  */
  MPI_Type_create_subarray(2, file_mesh_sizes, mesh_sizes,
                           file_mesh_starts,
                           MPI_ORDER_C, MPI_DOUBLE, file_type);
  MPI_Type_commit(file_type);
  
  MPI_Type_create_subarray(2, memory_mesh_sizes, mesh_sizes,
                           memory_mesh_starts,
                           MPI_ORDER_C, MPI_DOUBLE, memory_type);
  
  MPI_Type_commit(memory_type);

  return 0;
}


/* returns the number of bytes in a tile */
int64_t tileBytes(Options *opt) {
  return sizeof(element_type) * opt->tile_height * opt->tile_width;
}



void testWrite(MPI_File *fh, Options *opt, double *array,
               MPI_Datatype memory_type) {
  int row, col;
  double *p = array, start_time, elapsed;
  MPI_Status status;
  double mbps;
  
  for (row=0; row < opt->tile_height; row++) {
    for (col=0; col < opt->tile_width; col++) {
      double value = (opt->tile_top + row) * 1000000 +
        (opt->tile_left + col);
      *p++ = value;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  start_time = MPI_Wtime();
  MPI_File_write_at_all(*fh, 0, array, 1, memory_type, &status);
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed = MPI_Wtime() - start_time;
  
  mbps = np * tileBytes(opt) / (elapsed * 1024*1024);
  if (rank == 0) printf("write time: %.3fs, %.3f MiB/s\n", elapsed, mbps);
}


void testRead(MPI_File *fh, Options *opt, double *array,
              MPI_Datatype memory_type) {
  int row, col;
  double *p = array, start_time, elapsed;
  MPI_Status status;
  double mbps;

  MPI_Barrier(MPI_COMM_WORLD);
  start_time = MPI_Wtime();
  MPI_File_read_at_all(*fh, 0, array, 1, memory_type, &status);
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed = MPI_Wtime() - start_time;

  mbps = np * tileBytes(opt) / (elapsed * 1024*1024);
  if (rank == 0) printf("read time: %.3fs, %.3f MiB/s\n", elapsed, mbps);
  
  for (row=0; row < opt->tile_height; row++) {
    for (col=0; col < opt->tile_width; col++) {
      double expected = (opt->tile_top + row) * 1000000 +
        (opt->tile_left + col);

      if (*p != expected) {
        printf("%d: error at [%d,%d]: expected %e, got %e\n", rank, row, col,
               expected, *p);
        return;
      }
      p++;
    }
  }

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
  case MPI_ERR_FILE: return "MPI_ERR_FILE";
  default: return NULL;
  }
}
