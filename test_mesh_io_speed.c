/*
  Test the throughput of Mesh_IO_read() and Mesh_IO_write() on two-dimensional
  data. 

  Arguments:
    - Size of the data (height and width). The data will be 8-byte doubles.
      For example, a 1000x1000 grid would be 8,000,000 bytes.
    - Number of ranks in each direction. The product must be equal to the total
      number of ranks, but it does not need to evenly divide the data.
    - Number of write/read iterations.

  For example, here is how it can be use to test a few different data
  access patterns:
  
  ranks = 3x1
     +-----------------------------+
     | rank 0                      |
     |                             |
     +-----------------------------+
     | rank 1                      |
     |                             |
     +-----------------------------+
     | rank 2                      |
     |                             |
     +-----------------------------+

  ranks = 1x3
     +--------+--------+--------+
     | rank 0 | rank 1 | rank 2 |
     |        |        |        |
     |        |        |        |
     |        |        |        |
     |        |        |        |
     |        |        |        |
     |        |        |        |
     |        |        |        |
     +--------+--------+--------+

  ranks = 3x3
     +--------+--------+--------+
     | rank 0 | rank 1 | rank 2 |
     |        |        |        |
     +--------+--------+--------+
     | rank 3 | rank 4 | rank 5 |
     |        |        |        |
     +--------+--------+--------+
     | rank 6 | rank 7 | rank 8 |
     |        |        |        |
     +--------+--------+--------+

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "mesh_io.h"

#define GB (1024*1024*1024)
#define DEFAULT_FILENAME "test_mesh_io_speed.tmp"
typedef double TYPE;
#define MPITYPE MPI_DOUBLE

int file_endian = MESH_IO_IGNORE_ENDIAN;
/* int file_endian = MESH_IO_SWAP_ENDIAN; */

int rank, np;

typedef struct {
  int data_height, data_width;
  int ranks_height, ranks_width;
  int iters;
  const char *filename;
  int stripe_len, stripe_count;

  TYPE *buf_write, *buf_read;
  int x, y;            /* index of my chunk */
  int ystart, xstart;  /* position of my chunk */
  int ysize, xsize;    /* size of my chunk */
} Params;

typedef struct {
  double min, max, avg, std_dev, median;
} Stats;

int init(Params *opt, int argc, char **argv);
void runTests(Params *opt);
void computeStats(Stats *s, double *data, size_t count);


int main(int argc, char **argv) {
  Params opt;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (!init(&opt, argc, argv)) goto fail;
  
  runTests(&opt);

 fail:
  MPI_Finalize();
  return 0;
}


void printHelp() {
  if (rank == 0)
    printf("\n  test_mesh_io_speed [opt] <data> <ranks> <iters> [filename]\n"
           "    <data> : <height>x<width> - size of the data\n"
           "    <ranks> : <height>x<width> - distribution of the ranks\n"
           "    <iters> : number of write/read cycles\n"
           "    opt:\n"
           "      -stripe_len <n> : set stripe length to <n> MiB\n"
           "      -stripe_count <n> : set stripe count\n"
           "\n");
  MPI_Finalize();
  exit(1);
}


int init(Params *p, int argc, char **argv) {
  long i, count, argno;
  double start_time = MPI_Wtime();
  int success = 1;

  if (argc < 4) printHelp();

  p->stripe_len = 4;
  p->stripe_count = 1;

  for (argno = 1; argno < argc && argv[argno][0] == '-'; argno++) {

    if (!strcmp(argv[argno], "-stripe_len")) {
      argno++;
      if (argno >= argc) printHelp();
      if (1 != sscanf(argv[argno], "%d", &p->stripe_len)
          || p->stripe_len < 1) {
        if (rank==0) printf("Invalid stripe length: %s\n", argv[argno]);
        return 0;
      }
    }

    else if (!strcmp(argv[argno], "-stripe_count")) {
      argno++;
      if (argno >= argc) printHelp();
      if (1 != sscanf(argv[argno], "%d", &p->stripe_count)
          || p->stripe_count < 1) {
        if (rank==0) printf("Invalid stripe count: %s\n", argv[argno]);
        return 0;
      }
    }

    else {
      if (rank==0) printf("Invalid argument: %s\n", argv[argno]);
    }
  }

  if (argc - argno < 3 || argc - argno > 4) printHelp();
    
  if (2 != sscanf(argv[argno++], "%dx%d", &p->data_height, &p->data_width)
      || p->data_height <= 0
      || p->data_width <= 0) {
    if (rank == 0)
      printf("Invalid data size (must be in the form 000x000, and both must "
             "be positive: \"%s\"\n", argv[argno-1]);
    return 0;
  }
    
  if (2 != sscanf(argv[argno++], "%dx%d", &p->ranks_height, &p->ranks_width)
      || p->ranks_height <= 0
      || p->ranks_width <= 0) {
    if (rank == 0)
      printf("Invalid rank split (must be in the form 000x000, and both must "
             "be positive: \"%s\"\n", argv[argno-1]);
    return 0;
  }

  if (1 != sscanf(argv[argno++], "%d", &p->iters)
      || p->iters <= 0) {
    if (rank == 0)
      printf("Invalid # of iterations: \"%s\"\n", argv[argno-1]);
    return 0;
  }

  if (p->ranks_height * p->ranks_width != np) {
    if (rank == 0)
      printf("Rank split %dx%d does not match rank count %d\n",
             p->ranks_height, p->ranks_width, np);
    return 0;
  }

  p->filename = DEFAULT_FILENAME;
  if (argno < argc) {
    p->filename = argv[argno++];
  }

  if (rank==0)
    printf("test_mesh_io_speed data %dx%d, ranks %dx%d iters %d, file %s\n",
           p->data_height, p->data_width, p->ranks_height, p->ranks_width,
           p->iters, p->filename);

  /* determine the position and size of my chunk */
  p->y = rank / p->ranks_width;
  p->x = rank % p->ranks_width;
  
  p->ysize = (p->data_height + p->ranks_height - 1) / p->ranks_height;
  p->xsize = (p->data_width + p->ranks_width - 1) / p->ranks_width;

  p->ystart = p->y * p->ysize;
  p->xstart = p->x * p->xsize;

  /* trim the last row and column if necessary */
  if (p->ystart + p->ysize > p->data_height)
    p->ysize = p->data_height - p->ystart;
  if (p->xstart + p->xsize > p->data_width)
    p->xsize = p->data_width - p->xstart;

  /*
  printf("%d: %lu bytes (%d,%d) at (%d,%d) size (%d,%d)\n",
         rank, (long unsigned)(sizeof(TYPE) * p->ysize * p->xsize),
         p->y, p->x, p->ystart, p->xstart,
         p->ysize, p->xsize);
  */

  /* allocate the buffer */
  count = (long)p->ysize * p->xsize;
  p->buf_write = (TYPE*) malloc(sizeof(TYPE) * count);
  p->buf_read  = (TYPE*) malloc(sizeof(TYPE) * count);
  if (!p->buf_write || !p->buf_read) {
    printf("%d: failed to allocate 2 * %lu bytes\n", rank,
           (long unsigned)(sizeof(TYPE) * p->ysize * p->xsize));
    success = 0;
  }

  /* make sure everyone succeeded */
  MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!success) {
    if (rank == 0) printf("Some ranks failed to allocate memory; exiting.\n");
    return 0;
  }

  /* fill the write buffer with random data */
  srand(rank * 1234);
  for (i=0; i < count; i++)
    p->buf_write[i] = rand() + rand() * 1000000000.0;

  /* fill the read buffer with zeros */
  memset(p->buf_read, 0, sizeof(TYPE) * count);

  if (rank == 0)
    printf("Init finished in %.3f sec\n", MPI_Wtime() - start_time);
  
  return 1;
}


void runTests(Params *p) {
  MPI_File f;
  int err, i;
  size_t buf_size = sizeof(TYPE) * p->ysize * p->xsize;
  size_t total_size = sizeof(TYPE) * p->data_height * p->data_width;
  int file_size[2] = {p->data_height, p->data_width};
  int mesh_size[2] = {p->ysize, p->xsize};
  int file_start[2] = {p->ystart, p->xstart};
  int mem_start[2] = {0, 0};
  double *write_times, *read_times;
  double total_write = 0, total_read = 0, write_time, read_time, start_time;
  double write_gbps, read_gbps, open_time;
  MPI_Info info;
  char int_str[50];
  
  write_times = (double*) malloc(sizeof(double) * p->iters);
  read_times  = (double*) malloc(sizeof(double) * p->iters);

  MPI_Info_create(&info);
  sprintf(int_str, "%d", p->stripe_count);
  MPI_Info_set(info, "striping_factor", int_str);
  sprintf(int_str, "%d", p->stripe_len * 1024 * 1024);
  MPI_Info_set(info, "striping_unit", int_str);
  
  start_time = MPI_Wtime();
  MPI_File_open(MPI_COMM_WORLD, p->filename, MPI_MODE_RDWR | MPI_MODE_CREATE,
                info, &f);
  open_time = MPI_Wtime() - start_time;
  printf("%d: MPI_File_open %.3f sec\n", rank, open_time);

  MPI_Info_free(&info);

  for (i = 0; i < p->iters; i++) {

    /* Write */
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    err = Mesh_IO_write(f, 0, MPITYPE, file_endian, p->buf_write, 2,
                        mesh_size, file_size, file_start, MPI_ORDER_C,
                        mesh_size, mem_start, MPI_ORDER_C);
    MPI_Barrier(MPI_COMM_WORLD);
    write_time = MPI_Wtime() - start_time;
    total_write += write_time;
    write_times[i] = write_time;

    if (err != MPI_SUCCESS)
      printf("%d: Mesh_IO_write error %d\n", rank, err);

    /* clear the read buffer */
    memset(p->buf_read, 0, buf_size);

    /* Read */
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    err = Mesh_IO_read(f, 0, MPITYPE, file_endian, p->buf_read, 2,
                       mesh_size, file_size, file_start, MPI_ORDER_C,
                       mesh_size, mem_start, MPI_ORDER_C);
    MPI_Barrier(MPI_COMM_WORLD);
    read_time = MPI_Wtime() - start_time;
    total_read += read_time;
    read_times[i] = read_time;
  
    if (err != MPI_SUCCESS)
      printf("%d: Mesh_IO_read error %d\n", rank, err);

    if (memcmp(p->buf_read, p->buf_write, buf_size))
      printf("%d: data mismatch\n", rank);

    if (rank == 0) {
      write_gbps = total_size / (GB * write_time);
      read_gbps = total_size / (GB * read_time);
      printf("iter %d write %.3fs %.3f GiB/s read %.3fs %.3f GiB/s\n",
	     i, write_time, write_gbps, read_time, read_gbps);
    }
  }

  if (rank == 0) {
    Stats s;
    printf("%d ranks, %.6f GiB data in %s (%d, %d, %ld), %d x %d MiB stripes\n",
           np, (double)total_size / GB,
           p->filename, p->data_height, p->data_width, (long)total_size,
           p->stripe_count, p->stripe_len);
    computeStats(&s, write_times, p->iters);
    printf("write speed GiB/s %.6f .. %.6f, avg %.3f, median %.3f, "
           "stddev %.1f%%\n",
           total_size / (GB * s.max),
	   total_size / (GB * s.min),
           total_size / (GB * s.avg),
           total_size / (GB * s.median), s.std_dev / s.avg * 100);

    computeStats(&s, read_times, p->iters);
    printf("read speed GiB/s %.6f .. %.6f, avg %.3f, median %.3f, "
           "stddev %.1f%%\n",
           total_size / (GB * s.max),
	   total_size / (GB * s.min),
           total_size / (GB * s.avg),
           total_size / (GB * s.median), s.std_dev / s.avg * 100);

    /*
    write_gbps = (total_size * p->iters) / (GB * total_write);
    read_gbps = (total_size * p->iters) / (GB * total_read);
    printf("Average write speed %.6f GiB/s\n", write_gbps);
    printf("Average read speed %.6f GiB/s\n", read_gbps);
    */
  }

  MPI_File_close(&f);
  remove(p->filename);
  free(write_times);
  free(read_times);
}


int compareDoubles(const void *av, const void *bv) {
  double a = * (double*) av;
  double b = * (double*) bv;
  if (a < b)
    return -1;
  if (a > b)
    return 1;
  return 0;
}


/* Compute simple statistice on an array of doubles.
   Trim 20% of the data off either end. */
void computeStats(Stats *s, double *data, size_t count) {
  int i;
  double sum = 0, sum2 = 0;

  if (count == 0) {
    s->min = s->max = s->avg = s->std_dev = s->median = 0;
    return;
  }

  qsort(data, count, sizeof(double), compareDoubles);

  if (count >= 5) {
    int trim = count / 5;
    data += trim;
    count -= 2*trim;
  }

  s->min = s->max = data[0];
  
  for (i=0; i < count; i++) {
    /* printf("%.8f\n", data[i]); */
    sum += data[i];
    sum2 += data[i] * data[i];
    if (data[i] < s->min) s->min = data[i];
    if (data[i] > s->max) s->max = data[i];
  }

  s->avg = sum / count;
  s->std_dev = sqrt( (count * sum2 - sum*sum) / (count * (count-1)) );
  
  if (count & 1) {
    s->median = data[count/2];
  } else {
    s->median = 0.5 * (data[(count-1)/2] + data[(count+1)/2]);
  }
}

