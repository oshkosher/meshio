/*
  Test the performance of 1-dimensional collective IO.
  mesh_io isn't good at this because the full file size will tend
  to be larger than 2^31 elements even when every process writes less
  than that. Thus the file_mesh_sizes and file_mesh_starts arguments
  would need to be made into longs, which will likely cause a cascade of
  type changes, running into the brick wall that most MPI-IO implementations
  don't handle operations of more than 2GiB.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include "mesh_io.h"

#define GB ((int64_t)1024*1024*1024)

#ifdef __cplusplus
extern "C" {
#endif

int Mesh_IO_read_1d
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 void *buf,
 MPI_Count mesh_size,
 MPI_Count file_mesh_size,
 MPI_Count file_mesh_start,
 MPI_Count memory_mesh_size,
 MPI_Count memory_mesh_start,
 MPI_Comm comm);

int Mesh_IO_write_1d
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 void *buf,
 MPI_Count mesh_size,
 MPI_Count file_mesh_size,
 MPI_Count file_mesh_start,
 MPI_Count memory_mesh_size,
 MPI_Count memory_mesh_start,
 MPI_Comm comm);

static int readWriteInit1d
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 long mesh_size,
 long file_mesh_size,
 long file_mesh_start,
 long memory_mesh_size,
 long memory_mesh_start,
 size_t *element_size,
 MPI_Offset *sub_offset);

#ifdef __cplusplus
}
#endif


int Mesh_IO_read_1d
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 void *buf,
 MPI_Count mesh_size,
 MPI_Count file_mesh_size,
 MPI_Count file_mesh_start,
 MPI_Count memory_mesh_size,
 MPI_Count memory_mesh_start,
 MPI_Comm comm) {

  int i, err = MPI_SUCCESS, rank;
  size_t element_size;
  MPI_Offset sub_offset;  /* my data will be at offset + sub_offset */
  MPI_Status status;
  
  MPI_Comm_rank(comm, &rank);

  if (file_endian != MESH_IO_IGNORE_ENDIAN) {
    fprintf(stderr, "Only MESH_IO_IGNORE_ENDIAN is currently supported "
            "in Mesh_IO_read_1d.\n");
    return MPI_ERR_ARG;
  }

  /* check for argument errors and initialize datatypes */
  err = readWriteInit1d(fh, offset, etype, file_endian, mesh_size,
                        file_mesh_size, file_mesh_start, 
                        memory_mesh_size, memory_mesh_start,
                        &element_size, &sub_offset);
  if (err != MPI_SUCCESS) return err;

  buf = (void*)((char*)buf + memory_mesh_start * element_size);
  
  /* read the mesh */
  err = MPI_File_read_at_all(fh, file_mesh_start, buf, mesh_size, etype,
                             &status);
  if (err != MPI_SUCCESS) return err;

  /* check if data was read */
  MPI_Get_count(&status, etype, &i);
  if (i != mesh_size) return MPI_ERR_TRUNCATE;

  /* Fix the endianness of the data */
  /*
  if (isEndianSwapNeeded(file_endian)) {
    Mesh_IO_endian_swap_in_place
      (buf, element_size, ndims,
       memory_mesh_sizes, mesh_sizes,
       memory_mesh_starts, order);
  }
  */

  return err;
};


int Mesh_IO_write_1d
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 void *buf,
 MPI_Count mesh_size,
 MPI_Count file_mesh_size,
 MPI_Count file_mesh_start,
 MPI_Count memory_mesh_size,
 MPI_Count memory_mesh_start,
 MPI_Comm comm) {

  int i, err = MPI_SUCCESS, rank;
  size_t element_size;
  MPI_Offset sub_offset;  /* my data will be at offset + sub_offset */
  MPI_Status status;
  
  MPI_Comm_rank(comm, &rank);

  if (file_endian != MESH_IO_IGNORE_ENDIAN) {
    fprintf(stderr, "Only MESH_IO_IGNORE_ENDIAN is currently supported "
            "in Mesh_IO_write_1d.\n");
    return MPI_ERR_ARG;
  }

  /* check for argument errors and initialize datatypes */
  err = readWriteInit1d(fh, offset, etype, file_endian, mesh_size,
                        file_mesh_size, file_mesh_start, 
                        memory_mesh_size, memory_mesh_start,
                        &element_size, &sub_offset);
  if (err != MPI_SUCCESS) return err;

  buf = (void*)((char*)buf + memory_mesh_start * element_size);
  
  /* read the mesh */
  err = MPI_File_write_at_all(fh, file_mesh_start, buf, mesh_size, etype,
                              &status);
  if (err != MPI_SUCCESS) return err;

  /* check if data was read */
  MPI_Get_count(&status, etype, &i);
  if (i != mesh_size) return MPI_ERR_TRUNCATE;

  return err;
};


static int readWriteInit1d
(MPI_File fh,
 MPI_Offset offset,
 MPI_Datatype etype,
 int file_endian,
 long mesh_size,
 long file_mesh_size,
 long file_mesh_start,
 long memory_mesh_size,
 long memory_mesh_start,
 size_t *element_size,
 MPI_Offset *sub_offset) {
  
  int i, err = MPI_SUCCESS;
  
  /* check for argument errors */
  if (file_endian < MESH_IO_LITTLE_ENDIAN ||
      file_endian > MESH_IO_SWAP_ENDIAN)
    return MPI_ERR_ARG;

  /* if endianness is to be considered, check for a basic datatype */
  /*
  if (file_endian != MESH_IO_IGNORE_ENDIAN
      && !isSupportedType(etype)) return MPI_ERR_TYPE;
  */

  /* check for mesh boundary errors */
  if (file_mesh_start < 0
      || file_mesh_start + mesh_size > file_mesh_size
      || memory_mesh_start < 0
      || memory_mesh_start + mesh_size > memory_mesh_size
      ) {
    return MPI_ERR_ARG;
  }

  /* get the size of each element */
  MPI_Type_size(etype, &i);
  *element_size = i;

  /* complain if it's nonpositive */
  if (*element_size < 1)
    return MPI_ERR_ARG;

  /* warn about 2GB ops */
  if ((int64_t)*element_size * mesh_size >= 2*GB) {
    fprintf(stderr, "Warning: many MPI-IO implementation cannot handle IO "
            "operations of 2GiB or more per process.\n");
  }

  /* if endian swapping is to be done, complain if the size is larger
     than 1 and odd */
  if (file_endian != MESH_IO_IGNORE_ENDIAN
      && *element_size > 1
      && (*element_size & 1))
    return MPI_ERR_ARG;    
  
  /* set the file view */
  err = MPI_File_set_view
    (fh, offset, etype, etype, "native", MPI_INFO_NULL);
  if (err != MPI_SUCCESS) return err;

  /* compute the offset from 'offset' from which this process's data
     will begin */
  *sub_offset = (MPI_Offset)((int64_t)*element_size * file_mesh_start);

  return MPI_SUCCESS;
}  



#if 0 /* simple test */
int np, rank;
#define FILENAME "test_coll_1d.tmp"

int main(int argc, char **argv) {
  MPI_File fh;
  int i, err;
  int64_t size_bytes;
  double *data_write = NULL;
  double *data_read = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  /* rank R writes R+1 elements, each consecutively */
  MPI_Count mesh_size = rank+1,
    file_mesh_size = (np+1)*np/2, 
    file_mesh_start = (rank+1)*rank/2;

  printf("[%d] mesh_size=%d, offset=%d, full size=%d\n",
         rank, (int)mesh_size, (int)file_mesh_start,
         (int)file_mesh_size);
  
  size_bytes = sizeof(double) * mesh_size;
  data_write = (double*) malloc(size_bytes);
  data_read = (double*) malloc(size_bytes);
  for (i=0; i < mesh_size; i++) {
    data_write[i] = 1000000 + rank + i*1e-5;
  }
  memset(data_read, 0, size_bytes);
  
  if (rank==0) {
    printf("opening %s\n", FILENAME);
    remove(FILENAME);
  }
  
  MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_RDWR | MPI_MODE_CREATE,
                MPI_INFO_NULL, &fh);

  if (rank==0) printf("Writing...\n");
  err = Mesh_IO_write_1d(fh, 16, MPI_DOUBLE, MESH_IO_IGNORE_ENDIAN,
                         data_write, mesh_size, file_mesh_size,
                         file_mesh_start, mesh_size, 0, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS) {
    printf("[%d] Error writing: %d\n", rank, err);
    return 1;
  }

  if (rank==0) printf("Reading...\n");
  err = Mesh_IO_read_1d(fh, 16, MPI_DOUBLE, MESH_IO_IGNORE_ENDIAN,
                         data_read, mesh_size, file_mesh_size,
                         file_mesh_start, mesh_size, 0, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS) {
    printf("[%d] Error read: %d\n", rank, err);
    return 1;
  }

  if (memcmp(data_write, data_read, size_bytes)) {
    printf("[%d] read error\n", rank);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank==0) printf("done.\n");

  free(data_write);
  free(data_read);
  MPI_File_close(&fh);
  MPI_Finalize();
  return 0;
}
#endif /* simple test */

  

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <mpi.h>
#include "mesh_io.h"

#define MB ((int64_t)1024*1024)
#define GB ((int64_t)1024*1024*1024)
#define DEFAULT_FILENAME "test_mesh_io_speed.tmp"
typedef double TYPE;
#define MPITYPE MPI_DOUBLE
#define DEFAULT_MIN_TEST_TIME 10.0
#define PRINT_COORDS 1
#define PRINT_AUTOSIZE 1

#ifdef _CRAYC
#define PRIu64 "llu"
#define PRIi64 "lld"
#define SCNu64 "llu"
#define SCNi64 "lld"
#endif

int file_endian = MESH_IO_IGNORE_ENDIAN;
/* int file_endian = MESH_IO_SWAP_ENDIAN; */

using std::vector;
using std::string;
using std::ostringstream;
using std::min;
using std::sort;


int rank, np, n_nodes;
double t0;  // start time

struct Params {
  int64_t data_size;  // total number of elements
  int iters; // number of times to repeat each test
  int auto_size; // automatically set data_size and rank_size
  const char *filename;
  int stripe_len, stripe_count;

  // when sizing the data automatically, the IO should take at least
  // this many seconds
  double min_test_time;

  vector<TYPE> buf_write, buf_read;

  int64_t my_offset, my_size;

  MPI_File file;
};

typedef struct {
  double min, max, avg, std_dev, median;
} Stats;

int getNodeCount();
void quickRandInit(uint64_t seed);
double quickRandDouble();
int init(Params *opt, int argc, char **argv);
int parseSize(const char *str, int64_t *result);
int autoSize(Params *p);
int allocateBuffers(Params *p);
bool readVector(vector<int> &v, const char *s);
bool isVectorPositive(const vector<int> &v);
long vectorProd(const vector<int> &v, int offset=0, int len=INT_MAX);
string vectorToStr(const vector<int> &v);
int openFile(Params *opt);
void runTests(Params *opt);
void computeSpeeds(Stats *s, const vector<double> &times, double data_size_gb);
const char *formatLen(char buf[10], int len);


int main(int argc, char **argv) {
  Params opt;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  t0 = MPI_Wtime();
  n_nodes = getNodeCount();

  if (init(&opt, argc, argv)) goto fail;

  if (opt.auto_size) {
    if (autoSize(&opt)) goto fail;
  } else {
    allocateBuffers(&opt);
  }
  
  runTests(&opt);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0)
    printf("%.6f done\n\n", MPI_Wtime() - t0);

 fail:
  MPI_Finalize();
  return 0;
}


void printHelp() {
  if (rank == 0)
    printf("\n  test_coll_1d [opt] <data>|auto <iters> [filename]\n"
           "    <data> : total size of the data in bytes (suffixes k, m, g recognized)\n"
           "    <iters> : number of write/read cycles\n"
           "    opt:\n"
           "      -stripe_len <n> : set stripe length to <n> bytes (suffixes OK)\n"
           "      -stripe_count <n> : set stripe count\n"
           "      -min_time <sec> : minimum IO time when auto-sizing. default=%.1f\n"
           "  If <data> is 'auto', the size of the data is chosen automatically.\n"
           "  Starting with a small data set, the size is doubled until writing the data\n"
           "  takes at least <min_time> seconds.\n"
           "\n",
           DEFAULT_MIN_TEST_TIME);
  MPI_Finalize();
  exit(1);
}


// borrowed from https://stackoverflow.com/questions/34115227/how-to-get-the-number-of-physical-machine-in-mpi
int getNodeCount() {
  int rank, is_rank0, nodes;
  MPI_Comm shmcomm;

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(shmcomm, &rank);
  is_rank0 = (rank == 0) ? 1 : 0;
  MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Comm_free(&shmcomm);
  return nodes;
}


static uint64_t quick_rand_seed;
void quickRandInit(uint64_t seed) {
  quick_rand_seed = seed;
}


double quickRandDouble() {
  quick_rand_seed = 2862933555777941757ULL * quick_rand_seed + 3037000493;
  return quick_rand_seed * 1e-10;
}


int init(Params *p, int argc, char **argv) {
  int argno;

  if (argc < 4) printHelp();

  p->auto_size = 0;
  p->stripe_len = 4 * MB;
  p->stripe_count = 1;
  p->min_test_time = DEFAULT_MIN_TEST_TIME;

  for (argno = 1; argno < argc && argv[argno][0] == '-'; argno++) {

    if (!strcmp(argv[argno], "-stripe_len")) {
      argno++;
      if (argno >= argc) printHelp();
      int64_t tmp_size;
      if (parseSize(argv[argno], &tmp_size)) {
        if (rank==0) printf("Invalid stripe length: %s\n", argv[argno]);
        return 1;
      }

      if (tmp_size > GB) {
        if (rank==0)
          printf("Stripe length too large: %llu\n",
                 (long long unsigned)tmp_size);
        return 1;
      }
      
      p->stripe_len = tmp_size;
    }

    else if (!strcmp(argv[argno], "-stripe_count")) {
      argno++;
      if (argno >= argc) printHelp();
      if (1 != sscanf(argv[argno], "%d", &p->stripe_count)
          || p->stripe_count < 1) {
        if (rank==0) printf("Invalid stripe count: %s\n", argv[argno]);
        return 1;
      }
    }

    else if (!strcmp(argv[argno], "-min_time")) {
      argno++;
      if (argno >= argc) printHelp();
      if (1 != sscanf(argv[argno], "%lf", &p->min_test_time)
          || p->min_test_time < 0) {
        if (rank==0) printf("Invalid minimum test time: \"%s\"\n", argv[argno]);
        return 1;
      }
    }

    else {
      if (rank==0) printf("Invalid argument: %s\n", argv[argno]);
    }
  }

  if (argc - argno < 3 || argc - argno > 4) printHelp();

  if (!strcmp(argv[argno], "auto")) {
    p->auto_size = 1;
    p->data_size = 0;
  } else {
    // read data size in bytes, convert to # of elements
    if (parseSize(argv[argno], &p->data_size) ||
        p->data_size == 0) {
      if (rank == 0)
        printf("Invalid data size: \"%s\"\n", argv[argno]);
      return 1;
    }
    p->data_size /= sizeof(TYPE);
  }
  argno++;
  
  if (1 != sscanf(argv[argno], "%d", &p->iters)
      || p->iters <= 0) {
    if (rank == 0)
      printf("Invalid # of iterations: \"%s\"\n", argv[argno]);
    return 1;
  }
  argno++;

  p->filename = DEFAULT_FILENAME;
  if (argno < argc) {
    p->filename = argv[argno++];
  }

  if (rank==0) {
    printf("%.6f test_coll_1d %d processes on %d nodes\n",
           MPI_Wtime() - t0, np, n_nodes);
    if (p->auto_size) {
      printf("size=auto min_test_time=%.1f sec\n",
             p->min_test_time);
    } else {
      printf("total_data[%" PRIi64 "]\n", p->data_size);
    }
    char len_str[10];
    printf("iters=%d, file=%s, %d x %s stripes\n",
           p->iters, p->filename, p->stripe_count,
           formatLen(len_str, p->stripe_len));
  }
  
  openFile(p);

  return 0;
}


/* Parse an unsigned number with a case-insensitive magnitude suffix:
     k : multiply by 2^10
     m : multiply by 2^20
     g : multiply by 2^30
     t : multiply by 2^40
     p : multiply by 2^50
     e : multiply by 2^60

   For example, "32m" would parse as 33554432.
   Floating point numbers are allowed in the input, but the result is always
   a 64-bit integer:  ".5g" yields uint64_t(.5 * 2^30)
   Return nonzero on error.
*/
int parseSize(const char *str, int64_t *result) {
  int64_t multiplier = 1;
  const char *last;
  char suffix;
  int consumed;
  double mantissa;
  
  /* missing argument check */
  if (!str || !str[0] || (!isdigit(str[0]) && str[0] != '.')) return 1;

  if (1 != sscanf(str, "%lf%n", &mantissa, &consumed))
    return 1;
  last = str + consumed;

  /* these casts are to avoid issues with signed chars */
  suffix = (char)tolower((int)*last);
  switch(suffix) {
  case 0:
    multiplier = 1; break;
  case 'k':
    multiplier = (int64_t)1 << 10; break;
  case 'm':
    multiplier = (int64_t)1 << 20; break;
  case 'g':
    multiplier = (int64_t)1 << 30; break;
  case 't':
    multiplier = (int64_t)1 << 40; break;
  case 'p':
    multiplier = (int64_t)1 << 50; break;
  case 'e':
    multiplier = (int64_t)1 << 60; break;
  default:
    return 1;  /* unrecognized suffix */
  }

  *result = (int64_t)(multiplier * mantissa);
  return 0;
}


int openFile(Params *p) {
  MPI_Info info;
  char int_str[50];

  MPI_Info_create(&info);
  sprintf(int_str, "%d", p->stripe_count);
  MPI_Info_set(info, "striping_factor", int_str);
  sprintf(int_str, "%d", p->stripe_len);
  MPI_Info_set(info, "striping_unit", int_str);
  
  MPI_Barrier(MPI_COMM_WORLD);
  double timer = MPI_Wtime();
  MPI_File_open(MPI_COMM_WORLD, p->filename, MPI_MODE_RDWR | MPI_MODE_CREATE,
                info, &p->file);
  MPI_Barrier(MPI_COMM_WORLD);
  timer = MPI_Wtime() - timer;

  if (rank==0)
    printf("MPI_File_open np=%d nn=%d sec=%.3f\n", np, n_nodes, timer);

  MPI_Info_free(&info);

  return 0;
}


int autoSize(Params *p) {

  if (rank == 0 && PRINT_AUTOSIZE) {
    char len_str[10];
    printf("%.6f auto-sizing data for np=%d, nn=%d, stripe_count=%d, "
           "stripe_len=%s\n", MPI_Wtime() - t0, np, n_nodes,
           p->stripe_count, formatLen(len_str, p->stripe_len));
  }

  // initialize data_size so each rank is getting a single element
  p->data_size = np;

  // grow until the total data is at least 1 MB
  uint64_t nbytes = sizeof(TYPE) * p->data_size;
  while (nbytes < (1 << 20)) {
    p->data_size *= 2;
    nbytes = sizeof(TYPE) * p->data_size;
    if (rank==0 && PRINT_AUTOSIZE) {
      printf("double size to %" PRIi64 " = %" PRIi64 " bytes\n",
             p->data_size, nbytes);
    }
  }

  // set my_size and my_offset, allocate buf_write and buf_read
  allocateBuffers(p);

  // open the file, including setting the striping parameters
  // code is in runTests()

  while (1) {
    MPI_Barrier(MPI_COMM_WORLD);
    double timer = MPI_Wtime();
    int err;
    err = Mesh_IO_write_1d
      (p->file, 0, MPITYPE, file_endian, &p->buf_write[0], 
       p->my_size, p->data_size, p->my_offset, p->my_size, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
      if (rank == 0) {
        printf("Mesh_IO_write_1d error %d\n", err);
      }
      return 1;
    }

    timer = MPI_Wtime() - timer;

    if (rank == 0 && PRINT_AUTOSIZE)
      printf("write %" PRIi64 ": %.3fs\n", p->data_size, timer);

    // Don't let the data grow to more than 2 GB per process because
    // mesh_io still doesn't handle that.
    if (timer > p->min_test_time || nbytes >= GB * np) {
      break;
    } else {
      p->data_size *= 2;
      nbytes = sizeof(TYPE) * p->data_size;
      if (allocateBuffers(p)) return 1;
    }
  }

  return 0;
}


int allocateBuffers(Params *p) {
  double timer;
  int64_t i;

  MPI_Barrier(MPI_COMM_WORLD);
  timer = MPI_Wtime();

  /* determine the position and size of my chunk */
  p->my_offset = rank * p->data_size / np;
  int64_t next_offset = (rank+1) * p->data_size / np;
  p->my_size = next_offset - p->my_offset;

  // all processes print their coords
#if PRINT_COORDS
  if (rank==0) printf("init data_size=%" PRIi64 "\n", p->data_size);
  for (int i=0; i < np; i++) {
    if (rank == i) {
      printf("[%d] start %" PRIi64 ", size %" PRIi64 "\n", rank, 
             p->my_offset, p->my_size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  /* allocate the buffer */
  assert(p->my_size >= 0 && p->my_size < 2*GB);
  p->buf_write.resize(p->my_size, 0);
  p->buf_read.resize(p->my_size, 0);

  /* fill the write buffer with random data */
  quickRandInit((long)rank * 1234);
  for (i=0; i < p->my_size; i++)
    p->buf_write[i] = quickRandDouble();

  /* fill the read buffer with zeros */
  memset(&p->buf_read[0], 0, sizeof(TYPE) * p->my_size);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0 && PRINT_AUTOSIZE) {
    timer = MPI_Wtime() - timer;
    printf("%.6f init in %.3f sec\n", MPI_Wtime() - t0, timer);
  }
  
  return 0;
}



void runTests(Params *p) {
  int err, i;
  int64_t buf_size = sizeof(TYPE) * p->my_size;
  int64_t total_size = sizeof(TYPE) * p->data_size;
  double total_size_gb = (double)total_size / GB;
  vector<double> write_times(p->iters), read_times(p->iters);
  double total_write = 0, total_read = 0, write_time, read_time, start_time;
  double write_gbps, read_gbps;

  if (rank == 0) {
    printf("%.6f Test size %" PRIi64 " = %" PRIi64 " bytes = %.3f GiB"
           ", ranks = %d\n",
           MPI_Wtime() - t0, p->data_size, total_size, total_size_gb, np);
  }

  for (i = 0; i < p->iters; i++) {

    /* Write */
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    err = Mesh_IO_write_1d
      (p->file, 0, MPITYPE, file_endian, &p->buf_write[0], 
       p->my_size, p->data_size, p->my_offset, p->my_size, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    write_time = MPI_Wtime() - start_time;
    total_write += write_time;
    write_times[i] = write_time;

    if (err != MPI_SUCCESS)
      printf("%d: Mesh_IO_write error %d\n", rank, err);

    /* clear the read buffer */
    memset(&p->buf_read[0], 0, buf_size);

    /* Read */
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    err = Mesh_IO_read_1d
      (p->file, 0, MPITYPE, file_endian, &p->buf_read[0], 
       p->my_size, p->data_size, p->my_offset, p->my_size, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    read_time = MPI_Wtime() - start_time;
    total_read += read_time;
    read_times[i] = read_time;
  
    if (err != MPI_SUCCESS)
      printf("%d: Mesh_IO_read error %d\n", rank, err);

    if (memcmp(&p->buf_read[0], &p->buf_write[0], buf_size))
      printf("%d: data mismatch\n", rank);

    if (rank == 0) {
      write_gbps = total_size / (GB * write_time);
      read_gbps = total_size / (GB * read_time);
      printf("sample iter=%d np=%d nn=%d sc=%d sl=%d dims=%d data_gb=%.3g "
             "write_sec=%.6g write_gbps=%.6g read_sec=%.6g read_gbps=%.6g\n",
             i, np, n_nodes, p->stripe_count, p->stripe_len, 1,
             total_size_gb, write_time, write_gbps, read_time, read_gbps);
	     
    }
  }

  if (rank == 0) {
    Stats s;
    /*
    printf("%d ranks, %.6f GiB data in %s (%s), %d x %d MiB stripes\n",
           np, total_size_gb, p->filename, size_str.c_str(),
           p->stripe_count, p->stripe_len);
    */
    computeSpeeds(&s, write_times, total_size_gb);
    printf("write_speed_gbps_summary np=%d nn=%d sc=%d sl=%d dims=%d "
           "data_gb=%.3g min=%.6g max=%.6g avg=%.6g median=%.6g stddev=%.6g\n",
           np, n_nodes, p->stripe_count, p->stripe_len, 1,
           total_size_gb, s.min, s.max, s.avg, s.median, s.std_dev);

    computeSpeeds(&s, read_times, total_size_gb);
    printf("read_speed_gbps_summary np=%d nn=%d sc=%d sl=%d dims=%d "
           "data_gb=%.3g min=%.6g max=%.6g avg=%.6g median=%.6g stddev=%.6g\n",
           np, n_nodes, p->stripe_count, p->stripe_len, 1,
           total_size_gb, s.min, s.max, s.avg, s.median, s.std_dev);
  }

  MPI_File_close(&p->file);
  remove(p->filename);
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


/* Compute simple statistics on an array of doubles.
   Trim 20% of the data off either end. */
void computeSpeeds(Stats *s, const vector<double> &times, double data_size_gb) {
  double sum = 0, sum2 = 0;
  int count = times.size();
  vector<double> speeds(count);

  if (count == 0) {
    s->min = s->max = s->avg = s->std_dev = s->median = 0;
    return;
  }

  for (int i=0; i < count; i++)
    speeds[i] = data_size_gb / times[i];

  sort(speeds.begin(), speeds.end());

  // don't trim
  /*
  if (count >= 5) {
    int trim = count / 5;
    data.erase(data.begin(), data.begin()+trim);
    data.erase(data.end()-trim, data.end());
    count = data.size();
  }
  */

  s->min = speeds[0];
  s->max = speeds[count-1];
  
  for (size_t i=0; i < count; i++) {
    /* printf("%.8f\n", data[i]); */
    sum += speeds[i];
    sum2 += speeds[i] * speeds[i];
  }

  s->avg = s->std_dev = 0;
  
  s->avg = sum / count;
  if (count > 1) {
    s->std_dev = sqrt( (count * sum2 - sum*sum) / (count * (count-1)) );
  }
  
  if (count & 1) {
    s->median = speeds[count/2];
  } else {
    s->median = 0.5 * (speeds[(count-1)/2] + speeds[(count+1)/2]);
  }
}


const char *formatLen(char buf[10], int len) {
  if (len <= 0) {
    sprintf(buf, "%d", len);
  } else if ((len % MB) == 0) {
    sprintf(buf, "%dm", (int)(len / MB));
  } else if ((len % 1024) == 0) {
    sprintf(buf, "%dk", len / 1024);
  } else {
    sprintf(buf, "%d", len);
  }
  return buf;
}
