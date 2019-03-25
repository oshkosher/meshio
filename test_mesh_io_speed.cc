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
#define PRINT_COORDS 0
#define PRINT_AUTOSIZE 0

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
  vector<int> data_size, rank_size;
  int dims;
  int iters; // number of times to repeat each test
  int auto_size; // automatically set data_size and rank_size
  const char *filename;
  int stripe_len, stripe_count;

  // when sizing the data automatically, the IO should take at least
  // this many seconds
  double min_test_time;

  vector<TYPE> buf_write, buf_read;

  vector<int> dim_rank, my_offset, my_size;

  MPI_File file;
};

typedef struct {
  double min, max, avg, std_dev, median;
} Stats;

int getNodeCount();
void quickRandInit(uint64_t seed);
double quickRandDouble();
int init(Params *opt, int argc, char **argv);
int parseSize(const char *str, uint64_t *result);
int autoSize(Params *p);
void doubleSize(vector<int> &v);
int allocateBuffers(Params *p);
bool readVector(vector<int> &v, const char *s);
bool isVectorPositive(const vector<int> &v);
long vectorProd(const vector<int> &v, int offset=0, int len=INT_MAX);
string vectorToStr(const vector<int> &v);
int anyFailed(int fail);
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
    printf("\n  test_mesh_io_speed [opt] <data> <ranks> <iters> [filename]\n"
           "  test_mesh_io_speed [opt] auto <dims> <iters> [filename]\n"
           "    <data> : 1x2x3x... - size of the data\n"
           "    <ranks> : 1x2x3x... - distribution of the ranks\n"
           "    <dims> : number of dimensions when data size is automatic\n"
           "    <iters> : number of write/read cycles\n"
           "    opt:\n"
           "      -stripe_len <n>[m] : set stripe length to <n> bytes or <n>m MiB\n"
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
      uint64_t tmp_size;
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

    argno++;

    p->auto_size = 1;
    if (1 != sscanf(argv[argno], "%d", &p->dims) || p->dims < 1) {
      if (rank == 0)
        printf("Invalid dimension count: \"%s\"\n", argv[argno]);
      return 1;
    }

    argno++;

    p->data_size.clear();
    p->rank_size.clear();

  } else {

    // read data dimension string
    if (!readVector(p->data_size, argv[argno])
        || p->data_size.size() == 0){
      if (rank == 0)
        printf("Invalid data size: \"%s\"\n", argv[argno]);
      return 1;
    }
    if (!isVectorPositive(p->data_size)) {
      if (rank==0)
        printf("Data size must be positive in all dimensions.\n");
      return 1;
    }

    p->dims = p->data_size.size();
  
    argno++;

    // read rank dimension string
    if (!readVector(p->rank_size, argv[argno])
        || p->rank_size.size() == 0) {
      if (rank == 0)
        printf("Invalid rank layout: \"%s\"\n", argv[argno]);
      return 1;
    }
    if (!isVectorPositive(p->rank_size)) {
      if (rank==0)
        printf("Rank layout must be positive in all dimensions.\n");
      return 1;
    }
    if (p->rank_size.size() != p->dims) {
      if (rank==0)
        printf("Data described in %d dimensions, ranks in %d. Must match.\n",
               p->dims, (int)p->rank_size.size());
      return 1;
    }
    if (np != vectorProd(p->rank_size)) {
      if (rank==0)
        printf("Rank layout describes %d processes, but job has %d.\n",
               (int)vectorProd(p->rank_size), np);
      return 1;
    }

    argno++;

  }
  
  if (1 != sscanf(argv[argno++], "%d", &p->iters)
      || p->iters <= 0) {
    if (rank == 0)
      printf("Invalid # of iterations: \"%s\"\n", argv[argno-1]);
    return 1;
  }

  p->filename = DEFAULT_FILENAME;
  if (argno < argc) {
    p->filename = argv[argno++];
  }

  if (rank==0) {
    printf("%.6f test_mesh_io_speed %d processes on %d nodes\n",
           MPI_Wtime() - t0, np, n_nodes);
    if (p->auto_size) {
      printf("size=auto dims=%d min_test_time=%.1f sec\n",
             p->dims, p->min_test_time);
    } else {
      string data_str = vectorToStr(p->data_size),
        rank_str = vectorToStr(p->rank_size);
      printf("data=%s ranks=%s\n",
             data_str.c_str(), rank_str.c_str());
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
int parseSize(const char *str, uint64_t *result) {
  uint64_t multiplier = 1;
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
    multiplier = (uint64_t)1 << 10; break;
  case 'm':
    multiplier = (uint64_t)1 << 20; break;
  case 'g':
    multiplier = (uint64_t)1 << 30; break;
  case 't':
    multiplier = (uint64_t)1 << 40; break;
  case 'p':
    multiplier = (uint64_t)1 << 50; break;
  case 'e':
    multiplier = (uint64_t)1 << 60; break;
  default:
    return 1;  /* unrecognized suffix */
  }

  *result = (uint64_t)(multiplier * mantissa);
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

  // let MPI split the dimensions by rank
  vector<int> tmp(p->dims, 0);
  MPI_Dims_create(np, p->dims, &tmp[0]);
  p->rank_size.resize(p->dims, 0);
  for (int i=0; i < p->dims; i++)
    p->rank_size[i] = tmp[i];
  
  /*if (rank == 0) {
    string s = vectorToStr(p->rank_size);
    printf("MPI_Dims_create(np=%d, ndims=%d) yielded %s\n",
           np, p->dims, s.c_str());
           }*/

  // initialize data_size so each rank is getting a single element
  p->data_size = p->rank_size;

  // grow until the total data is at least 1 MB
  long nbytes = sizeof(TYPE) * np;
  while (nbytes < (1 << 20)) {
    doubleSize(p->data_size);
    nbytes = sizeof(TYPE) * vectorProd(p->data_size);
    if (rank==0 && PRINT_AUTOSIZE) {
      string s = vectorToStr(p->data_size);
      printf("double size to %s = %ld bytes\n", s.c_str(), nbytes);
    }
  }

  // set my_size and my_offset, allocate buf_write and buf_read
  allocateBuffers(p);

  // open the file, including setting the striping parameters
  // code is in runTests()

  vector<int> mem_start(p->dims, 0);

  while (1) {
    MPI_Barrier(MPI_COMM_WORLD);
    double timer = MPI_Wtime();
    int err;
    err = Mesh_IO_write(p->file, 0, MPITYPE, file_endian,
                        &p->buf_write[0], p->dims,
                        &p->my_size[0], &p->data_size[0],
                        &p->my_offset[0], &p->my_size[0],
                        &mem_start[0], MPI_ORDER_C, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
      if (rank == 0) {
        printf("Mesh_IO_write error %d\n", err);
      }
      return 1;
    }

    timer = MPI_Wtime() - timer;

    if (rank == 0 && PRINT_AUTOSIZE) {
      string s = vectorToStr(p->data_size);
      printf("write %s: %.3fs\n", s.c_str(), timer);
    }

    // Don't let the data grow to more than 2 GB per process because
    // mesh_io still doesn't handle that. Also, don't go past 2**30 elements
    // in any dimension because p->data_size[] is ints, and they'll overflow.
    if (timer > p->min_test_time
        || nbytes >= GB * np
        || p->data_size[0] >= GB) {
      break;
    } else {
      doubleSize(p->data_size);
      nbytes = sizeof(TYPE) * vectorProd(p->data_size);
      if (allocateBuffers(p)) return 1;
    }
  }

  return 0;
}


void doubleSize(vector<int> &data_size) {
  int min_d = 0;
  for (size_t i=1; i < data_size.size(); i++)
    if (data_size[i] < data_size[min_d])
      min_d = i;
  data_size[min_d] *= 2;
  assert(isVectorPositive(data_size));
}


int allocateBuffers(Params *p) {
  double timer;
  long i, count;

  MPI_Barrier(MPI_COMM_WORLD);
  timer = MPI_Wtime();

  /* determine the position and size of my chunk */
  p->my_offset.resize(p->dims);
  p->my_size.resize(p->dims);
  p->dim_rank.resize(p->dims);
  for (int dim=0; dim < p->dims; dim++) {
    int subdim_size = vectorProd(p->rank_size, dim+1);
    p->dim_rank[dim] = (rank / subdim_size) % p->rank_size[dim];
    
    p->my_offset[dim] = (long)p->dim_rank[dim] * p->data_size[dim]
      / p->rank_size[dim];
    int next = (long)(p->dim_rank[dim] + 1) * p->data_size[dim]
      / p->rank_size[dim];
    p->my_size[dim] = next - p->my_offset[dim];
  }

  // all processes print their coords
#if PRINT_COORDS
  string ds = vectorToStr(p->data_size);
  string rs = vectorToStr(p->rank_size);
  if (rank==0) printf("init data_size=%s rank_size=%s\n", ds.c_str(), rs.c_str());
  for (int i=0; i < np; i++) {
    if (rank == i) {
      string coord = vectorToStr(p->dim_rank),
        offset = vectorToStr(p->my_offset),
        size = vectorToStr(p->my_size);
      printf("[%d] coord %s, start %s, size %s\n", rank, coord.c_str(),
             offset.c_str(), size.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
  
  /* trim the last row and column if necessary */
  /*
  if (p->ystart + p->ysize > p->data_height)
    p->ysize = p->data_height - p->ystart;
  if (p->xstart + p->xsize > p->data_width)
    p->xsize = p->data_width - p->xstart;
  */

  /* allocate the buffer */
  count = vectorProd(p->my_size);
  // printf("[%d] Allocating buffers, count=%lld\n", rank, (long long)count);
  assert(count > 0);
  p->buf_write.resize(count, 0);
  p->buf_read.resize(count, 0);

  /* fill the write buffer with random data */
  quickRandInit((long)rank * 1234);
  for (i=0; i < count; i++)
    p->buf_write[i] = quickRandDouble();

  /* fill the read buffer with zeros */
  memset(&p->buf_read[0], 0, sizeof(TYPE) * count);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0 && PRINT_AUTOSIZE) {
    timer = MPI_Wtime() - timer;
    printf("%.6f init in %.3f sec\n", MPI_Wtime() - t0, timer);
  }
  
  return 0;
}


/* Read a vector description in the form 100[x200[x300[...]]] */
bool readVector(vector<int> &v, const char *s) {
  int pos = 0, consumed, value;
  char separator;
  v.clear();

  while (s[pos]) {
    if (1 != sscanf(s+pos, "%d%n", &value, &consumed))
      return false;

    if (value <= 0)
      return false;
    
    v.push_back(value);
    pos += consumed;

    // no more 'x'
    if (1 != sscanf(s+pos, " %c%n", &separator, &consumed))
      break;

    if (separator != 'x')
      return false;

    pos += consumed;
  }

  return true;
}


bool isVectorPositive(const vector<int> &v) {
  for (size_t i=0; i < v.size(); i++)
    if (v[i] <= 0) return false;
  return true;
}


string vectorToStr(const vector<int> &v) {
  ostringstream s;
  for (size_t i=0; i < v.size(); i++) {
    if (i > 0)
      s << 'x';
    s << v[i];
  }
  return s.str();
}


int anyFailed(int fail) {
  int any_fail;

  MPI_Allreduce(&fail, &any_fail, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  return any_fail;
}


long vectorProd(const vector<int> &v, int offset, int len) {
  len = min(len, (int)v.size()-offset);
  long s = 1;
  for (int i=offset; i < offset+len; i++)
    s *= v[i];
  return s;
}



void runTests(Params *p) {
  int err, i;
  size_t buf_size = sizeof(TYPE) * vectorProd(p->my_size);
  size_t total_size = sizeof(TYPE) * vectorProd(p->data_size);
  double total_size_gb = (double)total_size / GB;
  vector<int> mem_start(p->my_size.size(), 0);
  vector<double> write_times(p->iters), read_times(p->iters);
  double total_write = 0, total_read = 0, write_time, read_time, start_time;
  double write_gbps, read_gbps;

  if (rank == 0) {
    string s = vectorToStr(p->data_size);
    string s2 = vectorToStr(p->rank_size);
    printf("%.6f Test size %s = %ld bytes = %.3f GiB, ranks = %s\n",
           MPI_Wtime() - t0, s.c_str(), (long)total_size, total_size_gb,
           s2.c_str());
  }

  for (i = 0; i < p->iters; i++) {

    /* Write */
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    err = Mesh_IO_write(p->file, 0, MPITYPE, file_endian,
                        &p->buf_write[0], p->dims,
                        &p->my_size[0], &p->data_size[0],
                        &p->my_offset[0], &p->my_size[0],
                        &mem_start[0], MPI_ORDER_C, MPI_COMM_WORLD);
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
    err = Mesh_IO_read(p->file, 0, MPITYPE, file_endian,
                       &p->buf_read[0], p->dims,
                       &p->my_size[0], &p->data_size[0],
                       &p->my_offset[0], &p->my_size[0],
                       &mem_start[0], MPI_ORDER_C, MPI_COMM_WORLD);
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
             i, np, n_nodes, p->stripe_count, p->stripe_len, p->dims,
             total_size_gb, write_time, write_gbps, read_time, read_gbps);
	     
    }
  }

  if (rank == 0) {
    Stats s;
    string size_str = vectorToStr(p->data_size);
    /*
    printf("%d ranks, %.6f GiB data in %s (%s), %d x %d MiB stripes\n",
           np, total_size_gb, p->filename, size_str.c_str(),
           p->stripe_count, p->stripe_len);
    */
    computeSpeeds(&s, write_times, total_size_gb);
    printf("write_speed_gbps_summary np=%d nn=%d sc=%d sl=%d dims=%d "
           "data_gb=%.3g min=%.6g max=%.6g avg=%.6g median=%.6g stddev=%.6g\n",
           np, n_nodes, p->stripe_count, p->stripe_len, p->dims,
           total_size_gb, s.min, s.max, s.avg, s.median, s.std_dev);
           

           /*
           total_size / (GB * s.max),
	   total_size / (GB * s.min),
           total_size / (GB * s.avg),
           total_size / (GB * s.median));
           */

    // stddev=%.1f%%
    // 100 * s.std_dev / s.avg);

    computeSpeeds(&s, read_times, total_size_gb);
    printf("read_speed_gbps_summary np=%d nn=%d sc=%d sl=%d dims=%d "
           "data_gb=%.3g min=%.6g max=%.6g avg=%.6g median=%.6g stddev=%.6g\n",
           np, n_nodes, p->stripe_count, p->stripe_len, p->dims,
           total_size_gb, s.min, s.max, s.avg, s.median, s.std_dev);

    /*
    write_gbps = (total_size * p->iters) / (GB * total_write);
    read_gbps = (total_size * p->iters) / (GB * total_read);
    printf("Average write speed %.6f GiB/s\n", write_gbps);
    printf("Average read speed %.6f GiB/s\n", read_gbps);
    */
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
