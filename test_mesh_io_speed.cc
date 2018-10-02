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
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <mpi.h>
#include "mesh_io.h"

#define GB (1024*1024*1024)
#define DEFAULT_FILENAME "test_mesh_io_speed.tmp"
typedef double TYPE;
#define MPITYPE MPI_DOUBLE

int file_endian = MESH_IO_IGNORE_ENDIAN;
/* int file_endian = MESH_IO_SWAP_ENDIAN; */

using std::vector;
using std::string;
using std::ostringstream;
using std::min;
using std::sort;


int rank, np;

typedef struct {
  vector<int> data_size, rank_size;
  int dims;
  int iters;
  const char *filename;
  int stripe_len, stripe_count;

  TYPE *buf_write, *buf_read;
  int x, y;            /* index of my chunk */
  vector<int> dim_rank, my_offset, my_size;
} Params;

typedef struct {
  double min, max, avg, std_dev, median;
} Stats;

int init(Params *opt, int argc, char **argv);
bool readVector(vector<int> &v, const char *s);
bool isVectorPositive(const vector<int> &v);
long vectorProd(const vector<int> &v, int offset=0, int len=INT_MAX);
string vectorToStr(const vector<int> &v);
int anyFailed(int fail);
void runTests(Params *opt);
void computeStats(Stats *s, const vector<double> &data);


int main(int argc, char **argv) {
  Params opt;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (init(&opt, argc, argv)) goto fail;
  
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
  int fail = 0;

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
    string data_str = vectorToStr(p->data_size),
      rank_str = vectorToStr(p->rank_size);
    printf("test_mesh_io_speed data %s, ranks %s, iters %d, file %s\n",
           data_str.c_str(), rank_str.c_str(), p->iters, p->filename);
  }

  /* determine the position and size of my chunk */
  p->my_offset.resize(p->dims);
  p->my_size.resize(p->dims);
  p->dim_rank.resize(p->dims);
  for (size_t dim=0; dim < p->dims; dim++) {
    int subdim_size = vectorProd(p->rank_size, dim+1);
    p->dim_rank[dim] = (rank / subdim_size) % p->rank_size[dim];
    
    p->my_offset[dim] = (long)p->dim_rank[dim] * p->data_size[dim]
      / p->rank_size[dim];
    int next = (long)(p->dim_rank[dim] + 1) * p->data_size[dim]
      / p->rank_size[dim];
    p->my_size[dim] = next - p->my_offset[dim];
  }

  // all processes print their coords
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
  
  /* trim the last row and column if necessary */
  /*
  if (p->ystart + p->ysize > p->data_height)
    p->ysize = p->data_height - p->ystart;
  if (p->xstart + p->xsize > p->data_width)
    p->xsize = p->data_width - p->xstart;
  */

  /*
  printf("%d: %lu bytes (%d,%d) at (%d,%d) size (%d,%d)\n",
         rank, (long unsigned)(sizeof(TYPE) * p->ysize * p->xsize),
         p->y, p->x, p->ystart, p->xstart,
         p->ysize, p->xsize);
  */

  /* allocate the buffer */
  count = (long)vectorProd(p->my_size);
  p->buf_write = new TYPE[count];
  p->buf_read  = new TYPE[count];
  if (!p->buf_write || !p->buf_read) {
    printf("%d: failed to allocate %ld bytes\n", rank,
           (long)(sizeof(TYPE) * count));
    fail = 1;
  }

  /* make sure everyone succeeded */
  if (anyFailed(fail)) {
    if (rank == 0) printf("Some ranks failed to allocate memory; exiting.\n");
    return 1;
  }

  /* fill the write buffer with random data */
  srand(rank * 1234);
  for (i=0; i < count; i++)
    p->buf_write[i] = rand() + rand() * 1000000000.0;

  /* fill the read buffer with zeros */
  memset(p->buf_read, 0, sizeof(TYPE) * count);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    printf("Init finished in %.3f sec\n", MPI_Wtime() - start_time);
  
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
  MPI_File f;
  int err, i;
  size_t buf_size = sizeof(TYPE) * vectorProd(p->my_size);
  size_t total_size = sizeof(TYPE) * vectorProd(p->data_size);
  vector<int> mem_start(p->my_size.size(), 0);
  vector<double> write_times(p->iters), read_times(p->iters);
  double total_write = 0, total_read = 0, write_time, read_time, start_time;
  double write_gbps, read_gbps, open_time;
  MPI_Info info;
  char int_str[50];

  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Info_create(&info);
  sprintf(int_str, "%d", p->stripe_count);
  MPI_Info_set(info, "striping_factor", int_str);
  sprintf(int_str, "%d", p->stripe_len * 1024 * 1024);
  MPI_Info_set(info, "striping_unit", int_str);
  
  start_time = MPI_Wtime();
  MPI_File_open(MPI_COMM_WORLD, p->filename, MPI_MODE_RDWR | MPI_MODE_CREATE,
                info, &f);
  open_time = MPI_Wtime() - start_time;
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0)
    printf("MPI_File_open %.3f sec\n", open_time);

  MPI_Info_free(&info);

  for (i = 0; i < p->iters; i++) {

    /* Write */
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    err = Mesh_IO_write(f, 0, MPITYPE, file_endian, p->buf_write, p->dims,
                        p->my_size.data(), p->data_size.data(),
                        p->my_offset.data(), p->my_size.data(),
                        mem_start.data(), MPI_ORDER_C, MPI_COMM_WORLD);
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
    err = Mesh_IO_read(f, 0, MPITYPE, file_endian, p->buf_read, p->dims,
                        p->my_size.data(), p->data_size.data(),
                        p->my_offset.data(), p->my_size.data(),
                        mem_start.data(), MPI_ORDER_C, MPI_COMM_WORLD);
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
    string size_str = vectorToStr(p->data_size);
    printf("%d ranks, %.6f GiB data in %s (%s), %d x %d MiB stripes\n",
           np, (double)total_size / GB,
           p->filename, size_str.c_str(),
           p->stripe_count, p->stripe_len);
    computeStats(&s, write_times);
    printf("write speed GiB/s %.6f .. %.6f, avg %.3f, median %.3f, "
           "stddev %.1f%%\n",
           total_size / (GB * s.max),
	   total_size / (GB * s.min),
           total_size / (GB * s.avg),
           total_size / (GB * s.median), s.std_dev / s.avg * 100);

    computeStats(&s, read_times);
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
void computeStats(Stats *s, const vector<double> &data_orig) {
  double sum = 0, sum2 = 0;
  vector<double> data(data_orig);
  int count = data.size();

  if (count == 0) {
    s->min = s->max = s->avg = s->std_dev = s->median = 0;
    return;
  }

  sort(data.begin(), data.end());

  if (count >= 5) {
    int trim = count / 5;
    data.erase(data.begin(), data.begin()+trim);
    data.erase(data.end()-trim, data.end());
    count = data.size();
  }

  s->min = s->max = data[0];
  
  for (size_t i=0; i < count; i++) {
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

