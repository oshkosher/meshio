#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/*
  Speed in GiB/sec       element size (bytes)
                          2      4      8     16
  Core i7@3.40GHz
    GCC 4.8            3700   7500  13500  14800
  Xeon E5-2660
    GCC 4.9.2         14000   6200  11400  12600
    Intel icpc 15.0   14100   6600  12000  13300
  AMD Opteron 6276 Interlagos (Blue Waters):
    craycc             2700   3700   4600   4500
    pgic v15           1200   3700   2800   1000
    pgic v16           1200   3600   2800   1000
    icpc v15           3700   3000   3800   3900
    icpc v16           4100   3200   4000   4200
    gcc 4.9.3          4600   3400   4100   4700
    gcc 5.3.0          4600   3400   4400   4700

  There are also compiler intrinsics __builtin_bswap{16,32,64} on
  GCC 4.8 and up and on ICPC, but their performance seems identical 
  to the macros below.  Same thing with htobe{16,32,64} macros in endian.h

  Ed Karrels, edk@illinois.edu
*/
#define BSWAP_16(x)                             \
  (((uint16_t)(x) << 8) |                       \
   ((uint32_t)(x) >> 8))

#define BSWAP_32(x)                                       \
  (((uint32_t)(x) << 24) |                                \
   (((uint32_t)(x) << 8) & 0xff0000) |                    \
   (((uint32_t)(x) >> 8) & 0xff00) |                      \
   ((uint32_t)(x) >> 24))

#define BSWAP_64(x)                                                   \
  (((uint64_t)(x) << 56) |                                            \
   (((uint64_t)(x) << 40) & 0xff000000000000ULL) |                    \
   (((uint64_t)(x) << 24) & 0xff0000000000ULL) |                      \
   (((uint64_t)(x) <<  8) & 0xff00000000ULL) |                        \
   (((uint64_t)(x) >>  8) & 0xff000000ULL) |                          \
   (((uint64_t)(x) >> 24) & 0xff0000ULL) |                            \
   (((uint64_t)(x) >> 40) & 0xff00ULL) |                              \
   ((uint64_t)(x) >> 56))

#define	BSWAP_32_USING_64(x)                       \
  (((uint64_t)(x) & 0xff000000ffull) << 24 |       \
   ((uint64_t)(x) & 0xff000000ff00ull) << 8 |      \
   ((uint64_t)(x) & 0xff000000ff0000ull) >> 8 |    \
   ((uint64_t)(x) & 0xff000000ff000000ull) >> 24)


#define SWAP(a,b) temp=(a); (a)=(b); (b)=temp;


void reverseBytes(void *v, int element_size, size_t count) {
  if (element_size < 2) return;

  switch (element_size) {
  case 2: {
    uint16_t *p = (uint16_t*)v, *end;
    end = p + count;
    for (; p < end; p++)
      *p = BSWAP_16(*p);
    break;
  }

  case 4: {
    uint32_t *p = (uint32_t*)v, *end = p + count;
    uint64_t *p64, *p64end;

    /* handle unaligned prefix */
    if ((uintptr_t)p & 7) {
      *p = BSWAP_32(*p);
      p++;
    }

    /* process two 4-byte words at a time */
    p64 = (uint64_t*) p;
    p64end = (uint64_t*) ((uintptr_t)end & ~7);
    for (; p64 < p64end; p64++)
      *p64 = BSWAP_32_USING_64(*p64);
    
    p = (uint32_t*) p64;

    /* handle unaligned suffix */
    if (p < end)
      *p = BSWAP_32(*p);
    break;
  }

  case 8: {
    uint64_t *p = (uint64_t*)v, *end;
    end = p + count;
    for (; p < end; p++)
      *p = BSWAP_64(*p);
    break;
  }

  case 16: {
    uint64_t *p = (uint64_t*)v, *end, tmp;
    end = p + count*2;
    for (; p < end; p += 2) {
      tmp = BSWAP_64(*p);
      *p = BSWAP_64(*(p+1));
      *(p+1) = tmp;
    }
    break;
  }

  default: {
    /* Written by Daniel J. Bodony (bodony@Stanford.EDU)
       Copyright (c) 2001 */
    size_t i;
    int x;
    char *a, temp;
    
    for (i = 0; i < count; i++) {
      a = (char *)v + i*element_size;
      for (x = 0; x < element_size/2; x++) {
        SWAP(a[x],a[element_size-x-1]);
      }
    }
  }
  }
}


void reverseBytesCopy(void *dest, const void *src, int element_size,
                      size_t count) {
                      
  if (element_size < 2 || element_size & 1) {
    memcpy(dest, src, element_size * count);
    return;
  }

  switch (element_size) {
  case 2: {
    const uint16_t *r = (uint16_t*)src, *end = r + count;
    uint16_t *w = (uint16_t*)dest;
    for (; r < end; r++,w++)
      *w = BSWAP_16(*r);
    break;
  }

  case 4: {
    const uint32_t *r = (uint32_t*)src, *end = r + count;
    uint32_t *w = (uint32_t*)dest;

    for (; r < end; r++,w++)
      *w = BSWAP_32(*r);
    break;
  }

  case 8: {
    const uint64_t *r = (uint64_t*)src, *end = r + count;
    uint64_t *w = (uint64_t*)dest;

    for (; r < end; r++,w++)
      *w = BSWAP_64(*r);
    break;
  }

  case 16: {
    const uint64_t *r = (uint64_t*)src, *end = r + count*2;
    uint64_t *w = (uint64_t*)dest, tmp;
    for (; r < end; r += 2, w += 2) {
      tmp = BSWAP_64(*r);
      *w = BSWAP_64(*(r+1));
      *(w+1) = tmp;
    }
    break;
  }

  default: {
    /* Written by Daniel J. Bodony (bodony@Stanford.EDU)
       Copyright (c) 2001 */
    size_t i;
    int x;
    const char *r;
    char *w;
    
    for (i = 0; i < count; i++) {
      r = (const char *)src + i*element_size;
      w = (char *)dest + i*element_size;
      for (x = 0; x < element_size/2; x++) {
        w[element_size-x-1] = r[x];
        w[x] = r[element_size-x-1];
      }
    }
  }
  }
}
