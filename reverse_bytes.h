#ifndef __REVERSE_BYTES_H__
#define __REVERSE_BYTES_H__

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reverse bytes in-place. */
void reverseBytes(void *v, int element_size, size_t count);

/* Copy data from src to dest, reversing bytes at the same time. */
void reverseBytesCopy(void *dest, const void *src, int element_size,
                      size_t count);

#ifdef __cplusplus
}
#endif

#endif
