# meshio
Mesh_IO - parallel IO for mesh-structured data

This is a thin wrapper around collective MPI-IO routines for efficiently accessing
mesh-structured data files.

Features supported
 * Multiple distinct meshes in one file (all operations are based on a given file offset)
 * Row-major (C) or column-major (Fortran) arrays
 * Different byte orderings - not automatic; specified by user
 * Striped data files - the MPI-IO routines will automatically access a striped data file in parallel
 * Avoid flooding file servers - MPI-IO will use one IO process for each stripe in the file. For example, if the file is striped 64 ways and 1024 processes are accessing it, only 64 of those processes will do IO, and they will distribute results to or from the other processes.
 * n-dimensional data; not limited to 2 or 3 dimensions
 
Features not supported
 * Complex datatype endian support. Endian modifications are only supported for basic datatypes.
 * Irregular-shaped data. Only n-dimensional rectangular grids of data are supposed.

It's just two functions: Mesh_IO_read() and Mesh_IO_write(). A description of the array references is below. See mesh_io.h for more details.

The data read may be a subset of the mesh encoded in the file, and it may be a subset of the mesh encoded in memory. 

For example, let the file contain a 6x10 mesh of data. We wish to read the upper-right quadrant into an in-memory array with 1 halo cell on each side.

file:

    . . . . . d d d d d
    . . . . . d d d d d
    . . . . . d d d d d
    . . . . . . . . . .
    . . . . . . . . . .
    . . . . . . . . . .

memory:

    . . . . . . .
    . d d d d d .
    . d d d d d .
    . d d d d d .
    . . . . . . .

 * mesh_size = {3, 5};
 * file_mesh_sizes = {6, 10};
 * file_mesh_starts = {0, 5};
 * memory_mesh_sizes = {5, 7};
 * memory_mesh_starts = {1, 1}

