# RecursivePartition.jl
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dfcorbin.github.io/RecursivePartition.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dfcorbin.github.io/RecursivePartition.jl/dev)
[![Build Status](https://travis-ci.com/dfcorbin/RecursivePartition.jl.svg?branch=main)](https://travis-ci.com/dfcorbin/RecursivePartition.jl)
[![Coverage](https://codecov.io/gh/dfcorbin/RecursivePartition.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dfcorbin/RecursivePartition.jl)

RecurisvePartitionin.jl is a library implementing flexible non-parametric regression models based on (hyperrectangular) recursive partitions. The intended user for this package falls primarily into two categories:

* Those who wish to construct and make use of recursive partitions, either to manipulate data or to incorperate them into their own algorithms.
* Those seeking to experiment with or use the partitioned regression models (or other implemented regression models) to analyse data.

For a detailed description of the library's features, see the [full documentation](https://dfcorbin.github.io/RecursivePartition.jl/stable).

## Installation

TBC - Currently trying to get this library on the julia registry.

## Recursive Partitioning

Suppose we have a data set of 2D coordinates (each element bound between [0,1]) and we wish to divide them into three disjoint subregions as pictured below. 

![Partition Demo](https://github.com/dfcorbin/RecursivePartition.jl/blob/main/docs/src/figures/partition_demo2.png?raw=true)

we can achieve this using this code

```julia
using RecursivePartitioning

# Data is stored in a matrix
X = rand(1000, 2)

# Create a 2 x 2 matrix where the 1st row gives the upper/lower bound
# of the 1st element in coordinate etc...
P = [repeat([0.0 1.0], 2, 1)]
insert_knot!(P, 1, 1, 0.0) # Split subregion 1 (the full space) - in dimension 1 (vertically) - at 0.0
insert_knot!(P, 1, 2, 0.0) # Split subregion 1 (now the left subregion) - in dimension 2 (horizontally) - at 0.0

# Return a (Vector{Matrix{Float64}}, Vector{Int64}) Tuple where each element of
# X_subsets corresponds to a subregion in the partition.
X_subsets, rows = partition(X, P; track=true)
```

## Non-Parametric Regression

In essence, the partitioned models implemented in this package are a collection of simple (low complexity and thus quick to fit) non-parametric regression
models which each relate to a specific subregion in the partition. If a sensible partition is known in advance, this can be specified by the user. Alternatively,
a novel algorithm is implemented which will recursively divide up the space and automatically detect when the partition is of sufficient granularity.

A simple 1D example is presented here.

```julia
using RecursivePartition

# Generate the data
N = 10000
SD = 0.5
f(x) = 2 * sin(2 * π * x[1]) * x[1] - 2 * x[1]^2
x, y = gendat(N, SD, f, 1; bounds=[-1.0, 1.0])
```

The generated data looks like,

![1D Data](https://github.com/dfcorbin/RecursivePartition.jl/blob/main/docs/src/figures/fplot.png?raw=true)

where the dashed red line represents the true underlying structure of the data, which we are trying to approximate. First we divide the region
directly in the middle (`x = 0.0`) and fit a partitioned model.

```julia
P = [[-1.0 1.0]] # Problem is 1D so we only need a single row in the matrix.
insert_knot!(P, 1, 1, 0.0)
fixed_partition_model = partition_polyblm(X, y, 3, [-1.0, 1.0]) 
```

The model outputted here is shown below.

![Static Partition](https://github.com/dfcorbin/RecursivePartition.jl/blob/main/docs/src/figures/part.png?raw=true)

let's see if we can improve upon this model by using RecursivePartition.jl's automatic partitioning algorithm.

```julia
auto_partition_model = auto_partition_polyblm(X, y, [-1.0, 1.0])
```

![Auto Partition](https://github.com/dfcorbin/RecursivePartition.jl/blob/main/docs/src/figures/autopart.png?raw=true)

Indeed, the automatic partitioning algorithm has recommended we split each subset once more, but no further. For further
detail on how to produce these figures, and using partitioned models for prediction, the reader is once again
referred to the relevant sections of [the documentation](https://dfcorbin.github.io/RecursivePartition.jl/dev/regression/).

## Contributions

**Author:** Douglas Corbin

**PhD Supervisors:** Anthony Lee, Mathieu Gerber
