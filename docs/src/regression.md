# Regression

The objective of regression is to model the relationship between a set of
predictor and response variables. Once trained, the regression model can be used
to predict the response variable, given a new predictor variable.
`RecursivePartition` implements a variety of regression models, however its
most notable feature is the [`PartitionModel`](@ref) type. This model is
fitted by partitioning the space of predictor variables
(see [Recursive Partitioning](@ref) for a visualization of this process) and fitting
a flexible models within each subregion. One can think of this as a
*divide and conquer* approach to learning complex relationships between
predictors and responses.

## Bayesian Linear Models



## Partitioned Models


```@autodocs
Modules = [RecursivePartition]
Pages = ["regression.jl"]
```
