using RecursivePartition
using Documenter

DocMeta.setdocmeta!(
    RecursivePartition,
    :DocTestSetup,
    :(using RecursivePartition);
    recursive = true,
)

makedocs(;
    modules = [RecursivePartition],
    authors = "Douglas Corbin <dfcorbin98@gmail.com>",
    repo = "https://github.com/dfcorbin/RecursivePartition.jl/blob/{commit}{path}#L{line}",
    sitename = "RecursivePartition.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Polynomial Basis Expansion" => [
            "Legendre Polynomials" => "legendre.md"
            "Polynomial Chaos Basis" => "pcb.md"
        ],
        "Recursive Partitioning" => "partition.md",
        "Regression" => "regression.md",
    ],
)


deploydocs(; repo = "github.com/dfcorbin/RecursivePartition.jl.git", devbranch = "main")
