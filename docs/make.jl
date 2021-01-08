using RecursivePartition
using Documenter

makedocs(;
    modules=[RecursivePartition],
    authors="Douglas Corbin <dfcorbin98@gmail.com>",
    repo="https://github.com/dfcorbin/RecursivePartition.jl/blob/{commit}{path}#L{line}",
    sitename="RecursivePartition.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
