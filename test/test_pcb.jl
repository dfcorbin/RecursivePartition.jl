function test_rescaling()
    X = [-3.0 2.0; -3.0 2.0]
    bounds = [-3.0 1.0; 0.0 2.0]
    @test RecursivePartition.rescale(X[1, 1], bounds[1, :]) == -1.0
    @test RecursivePartition.rescale(X[1, :], bounds) == [-1.0, 1.0]
    @test RecursivePartition.rescale(X, bounds) == repeat([-1.0 1.0], 2, 1)
    error = DomainError(1.1, "x uncontained by [-1.0, 1.0]")
    @test_throws(error, RecursivePartition.rescale(1.1, [-1.0, 1.0]))
end


function test_pcbmat()
    X = [-2.0 2.0; 0.0 0.0]
    bounds = [-2.0 2.0; -2.0 2.0]
    @test trunc_pcbmat(X, 2, bounds) == [
        1.0 1.0 -1.0 -1.0 1.0
        0.0 -0.5 0.0 0.0 -0.5
    ]
    indices = [MVPIndex([1, 1], [1, 2]), MVPIndex([1], [2])]
    @test index_pcbmat(X, indices, bounds) == [
        -1.0 * 1.0 1.0
        0.0 0.0
    ]
end


function test_pcb()
    test_rescaling()
    test_pcbmat()
end


test_pcb()
