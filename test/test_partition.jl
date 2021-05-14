function test_insert_knot()
    knotmat = repeat([-1.0 1.0], 2, 1)
    P = [splitmat(knotmat, 1, 0.0)...]
    @test P[1] == [-1.0 0.0; -1.0 1.0]
    @test P[2] == [0.0 1.0; -1.0 1.0]
    P1 = insert_knot(P, 1, 2, 0.0)
    @test P1 == [[-1.0 0.0; -1.0 0.0], [0.0 1.0; -1.0 1.0], [-1.0 0.0; 0.0 1.0]]
    domerr = DomainError(0.0, "uncontained knot.")
    @test_throws(domerr, insert_knot(P, 1, 1, 0.0))
    argerr = ArgumentError("Left col of knotmat must be < right col.")
    faulty_kmat = [1.0 0.0; 1.0 0.0]
    @test_throws(argerr, splitmat(faulty_kmat, 1, 0.5))
end


function test_is_contained()
    kmat = [-1.0 1.0; 0.0 0.5]
    X = [1.0 0.0; 1.0 0.5]
    upper = [1.0, 1.0]
    @test is_contained(X, kmat, upper) == [true, false]
end


function test_which_subset()
    P = [[-1.0 0.0; -1.0 1.0], [0.0 1.0; -1.0 1.0]]
    x = [1.0, -1.0]
    @test which_subset(x, P) == 2
end


function test_partition()
    P = [[-1.0 0.0; -1.0 1.0], [0.0 1.0; -1.0 1.0]]
    X = [-1.0 0.0; 1.0 0.0]
    y = [1.0, 2.0]
    Xsubs, ysubs, rows = partition(X, P, y; track = true)
    @test (Xsubs[1] == [-1.0 0.0]) && (ysubs[1] == [1.0])
    @test (Xsubs[2] == [1.0 0.0]) && (ysubs[2] == [2.0])
    @test rows == [[1], [2]]
    Xsubs, rows = partition(X, P, y; track = true)
    @test (Xsubs[1] == [-1.0 0.0])
    @test (Xsubs[2] == [1.0 0.0])
    @test rows == [[1], [2]]
    # Test what happens when one subset is empty
    X1 = [-1.0 0.0]
    y1 = [1.0]
    Xsubs1, ysubs1, rows1 = partition(X1, P, y1; track = true)
    @test (Xsubs1[1] == [-1.0 0.0]) && (ysubs1[1] == [1.0])
    @test size(Xsubs1[2]) == (0, 2) && length(ysubs1[2]) == 0
    @test rows1 == [[1], []]
end


function test_all_partition()
    test_insert_knot()
    test_is_contained()
    test_which_subset()
    test_partition()
end


test_all_partition()
