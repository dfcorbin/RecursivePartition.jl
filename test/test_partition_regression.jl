function test_PartitionModel()
    P = [repeat([-1.0 1.0], 2, 1)]
    insert_knot!(P, 1, 1, 0.0)
    coeff1, coeff2 = [5.0, 10.0], [1.0, 8.0]
    f(x) = x[1] < 0.0 ? x' * coeff1 : x' * coeff2 # piecewise linear
    X, y = gendat(5000, 1.0, f, 2)
    Xv, yv = partition(X, P, y)
    blmpart = partition_blm(X, y, P)
    leftblm = BayesLinearModel(Xv[1], yv[1])
    rightblm = BayesLinearModel(Xv[2], yv[2])
    @test get_loc_scalepost(blmpart, 1) == get_scalepost(leftblm)
    @test get_loc_scalepost(blmpart, 2) == get_scalepost(rightblm)
    # Perform same check for PolyBLM
    polypart = partition_polyblm(X, y, P)
    leftpolyblm = PolyBLM(Xv[1], yv[1], 3, P[1])
    rightpolyblm = PolyBLM(Xv[2], yv[2], 3, P[2])
    @test get_loc_scalepost(polypart, 1) == get_scalepost(leftpolyblm)
    @test get_loc_scalepost(polypart, 2) == get_scalepost(rightpolyblm)

    # Now test auto partitioned models line up with normal partitioned models.
    autoblm = auto_partition_blm(X, y, [-1.0, 1.0])
    autoblm_P = get_P(autoblm)
    fixedblm = partition_blm(X, y, P)
    @test get_logev(autoblm) ≈ get_logev(fixedblm)
    auto_partition_polyblm(X, y, [-1.0, 1.0])
end


function test_all_partition_regression()
    test_PartitionModel()
end

test_all_partition_regression()
