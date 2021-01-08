using LinearAlgebra


function test_BLMHyper()
    @test_throws(ArgumentError("Shape/Scale must be greater than 0."),
        BLMHyper(2, 0.0, 0.0))
end




function test_BLM()
    coeff = [- 10.0, 15.0, 5.0]
    foo(x) = coeff[1] + coeff[2] * x[1] + coeff[3] * x[2]
    Random.seed!()
    X, y = gendat(500000, 0.5, foo, 2)
    mod = BayesLinearModel(X, y, 2)
    @test get_dim(mod) == 2
    @test get_scaleprior(mod) == 0.001
    @test get_shapeprior(mod) == 0.001
    @test get_coeffprior(mod) == zeros(Float64, 3)
    @test get_covinvprior(mod) == diagm(ones(Float64, 3))
    @test get_covprior(mod) ==  diagm(ones(Float64, 3))
    @test get_N(mod) == length(y)
    @test approxeq(get_coeffpost(mod), coeff, 0.01)
    # Now test predicition functions.
    Xtest, ytest = gendat(500, 0.5, foo, 2)
    pred1 = predict(mod, Xtest)
    pfun = predfun(mod)
    pred2 = mapslices(pfun, Xtest; dims=2)
    @test pred1 ≈ pred2
    truevals = mapslices(foo, Xtest; dims=2)
    truevals = reshape(truevals, (:))
    @test approxeq(pred1, truevals, 0.01)
    logevidence(mod)
end


function test_SBLM()
    # testfun(x) = x[1] * x[2]^2 - 2 * x[1]^3
    # X, y = gendat(500000, 0.5, testfun, 2; bounds=[0.0, 1.0])
    # maxparam = 100
    # m = SparsePoly(X, y, 5, [0.0, 1.0]; maxparam=maxparam)
    # coeff, cov = get_coeffpost(m), get_covpost(m)
    # @test length(coeff) == size(cov, 1) == length(get_indices(m)) + 1
    # @test length(coeff) <= maxparam
    # xtest = rand(2)
    # pred1 = predict(m, reshape(xtest, (1, :)))[1]
    # pred2 = predfun(m)(xtest)
    # truemean = testfun(xtest)
    # @test pred1 ≈ pred2
    # @test approxeq(pred1, truemean, 0.01)
end



function test_all_regression()
    test_BLMHyper()
    test_BLM()
    test_SBLM()
end


test_all_regression()
