function test_legendre_references()
    @test true
    x1, x2, x3 = 0.0, 0.5, 1.0
    @test legendre_next(2, x1, x1, 1.0) == -0.5
    @test legendre_next(2, x2, x2, 1.0) == -0.125
    @test legendre_next(2, x3, x3, 1.0) == 1.0
    @test legendre_poly(2, x1) == -0.5
    @test legendre_poly(2, x2) == -0.125
    @test legendre_poly(2, x3) == 1.0
end


function test_legendre_domain()
    error = DomainError(1.1, "Legendre polynomials defined on [-1,1].")
    @test_throws(error, legendre_next(1, 1.1, 1.1, 1.0))
    @test_throws(error, legendre_poly(1, 1.1))
end


function test_legendre()
    test_legendre_references()
    test_legendre_domain()
end


test_legendre()
