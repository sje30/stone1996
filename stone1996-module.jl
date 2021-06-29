# module for Stone1996

module Stone1996

using DelimitedFiles
export read_inputs, test, Net, forward, backward

test() = 5

mutable struct Net
    inputs::Array{Float64,2}
    w::Vector{Float64}
    ni::Int
    nj::Int
    nk::Int
    lambda_s::Float64
    lambda_l::Float64
    U::Float64
    V::Float64
    F::Float64
    
    ztilde::Float64
    zbar::Float64
    ## now the memories
    z::Vector{Float64}
    z1::Vector{Float64}
    dUdw::Vector{Float64}
    dVdw::Vector{Float64}
    dzdw::Vector{Float64}
    dzdw1::Vector{Float64}
    dztildedw::Vector{Float64}
    dztildedw1::Vector{Float64}
    dzbardw::Vector{Float64}
    dzbardw1::Vector{Float64}
    dFdw::Vector{Float64}
end


function read_inputs(left, right, target)
    l = readdlm(expanduser(left))
    r = readdlm(expanduser(right))
    t = readdlm(expanduser(target))
    l = convert(Array{Float64}, l[1,1:7000])
    r = convert(Array{Float64}, r[1,1:7000])
    out = zeros(1000, 12)
    for row in 1:1000
        index = (7*(row-1))+1
        for i in 1:5
            out[row,i]   = l[index+i-1]
            out[row,i+5] = r[index+i-1]
            out[row,11]  = +1.0   # bias
            out[row,12]  = t[index+2]
        end
    end
    out
end


function wij(i, j, ni)
   ((j-1)*ni) + i
end

function wjk(j, k, ni, nj)
   ((k-1)*j) + j + (ni*nj)
end

function ak(k, ni, nj)
    ni+nj+k
end

function aj(j, ni)
    ni+j
end

function ai(i)
    i
end


"""
Forward propagation of activity
WHICH is the input sample to propagate forward.
"""
function forward(net, which)
    ni = net.ni
    nj = net.nj
    nk = net.nk
    units = ni + nj + nk
    for i in 1:11
        net.z[i]   = net.inputs[which,i]
    end


    
    ##input to hidden
    n=1
    for j in 1:nj 
        tot = 0.0
        for i in 1:ni
            
            @assert n == wij(i,j, ni)
            tot = tot + (net.w[n] * net.z[i])
            n = n+1
        end
        net.z[ni+j] = tanh(tot)
    end

    ## hidden to output
    k = 1
    tot = 0.0
    for j in 1:nj
        @assert n == wjk(j,k, ni, nj)
        tot = tot + (net.w[n] * net.z[j+ni])
        n = n+1
    end
    net.z[ni+nj+1] = tot
end


"""
Propagate backward
"""
function backward(net)
    ## Propagate errors backward
    ni = net.ni
    nj = net.nj
    k = ak(1, ni, nj)
    net.ztilde = (net.lambda_s * net.ztilde) + ( 1 - net.lambda_s) * net.z1[ k ]
    net.zbar   = (net.lambda_l * net.zbar)   + ( 1 - net.lambda_l) * net.z1[ k ]


    ## A.6
    net.U = net.U + 0.5*(net.zbar   - net.z[ k ])^2
    net.V = net.V + 0.5*(net.ztilde - net.z[ k ])^2
    @show net.U, net.V

    ## ∂z/∂w for w_jk weights (A10)
    for j = 1:nj, k=1
        net.dzdw[ wjk(j, k, ni, nj) ] = net.z[ aj(j, ni)]
    end

    ## ∂z/∂w for w_ij (underneath A11)
    for j = 1:nj, i=1:ni
        net.dzdw[ wij(i, j, ni) ] = net.w[wjk(j,1,ni, nj)] *
            (1. - net.z[aj(j, ni)]^2) * net.z[ ai(i) ]
    end

    ## A8:  dz̃dw (could vectorize)
    for w in 1:length(net.w)
        net.dztildedw[ w ] = net.lambda_s * net.dztildedw1[ w ] +
            (1. - net.lambda_s) * net.dzdw1[ w ]
        net.dzbardw[ w ] = net.lambda_l * net.dzbardw1[ w ] +
            (1. - net.lambda_l) * net.dzdw1[ w ]
    end

    ## dUdW (A7)
    k1 = (net.ztilde - net.z[ ak(1, ni, nj)])
    k2 = (net.zbar   - net.z[ ak(1, ni, nj)])
    for w in 1:length(net.w)
        net.dUdw[ w ] = net.dUdw[ w ] + k1*(net.dztildedw[ w ] - net.dzdw[ w ])
        net.dVdw[ w ] = net.dVdw[ w ] + k2*(net.dzbardw[ w ]   - net.dzdw[ w ])
    end

    ## Now we update F and dF/dw
    net.F = log(net.V / net.U)
    net.dFdw = @. (1/net.V) * net.dVdw - (1/net.U) * net.dUdw

    return nothing
end


end                             # end of module
