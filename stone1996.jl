using DelimitedFiles
using Plots
using Revise
includet("stone1996-module.jl")

using .Stone1996

inputs = read_inputs("data/image1.txt",
                     "data/image2.txt",
                     "data/shifts.txt");

wts = vec(readdlm("data/wts.0"))
wts_final = vec(readdlm("data/wts.990"))


typeof(wts)
varinfo()  ## show all variable info


function new_net(inputs, wts)
    z = zeros(22)
    z1 = zeros(22)
    lambda_s = 2.0^ 1/32
    lambda_l = 2.0^ 1/3200
    U = 0.0
    V = 0.0
    F = 0.0
    ztilde = 0.0
    zbar = 0.0
    nwts = 120
    dUdw = zeros(nwts)
    dVdw = zeros(nwts)
    dzdw = zeros(nwts)
    dzdw1 = zeros(nwts)
    dztildedw1 = zeros(nwts)
    dztildedw = zeros(nwts)
    dzbardw1 = zeros(nwts)
    dzbardw = zeros(nwts)
    dFdw = zeros(nwts)
    net = Net(inputs, wts, 11, 10, 1,
              lambda_s, lambda_l,
              U, V, F, 
              ztilde, zbar,
              z, z1,
              dUdw, dVdw,
              dzdw, dzdw1,
              dztildedw, dztildedw1,
              dzbardw, dzbardw1,
              dFdw
              )
    net
end

net = new_net(inputs, wts)
forward(net, 1)
backward(net)

@time allz = [forward(net, x) for x in 1:1000]
plot(allz, xlabel="input number", ylabel="output")


netf = new_net(inputs, wts_final)
@time allz = [forward(netf, x) for x in 1:1000]
plot(allz, xlabel="input number", ylabel="output")


