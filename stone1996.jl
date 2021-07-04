using DelimitedFiles
using Plots
using Revise
using LinearAlgebra
includet("stone1996-module.jl")

using .Stone1996

inputs = read_inputs("data/image1.txt",
                     "data/image2.txt",
                     "data/shifts.txt");

wts = vec(readdlm("data/wts.0"))
wts_final = vec(readdlm("data/wts.990"))


typeof(wts)
varinfo()  ## show all variable info

net = new_net(inputs, wts);

forward(net, 1)
backward(net)
remember(net)
plot(net.dUdw)
forward(net, 2)
backward(net)
remember(net)
@show net.U, net.V



net = new_net(inputs, wts);
for epoch=1:10000
    clear(net)
    for i=1:1000
        forward(net, i)
        backward(net)
        remember(net)
        if i==999
            @show net.F, net.V, net.U
        end
    end
    adapt(net)
end

    
@time allz = [forward(net, x) for x in 1:1000]
plot(allz, xlabel="input number", ylabel="output")



# Check that things are still working...

netf = new_net(inputs, wts_final)
@time allz = [forward(netf, x) for x in 1:1000]
plot(allz, xlabel="input number", ylabel="output")


z  = []
zb = []
zt = []
for j=1:4
    for i=1:1000
        forward(netf, i)
        backward(netf)
        remember(netf)
        push!(z, netf.z[22])
        push!(zb, netf.zbar)
        push!(zt, netf.ztilde)
        if i==999
            @show netf.F, netf.V, netf.U
        end
    end
end

plot(z)
plot!(zb)
plot!(zt)
savefig("smoothed.png")


