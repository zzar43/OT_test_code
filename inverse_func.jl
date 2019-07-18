include("acoustic_solver.jl")

# compute gradient
# this function is only for change c, fix rho
function compute_grad(received_data, c0, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position);

    source_num = size(source_position,1)

    # forward
    d, u =  multi_solver(c0, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position);
    utt = similar(u);
    for i = 2:Nt-1
        utt[:,:,i,:] = (u[:,:,i+1,:]-2*u[:,:,i,:]+u[:,:,i-1,:]) ./ (2*dt);
    end
    utt[:,:,:,:] = utt[:,:,end:-1:1,:];

    # adjoint source
    adj_source = d - received_data;
    # adj_source = adj_source[end:-1:1,:,:];

    # backward wavefield
    v = backward_solver(c0, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position);

    # Integration 
    grad = zeros(Nx, Ny);
    for i = 1:source_num
        g = utt[:,:,:,i] .* v[:,:,:,i];
        g = sum(g, dims=3)
        grad += g[:,:,1];
    end
    return grad
end