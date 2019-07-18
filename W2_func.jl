using LinearAlgebra
function W2_metric(f, g, t, Nt, dt)
# normalization
    f = f ./ (norm(f,1)*dt);
    g = g ./ (norm(g,1)*dt);
# numerical integration
    F = zeros(Nt);
    G = zeros(Nt);
    for i = 1:Nt
        F[i] = sum(f[1:i])
        G[i] = sum(g[1:i])
    end
    F = F .* dt;
    G = G .* dt;
# inverse of G 
    G_inv = zeros(Nt);
    for i = 1:Nt
        y = F[i]
        ind_g = findall(x -> x >= y, G)
        if length(ind_g) == 0
            G_inv[i] = t[end];
        else
            G_inv[i] = t[ind_g[1]];
        end
    end
    w2 = sum((t-G_inv).^2 .* f * dt)
    return sqrt.(w2)
end


function compute_adj_source_core(f, g, t, Nt, dt)
# normalization
    f = f ./ (norm(f,1)*dt);
    g = g ./ (norm(g,1)*dt);
    
# numerical integration
    U = ones(Nt, Nt);
    U = LowerTriangular(U);
    F = U * f * dt;
    G = U * g * dt;

    ind = findall(x-> x==maximum(F), F)
    F[ind] .= maximum(F)
    ind = findall(x-> x==minimum(F), F)
    F[ind] .= minimum(F)

    ind = findall(x-> x==maximum(G), G)
    G[ind] .= maximum(F)
    ind = findall(x-> x==minimum(G), G)
    G[ind] .= minimum(F)
    
# inverse of G 
    G_inv = zeros(Nt);
    for ind = 1:Nt
        if abs(F[ind]-G[ind]) < 1e-3
            G_inv[ind] = t[ind]
        else
            y0 = F[ind]
            ind_g = findall(x -> x >= y0, G)
            ind_l = findall(x -> x < y0, G)
            yg = G[ind_g[1]]
            yl = G[ind_l[end]]
            tg = t[ind_g[1]]
            tl = t[ind_l[end]]
            k = (yg-yl) / (tg-tl)
            b = -k*tl+yl
            t0 = 1/k*(y0-b)

            G_inv[ind] = t0
        end
    end
    
    U = ones(Nt, Nt);
    U = UpperTriangular(U);
    A = U * diagm(0=>(-2 * f./ G_inv*dt));
    A = A + diagm(0=>t-G_inv)
    return A * (t-G_inv) * dt
end