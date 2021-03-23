module SSPOC

const DIR = @__DIR__
using Pkg
Pkg.activate(DIR*"/../")

using DelimitedFiles, LinearAlgebra, Statistics, Convex, COSMO, StatsBase, Flux
using Flux.Data:DataLoader
using Flux.Losses: logitbinarycrossentropy
using Flux: onecold, onehot
using Flux: onehotbatch
using StatsBase, PyCall

plt = pyimport("matplotlib.pyplot")

cats = Float32.(readdlm(DIR*"/../data/cats.csv", ',')) #|> x -> x ./ 255
dogs = Float32.(readdlm(DIR*"/../data/dogs.csv", ',')) #|> x -> x ./ 255

function run_demo(cats, dogs, n, r)
    out=zeros(n)
    for i = 1:n
        @label start
        cats = deepcopy(cats[:, sample(1:size(cats, 2), size(cats, 2), replace=false)]);
        dogs = deepcopy(dogs[:, sample(1:size(dogs, 2), size(dogs, 2), replace=false)]);

        # create training and testing sets (.74, .25)
        train1 = [cats[:,1:60] dogs[:,1:60]];
        test1 = [cats[:,61:end] dogs[:,61:end]];

        # compute PCA and project data to get eigenrepresentation X̃
        n, m = size(train1)
        ## centering matrix
        C = I(m) - (1/m) * ones(m,m)
        train_centered = train1 * C
        U, S, V = svd(train_centered)
        # plt.plot(s)
        # plt.show()
        Ψ = U[:,1:r];
        X̃ = Ψ' * train1;

        # compute maximally discriminating component using LDA
        ## between-classes scatter Sb
        μ1 = mean(X̃[:, 1:60], dims=2);
        μ2 = mean(X̃[:,61:end], dims=2);
        Sb = (μ2 - μ1) * (μ2 - μ1)';

        function scatter_matrix(data, mu)
            (data - mu .* ones(size(data))) * (data - mu .* ones(size(data)))'
        end
        ## within-classes scatter Sw
        Σ1 = scatter_matrix(X̃[:,1:60], μ1);
        Σ2 = scatter_matrix(X̃[:,61:end], μ2);
        Sw = Σ1 + Σ2;

        # generalized eigenvalue decomposition to get projection matrix W
        Λ, W = eigen(Sb, Sw);
        ## choose eigenvector corresponding to largest eigenvalue w (this represents the maximally discriminating projection vector in the new subspace)
        Λ = reverse(real.(Λ));
        W = reverse(W, dims=2);
        w = real(W[:,1])

        # solve for S (this is the maximally sparse vector in original space that when projected to subspace approximates w)
        n = size(train1, 1)
        s = Variable(n)
        objective = norm(s,1)
        constraint = Ψ' * s == w;
        solver = COSMO.Optimizer;
        problem = minimize(objective, constraint);
        solve!(problem, solver, silent_solver=true)
        ## check linear separability imposed by sparse vector S
        if isnothing(s.value)
            @warn "Encountered a numerical error during optimization!  Repeating sample..."
            num_err += 1
            @goto start
        end
        S = reshape(s.value, n)
        sep = (Ψ' * S)' * (Ψ' * train1)
        # plt.bar(1:length(sep), sep)
        # plt.show()

        # identify indices above threshold
        ## see how close to sparse the sparse array is
        #StatsBase.describe(S)
        ## choose threshold
        # norm(S, 2) / (2*2*60)
        # C = (abs.(S) .> 0.35)
        # n = norm(S,2) / (2 * 2 * 60)
        # C = (abs.(S) .> n)
        thresh = abs(S[reverse(sortperm(abs.(S)))[21]])
        C = abs.(S) .> thresh;
        #C = zeros(length(S))
        #sum(C)


        # create measurement matrix
        Φ = Diagonal(C);

        # scale data (z score transform)
        dt = StatsBase.fit(ZScoreTransform, train1, dims=2);
        StatsBase.transform!(dt, train1);

        # sparsify original data with measurement matrix
        #train2 = train1 + .1 * randn(size(train1))
        y = Φ * train1;
        #y = Φ * hcat(train1, train2)

        ## see what pixels will be used
        # function pl(i)
        # cat_sparse_pixels = deepcopy(cats[:, i])
        # dog_sparse_pixels = deepcopy(dogs[:, i])
        # eig_sparse_pixels = deepcopy(Ψ[:,1])
        # cat_sparse_pixels[C] .= NaN
        # eig_sparse_pixels[C] .= NaN
        # dog_sparse_pixels[C] .= NaN
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        # cmap = plt.cm.gray
        # cmap.set_bad((1, 0, 0, 1))
        # ax1.imshow(reshape(cat_sparse_pixels, (64,64)), cmap=cmap)
        # ax3.imshow(reshape(eig_sparse_pixels, (64,64)), cmap=cmap)
        # ax2.imshow(reshape(dog_sparse_pixels, (64,64)), cmap=cmap)
        # plt.show()
        # end

        # pl(36)

        # create labels
        #labels = Int.(vcat(ones(60), zeros(60)));
        labels = Int.(vcat(ones(60), zeros(60)))
        l = onehotbatch(labels, [0, 1]);

        # prepare data for NN
        train = DataLoader((y, l), batchsize=32);

        # build logistic regression model with L1 regularization

        function loss(x, y)
            logitbinarycrossentropy(m(x), y) + .1 * norm(params(m), 1)
            #logitbinarycrossentropy(m(x), y)
        end

        # optimizer
        #opt = ADAM(1e-3)
        # opt = Descent(1e-2)
        opt = ADADelta(.5)
        m = Dense(4096, 2)
        # train for 100 epochs
        for epoch in 1:100
            Flux.train!(loss, params(m), train, opt)
        end

        # prepare test data
        test1 = [cats[:,61:end] dogs[:,61:end]];
        StatsBase.transform!(dt, test1);
        y_test = Φ * test1;
        labels_test = Int.(vcat(ones(20), zeros(20)));

        out[i] = mean(onecold(sigmoid.(m(y_test)), [0,1]) .== labels_test)
        @info "Fold "*string(i)*" accuracy: "*string(out[i])
    end
mean(out)
end #run_demo

end # module
