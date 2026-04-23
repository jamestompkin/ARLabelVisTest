% All-pairs RGD on icosphere_sub3 at alpha_hat = 0.25, dump max_indices.
% Mirror of demo.m for validation purposes.
this_dir = fileparts(mfilename('fullpath'));
rgd_dir = fullfile(this_dir, '..', 'external', 'matlab_rgd');
addpath(rgd_dir);
addpath(fullfile(rgd_dir, 'utils'));

oldwd = cd(rgd_dir);
cleanupObj = onCleanup(@() cd(oldwd));

Mm = MeshClass('icosphere_sub3');
fprintf('mesh: nv=%d  nf=%d\n', Mm.nv, Mm.nf);

alpha_hat0 = 0.25;
num_iterations = Mm.nv;
max_indices = zeros(1, num_iterations);

if isempty(gcp('nocreate'))
    parpool;
end

tic;
parfor i = 1:num_iterations
    u = rdg_ADMM(Mm, i, 'alpha_hat', alpha_hat0);
    [~, max_idx] = max(u);
    max_indices(i) = max_idx;
end
elapsed = toc;
fprintf('all-pairs time: %.2fs\n', elapsed);

out = fullfile(this_dir, 'matlab_allpairs_icosphere_sub3_alpha_p25.csv');
writematrix(max_indices, out);
fprintf('wrote %s\n', out);
