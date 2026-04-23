% E2E MATLAB pipeline: run all-pairs rdg_ADMM on e2e_cielab_hull.off at alpha_hat=0.25.
this_dir = fileparts(mfilename('fullpath'));
rgd_dir = fullfile(this_dir, '..', '..', 'external', 'matlab_rgd');
addpath(rgd_dir);
addpath(fullfile(rgd_dir, 'utils'));

oldwd = cd(rgd_dir);
cleanupObj = onCleanup(@() cd(oldwd));

Mm = MeshClass('e2e_cielab_hull');
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
fprintf('MATLAB all-pairs time: %.2fs\n', elapsed);

out = fullfile(this_dir, 'matlab_max_indices.csv');
writematrix(max_indices, out);
fprintf('wrote %s\n', out);
