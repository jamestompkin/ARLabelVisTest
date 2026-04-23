% Run rdg_allpairs_admm (joint ADMM variant) on icosphere_sub3.
this_dir = fileparts(mfilename('fullpath'));
rgd_dir = fullfile(this_dir, '..', 'external', 'matlab_rgd');
addpath(rgd_dir);
addpath(fullfile(rgd_dir, 'utils'));

oldwd = cd(rgd_dir);
cleanupObj = onCleanup(@() cd(oldwd));

Mm = MeshClass('icosphere_sub3');
fprintf('mesh: nv=%d  nf=%d\n', Mm.nv, Mm.nf);

alpha_hat0 = 0.25;

tic;
U = rdg_allpairs_admm(Mm, alpha_hat0);
elapsed = toc;

fprintf('allpairs_admm: alpha_hat=%.4g  time=%.2fs  max(U)=%.3e\n', ...
    alpha_hat0, elapsed, max(U(:)));

% Dump per-source argmax (so we can compare without shipping a 642x642 matrix).
[~, argmax_per_source] = max(U, [], 1);
writematrix(argmax_per_source, fullfile(this_dir, 'matlab_allpairs_admm_joint_argmax_p25.csv'));
% And a modest subsample of U itself for direct numeric diff.
writematrix(U, fullfile(this_dir, 'matlab_allpairs_admm_joint_U_p25.csv'));
fprintf('wrote argmax and U\n');
