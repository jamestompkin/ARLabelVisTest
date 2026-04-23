% Run rdg_ADMM with reg='vfa' on icosphere_sub3, given the Python-generated vf.
this_dir = fileparts(mfilename('fullpath'));
rgd_dir = fullfile(this_dir, '..', 'external', 'matlab_rgd');
addpath(rgd_dir);
addpath(fullfile(rgd_dir, 'utils'));

oldwd = cd(rgd_dir);
cleanupObj = onCleanup(@() cd(oldwd));

Mm = MeshClass('icosphere_sub3');
fprintf('mesh: nv=%d  nf=%d\n', Mm.nv, Mm.nf);

vf = readmatrix(fullfile(this_dir, 'vfa_test_vf.csv'));
fprintf('vf: size=%dx%d  max|norm-1|=%.2e\n', size(vf, 1), size(vf, 2), ...
    max(abs(vecnorm(vf, 2, 2) - 1)));

alpha_hat0 = 0.25;
beta_hat0  = 0.5;
src = 1;

tic;
u = rdg_ADMM(Mm, src, 'reg', 'vfa', ...
             'alpha_hat', alpha_hat0, 'beta_hat', beta_hat0, 'vf', vf);
elapsed = toc;

fprintf('vfa: alpha_hat=%.4g  beta_hat=%.4g  u(src)=%.3e  max(u)=%.3e  time=%.2fs\n', ...
    alpha_hat0, beta_hat0, u(src), max(u), elapsed);

out = fullfile(this_dir, 'matlab_u_vfa_alpha_p25_beta_p5.csv');
writematrix(u, out);
fprintf('wrote %s\n', out);
