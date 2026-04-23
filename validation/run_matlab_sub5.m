% Run rdg_ADMM on icosphere_sub5 at several alpha_hat values, dump u.
this_dir = fileparts(mfilename('fullpath'));
rgd_dir = fullfile(this_dir, '..', 'external', 'matlab_rgd');
addpath(rgd_dir);
addpath(fullfile(rgd_dir, 'utils'));
out = this_dir;

oldwd = cd(rgd_dir);
cleanupObj = onCleanup(@() cd(oldwd));

Mm = MeshClass('icosphere_sub5');
fprintf('mesh: nv=%d  nf=%d\n', Mm.nv, Mm.nf);

alphas = [0.05, 0.25, 1.25];
src = 1;

for k = 1:length(alphas)
    a = alphas(k);
    tic;
    u = rdg_ADMM(Mm, src, 'alpha_hat', a);
    elapsed = toc;

    suffix = strrep(sprintf('%g', a), '.', 'p');
    fname = fullfile(out, sprintf('matlab_sub5_u_alphahat_%s.csv', suffix));
    writematrix(u, fname);

    [~, argmax_idx] = max(u);
    fprintf('alpha_hat=%.4g  u(src)=%.6e  max(u)=%.6e  argmax=%d  time=%.2fs\n', ...
        a, u(src), max(u), argmax_idx, elapsed);
end

disp('done');
