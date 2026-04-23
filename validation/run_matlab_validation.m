% Tier-1 and Tier-2 RGD validation on the icosphere.
%
% Run from the repo root with:
%   matlab -batch "cd('validation'); run_matlab_validation"
% or (equivalent, explicit path):
%   "C:/Program Files/MATLAB/R2026a/bin/matlab.exe" -batch "cd('validation'); run_matlab_validation"
%
% Writes:
%   matlab_cot_laplacian_dense.csv   Tier 1 reference (MATLAB's cotLaplacian output)
%   matlab_u_alphahat_<a>.csv        Tier 2 RGD field for each alpha_hat
%   matlab_validation_log.txt        self-reported invariants

% Paths
this_dir = fileparts(mfilename('fullpath'));
rgd_dir = fullfile(this_dir, '..', 'external', 'matlab_rgd');
addpath(rgd_dir);
addpath(fullfile(rgd_dir, 'utils'));

out = fullfile(this_dir);
logf = fullfile(out, 'matlab_validation_log.txt');
fid = fopen(logf, 'w');
log = @(s) fprintf(fid, '%s\n', s);

% Run MeshClass from rgd_dir so readOff finds icosphere_sub3.off there
oldwd = cd(rgd_dir);
cleanupObj = onCleanup(@() cd(oldwd));

% --- Tier 0: operator-level dumps (for the port) ---
Mm = MeshClass('icosphere_sub3');
[W, A] = cotLaplacian(Mm);
writematrix(full(W), fullfile(out, 'matlab_cot_laplacian_dense.csv'));
writematrix(Mm.ta, fullfile(out, 'matlab_ta.csv'));
writematrix(Mm.va, fullfile(out, 'matlab_va.csv'));
writematrix(Mm.Nf, fullfile(out, 'matlab_Nf.csv'));
% G is sparse 3nf x nv; dump COO triplets for exact reconstruction on the Python side
[gi, gj, gv] = find(Mm.G);
writematrix([gi gj gv], fullfile(out, 'matlab_G_coo.csv'));
writematrix([Mm.nv; Mm.nf], fullfile(out, 'matlab_nv_nf.csv'));

sym_err  = full(max(max(abs(W - W.'))));
rowsum   = full(max(abs(sum(W, 2))));
nvnf = sprintf('nv=%d  nf=%d', Mm.nv, Mm.nf);
fprintf('%s\n', nvnf); log(nvnf);
s1 = sprintf('Tier 0: cot Laplacian sym residual = %.3e', sym_err);
s2 = sprintf('Tier 0: cot Laplacian max|row-sum|  = %.3e', rowsum);
fprintf('%s\n%s\n', s1, s2); log(s1); log(s2);
s3 = sprintf('Tier 0: dumped ta, va, Nf, G (COO, nnz=%d)', nnz(Mm.G));
fprintf('%s\n', s3); log(s3);

% --- Tier 2: RGD-ADMM at a sweep of alpha_hat ---
alphas = [0.001, 0.01, 0.05, 0.25, 1.0];
src = 1;   % MATLAB 1-indexed, matches Python vertex 0

for k = 1:length(alphas)
    a = alphas(k);
    u = rdg_ADMM(Mm, src, 'alpha_hat', a);

    suffix = strrep(sprintf('%g', a), '.', 'p');
    fname = fullfile(out, sprintf('matlab_u_alphahat_%s.csv', suffix));
    writematrix(u, fname);

    u_src = u(src);
    u_min = min(u);
    u_max = max(u);
    [~, argmax] = max(u);
    line = sprintf('alpha_hat=%.4g  u(src)=%.6e  min(u)=%.6e  max(u)=%.6e  argmax=%d', ...
        a, u_src, u_min, u_max, argmax);
    fprintf('%s\n', line); log(line);
end

fclose(fid);
disp('validation complete. outputs in validation/');
