% Run MATLAB smooth_vf on icosphere_sub3 with the Python-generated sparse input.
this_dir = fileparts(mfilename('fullpath'));
rgd_dir = fullfile(this_dir, '..', 'external', 'matlab_rgd');
addpath(rgd_dir);
addpath(fullfile(rgd_dir, 'utils'));

oldwd = cd(rgd_dir);
cleanupObj = onCleanup(@() cd(oldwd));

Mm = MeshClass('icosphere_sub3');
vf = readmatrix(fullfile(this_dir, 'smooth_vf_input.csv'));
fprintf('input vf: %dx%d  nonzero rows=%d\n', size(vf,1), size(vf,2), ...
    sum(MeshClass.normv(vf) > 1e-5));

w = smooth_vf(Mm, vf, 2);
fprintf('output w: %dx%d  max|norm-1|=%.3e\n', size(w,1), size(w,2), ...
    max(abs(vecnorm(w, 2, 2) - 1)));

writematrix(w, fullfile(this_dir, 'matlab_smooth_vf_output.csv'));
disp('wrote matlab_smooth_vf_output.csv');
