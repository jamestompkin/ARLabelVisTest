% Probe: is curved_hessian (Stein 2020 MEX) available? Is MeshClass.godf defined?
this_dir = fileparts(mfilename('fullpath'));
rgd_dir = fullfile(this_dir, '..', 'external', 'matlab_rgd');
addpath(rgd_dir);
addpath(fullfile(rgd_dir, 'utils'));

fprintf('curved_hessian exists flag: %d\n', exist('curved_hessian'));
fprintf('  (2=M-file, 3=MEX, 5=built-in, 0=not found)\n');

fprintf('lsqlin exists flag: %d  (Optimization Toolbox)\n', exist('lsqlin'));
fprintf('eigs   exists flag: %d\n', exist('eigs'));

Mm = MeshClass('icosphere_sub3');
try
    ff = Mm.godf(2);
    fprintf('MeshClass.godf(2) ok, size=%dx%d, nnz=%d\n', size(ff,1), size(ff,2), nnz(ff));
catch ME
    fprintf('MeshClass.godf FAILED: %s\n', ME.message);
end

try
    fprintf('MeshClass.EB size  = %dx%d, nnz=%d\n', size(Mm.EB,1), size(Mm.EB,2), nnz(Mm.EB));
    fprintf('MeshClass.EBI size = %dx%d, nnz=%d\n', size(Mm.EBI,1), size(Mm.EBI,2), nnz(Mm.EBI));
catch ME
    fprintf('EB/EBI access FAILED: %s\n', ME.message);
end
