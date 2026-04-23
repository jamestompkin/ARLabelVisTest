% Dump MATLAB's edge-based operators (F1, F2, EB, R, godf(2), edges, e2t, ea)
% for icosphere_sub3, so we can diff Python's port against them.
this_dir = fileparts(mfilename('fullpath'));
rgd_dir = fullfile(this_dir, '..', 'external', 'matlab_rgd');
addpath(rgd_dir);
addpath(fullfile(rgd_dir, 'utils'));

oldwd = cd(rgd_dir);
cleanupObj = onCleanup(@() cd(oldwd));

Mm = MeshClass('icosphere_sub3');
fprintf('nv=%d  nf=%d  ne=%d  nie=%d\n', Mm.nv, Mm.nf, Mm.ne, Mm.nie);

writematrix(Mm.F1, fullfile(this_dir, 'matlab_F1.csv'));
writematrix(Mm.F2, fullfile(this_dir, 'matlab_F2.csv'));

[ri, rj, rv] = find(Mm.R);
writematrix([ri rj rv], fullfile(this_dir, 'matlab_R_coo.csv'));

[ei, ej, ev] = find(Mm.EB);
writematrix([ei ej ev], fullfile(this_dir, 'matlab_EB_coo.csv'));

writematrix(Mm.edges,       fullfile(this_dir, 'matlab_edges.csv'));
writematrix(Mm.e2t,         fullfile(this_dir, 'matlab_e2t.csv'));
writematrix(Mm.inner_edges, fullfile(this_dir, 'matlab_inner_edges.csv'));
writematrix(Mm.ea,          fullfile(this_dir, 'matlab_ea.csv'));

[op, oph] = Mm.godf(2);
[oi, oj, ov] = find(op);
writematrix([oi oj ov], fullfile(this_dir, 'matlab_godf2_coo.csv'));

fprintf('dumped edge ops\n');
