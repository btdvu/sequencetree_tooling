%% To write RF pulse waveform in .mrp file, which can be imported into SequenceTree
% Author: Cheng Li@LSNI
% Date: Aug. 12, 2010
% function usage
% inputs: 
%******* fname - name of .mrp file
%******* rf - complex RF pulse waveform data (in uT)
%******* timestep - sampling spacing (in ms)
%******* rephase_time - time needed for transversal magnetization to be rephased (isodelay time)
%                       which is usually half of the RF duration for excitation pulse.
%                       For saturation pulse, it should be 0 (in ms).
%******* bandwidth - RF pulse bandwidth (in Hz). Note that this bandwidth is only a relative quantity
%                    and it is used to control the RF pulse duration.
function write_rf(fname,rf,timestep,rephase_time,bandwidth)
fname = strcat(fname,'.mrp');
rrf = real(rf);
irf = imag(rf);
F = fopen(fname,'wt');
fprintf(F,'%s\n','[General]');
fprintf(F,'%s','data_real=');
for indx = 1:length(rf)
    if indx ~= length(rf)
        fprintf(F,'%f, ',rrf(indx));
    else
        fprintf(F,'%f\n',rrf(indx));
    end
end
fprintf(F,'%s','data_imag=');
for indx = 1:length(rf)
    if indx ~= length(rf)
        fprintf(F,'%f, ',irf(indx));
    else
        fprintf(F,'%f\n',irf(indx));
    end
end
fprintf(F,'%s','timestep=');
fprintf(F,'%f\n',timestep);
fprintf(F,'%s','rephase_time=');
fprintf(F,'%f\n',rephase_time);
fprintf(F,'%s','bandwidth=');
fprintf(F,'%f\n',bandwidth);
fclose(F);