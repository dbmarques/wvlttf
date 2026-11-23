% WVLTTF() - Wavelet time-frequency transform. Based on EEGLAB's newtimef.
%       Performs complex morlet wavelet convolution. 
% 
%   Usage
%       [tf,fc,tfri,w] = wvlttf(data,Fs,fc,cyclesperfreq,method)
% 
%   Inputs 
%       data = lfp data (trials,samples)
%       Fs = sampling frequency
%       fc = frequencies to estimate. If single, calculate only that
%                       single frequency. If [f(1) f(end)], use the
%                       frequency limits spaced by next higher power of 2
%                       from 0 to Nyquist. If vector, calculates for each
%                       frequency. default: [0 Fs/2]
%       cyclesperfreq = wavelet cycles per fequency. If integer uses the
%                       same number for every frequency. If [f(1) f(end)]
%                       uses linearly spaced numbers of cycles from slower
%                       to highest frequencies. If zero, performs 'sinus'
%                       method. default: [3]
%       method = wavelet generation method: {'newtimef', 'cmorwavf',
%                       'mikexcohen','sinus'}. default: ['newtimef']
%                       newtimef: identical to 'dftfilt3' script from
%                           eeglab used in newtimef.
%                       cmorwavf: identical to 'cmorwavf' built-in script
%                           from matlab. In this case, cyclesperfreq
%                           constitutes 'fb' time-decay parameter.
%                       mikexcohen: identical to the script presented by
%                           Mike X Cohen (see References), which only the
%                           constant differs from dftfilt3.
%                       sinus: sine waves with windowing. If cyclesperfreq
%                           is zero, performs this method using hann
%                           windowing. If cyclesperfreq is vector,
%                           performs gaussian windowing with cyclesperfreq
%                           being equivalent to alpha. Adapted from dftfilt3. 
%               
%   Output 
%       tf = time-frequency decomposition
%       fc = frequency vector
%       tfri = time-frequency decomposition in complex
%       w = cell array of wavelets per fc
% 
%   Examples
%       [tf] = wvlttf(data,Fs,[0 12],[3 20],'newtimef');
%       [tf] = wvlttf(data,Fs,[0 12],[1],'cmorwavf');
%       [tf] = wvlttf(data,Fs,[0 12],[3 20],'mikexcohen');
%       [tf] = wvlttf(data,Fs,[0 12],[0],'sinus');
% 
% Referências:
%   EEGLAB
%   https://www.youtube.com/watch?v=fkQGnYUv-FI  
%   For more info check scripts: 
%       dftfilt3 (line: 163, EEGLAB).
% 
% Author: Danilo Benette Marques, 2022
% Last update: 2026-11-23

function [tf,fc,tfri,w] = wvlttf(data,Fs,fc,cyclesperfreq,method)

verbose = true;

%method 
if nargin<5
    method = 'newtimef';
elseif nargin == 5 &...
        ~strcmp(method,'newtimef') &...
        ~strcmp(method,'cmorwavf') &...
        ~strcmp(method,'mikexcohen') &...
        ~strcmp(method,'sinus')
    error('Method should be: newtimef, cmorwavf, mikexcohen or sinus')
end

%General params
lb = -4; 
ub = 4; 

N = (ub-lb)*Fs;
dt = 1/Fs;
df = Fs/2^nextpow2(N);

% t = linspace(lb,ub,N); %time (samples)
t = [lb:dt:ub]; N = length(t); %time (samples)

NtfSD = 7;

%frequency vector
if nargin<3 | isempty(fc)
    fc = 0:df:Fs/2;
    cyclesperfreq = 3;
end

if nargin<4 | isempty(cyclesperfreq)
    cyclesperfreq = 3;
end

if numel(fc)==2
    fc = fc(1):df:fc(end);
end

%cyclesperfreq vector
if cyclesperfreq == 0
    method = 'sinus';
else
    cyclesperfreq = linspace(cyclesperfreq(1),cyclesperfreq(end),length(fc));
end

%Allocation
tf = nan(length(fc),size(data,2),size(data,1));
tfri = nan(length(fc),size(data,2),size(data,1));
w = cell(0);

% Computations per frequency
for fi = 1:length(fc)

if verbose; fprintf('Computing wavelet transform for frequency: %.1f \n',fc(fi)); end

% Define wavelet time based on N standard deviations
autotwvlt = true; %0 = don't, 1 == auto
if autotwvlt
    if fc(fi)>0 & cyclesperfreq~=0
        sigf=fc(fi)/cyclesperfreq(fi); 
        sigt=1./(2*pi*sigf);

        lb = -sigt*NtfSD/2;
        ub = sigt*NtfSD/2;

        tneg=[-dt:-dt:-sigt*NtfSD/2];
        tpos=[0:dt:sigt*NtfSD/2];

        t=[fliplr(tneg) tpos];
        N = length(t);
    end
 end

% Wavelet
switch method
    case 'newtimef'
        % script dftfilt3 default from eeglab used in newtimef
        sigf=fc(fi)/cyclesperfreq(fi); 
        sigt=1./(2*pi*sigf);
        A=1./sqrt(sigt*sqrt(pi));
        
        psi=A.*(exp(-(t.^2)./(2*(sigt^2))).*exp(2*1i*pi*fc(fi)*t));
        
    case 'cmorwavf'
        % script cmorwavf built-in do Matlab, substitutes cyclesperfreq for 'fb' (time-decay param)
        fb = cyclesperfreq(fi); 
        psi = (1/sqrt(pi*fb)) * exp(2*pi*1i*fc(fi)*t) .* exp(-(t.^2)/fb); 
        
    case 'mikexcohen'
        % described by Mike X Cohen at: https://www.youtube.com/watch?v=fkQGnYUv-FI  
        psi = exp(1i*2*pi*fc(fi)*t) .* exp(-(t.^2)./(2*(cyclesperfreq(fi)/(2*pi*fc(fi)))^2));
        
    case 'sinus'
        % sines waves with Hann windowing
        if cyclesperfreq == 0
            psi = exp(2*1i*pi*fc(fi)*t) .* hann(N)'; %hann window
        else
            psi = exp(2*1i*pi*fc(fi)*t) .* gausswin(N,cyclesperfreq(fi))'; %gaussian window
        end
end
psi = psi ./ sqrt(sum(abs(psi).^2));  %normalize by wavelet energy
w{fi} = psi;

% figure,
% hold on,plot(t,real(psi))
% hold on,plot(t,imag(psi))

%Computations per trial
for trial = 1:size(data,1)
    [tf_aux] = conv(data(trial,:),psi,'same');
    
    tfri(fi,:,trial) = tf_aux; %complex
    tf(fi,:,trial) = tf_aux.*conj(tf_aux); %conjugate
end
tf(isnan(tf)) = abs(tf(isnan(tf))); %remove complex NaN+NaNi

if verbose; clc; end

end
