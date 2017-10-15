function Rf = ampRfNew(M,r,h,Rf0,Rfs,cd,rs)

% function Rf = ampRf(M,r,h,Rf0,Rfs,Q0,Qs,cd,ch)
% M = magnitude
% r = distance [km]
% h = depth [km]
%
% Rf0 = Rf amplitude parameter
% Rfs = exponent of power law for f-dependent Rf amplitude
% Q0 = Q-value of Earth for Rf waves at 1Hz
% Qs = exponent of power law for f-dependent Q
% cd = speed parameter for surface coupling  [km/s]
% ch = speed parameter for horizontal propagation  [km/s]
%
% exp(-2*pi*h.*fc./cd), coupling of source to surface waves
% exp(-2*pi*r.*fc./ch./Q), dissipation

fc = 10.^(2.3-M/2);
Af = Rf0./fc.^Rfs;

% Rf = Rf0.*(M./fc).*exp(-2*pi*h.*fc/c).*exp(-2*pi*r.*fc/c./Q)./r*1e-3;
% Rf = M.*Af.*exp(-2*pi*h.*fc./cd).*exp(-2*pi*r.*fc./ch./Q)./r.^rs*1e-3;
Rf = M.*Af.*exp(-2*pi*h.*fc./cd)./r.^rs*1e-3;

