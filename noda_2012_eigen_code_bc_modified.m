%%
clc
close all
clear 

nm = 10^(-9);
a = 295*nm;
B0 = (2*pi/a)*nm;
%d = 118*nm;
lambda0 = 960*nm;
k0 = 2*pi/lambda0;
ff = 0.16; %filling factor
eps0 = 1; %air
eps_GaAs = 12.7449; 

eps1 = 11.0224; %p-clad AlGaAs
eps2 = 12.7449; %GaAs
eps3 = ff*eps0+(1-ff)*eps_GaAs; %PC
eps4 = 12.8603; %active
eps5 = 11.0224; %n-clad AlGaAs

d1 = 1500*nm; %p-clad AlGaAs
d2 = d1+59*nm; %GaAs
d3 = d2 + 118*nm; %PC
d4 = d3 + 88.5*nm; %active
d5 = d4 + 1500*nm; %n-clad AlGaAs

%%
% Solving theta
% options = optimoptions(@fsolve,'Algorithm','levenberg-marquardt','plotfcn','optimplotfval','FunctionTolerance',1e-15);
% t11 = @(B) (exp(d1*sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps1))*((exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2)*(d1 - d2))*((exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)*(d2 - d3))*exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)*(d3 - d4))*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4) + 1)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps5) - 1))/4 + (exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)*(d2 - d3))*exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)*(d3 - d4))*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4) - 1)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps5) + 1))/4)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3) + 1))/2 + (exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2)*(d1 - d2))*((exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)*(d2 - d3))*exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)*(d3 - d4))*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4) - 1)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps5) - 1))/4 + (exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)*(d2 - d3))*exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)*(d3 - d4))*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4) + 1)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps5) + 1))/4)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3) - 1))/2)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps1)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2) - 1))/2 + (exp(d1*sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps1))*((exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2)*(d1 - d2))*((exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)*(d2 - d3))*exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)*(d3 - d4))*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4) + 1)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps5) - 1))/4 + (exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)*(d2 - d3))*exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)*(d3 - d4))*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4) - 1)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps5) + 1))/4)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3) - 1))/2 + (exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2)*(d1 - d2))*((exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)*(d2 - d3))*exp(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)*(d3 - d4))*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4) - 1)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps5) - 1))/4 + (exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)*(d2 - d3))*exp(-sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)*(d3 - d4))*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4) + 1)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps4)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps5) + 1))/4)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps3) + 1))/2)*(sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps1)/sqrt(((B(1)+1i*B(2))/nm)^2-k0^2*eps2) + 1))/2;
% 
% B_solved = fsolve(t11,[1.2*B0,0],options);
% 
%B = (B_solved(1) + 1i*B_solved(2))/nm;
B = 0.0220535527629603/nm; %B for ff = 0.16
%B = 0.0222299999/nm; %B for ff = 0.10
%B = 0.02195768/nm; %B for ff = 0.2
B0 = (2*pi)/a;

gamma1 = sqrt(B^2-k0^2*eps1);
gamma2 = sqrt(B^2-k0^2*eps2);
gamma3 = sqrt(B^2-k0^2*eps3);
gamma4 = sqrt(B^2-k0^2*eps4);
gamma5 = sqrt(B^2-k0^2*eps5);

A1 = 1; 
B1 = 0;
A2 = (A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2;
B2 = (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2 - (A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2;
A3 = (exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 - 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 + 1))/2;
B3 = - (exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 + 1))/2 - (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 - 1))/2;
A4 = (exp(gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 + 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 - 1))/2)*(gamma3/gamma4 - 1))/2 + (exp(-gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 - 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 + 1))/2)*(gamma3/gamma4 + 1))/2;
B4 = - (exp(gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 + 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 - 1))/2)*(gamma3/gamma4 + 1))/2 - (exp(-gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 - 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 + 1))/2)*(gamma3/gamma4 - 1))/2;
%A5 = (exp(gamma4*(d3 - d4))*((exp(gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 + 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 - 1))/2)*(gamma3/gamma4 + 1))/2 + (exp(-gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 - 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 + 1))/2)*(gamma3/gamma4 - 1))/2)*(gamma4/gamma5 - 1))/2 + (exp(-gamma4*(d3 - d4))*((exp(gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 + 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 - 1))/2)*(gamma3/gamma4 - 1))/2 + (exp(-gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 - 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 + 1))/2)*(gamma3/gamma4 + 1))/2)*(gamma4/gamma5 + 1))/2;
B5 = - (exp(gamma4*(d3 - d4))*((exp(gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 + 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 - 1))/2)*(gamma3/gamma4 + 1))/2 + (exp(-gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 - 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 + 1))/2)*(gamma3/gamma4 - 1))/2)*(gamma4/gamma5 + 1))/2 - (exp(-gamma4*(d3 - d4))*((exp(gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 + 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 - 1))/2)*(gamma3/gamma4 - 1))/2 + (exp(-gamma3*(d2 - d3))*((exp(gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 - 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 + 1))/2)*(gamma2/gamma3 - 1))/2 + (exp(-gamma2*(d1 - d2))*((A1*exp(d1*gamma1)*(gamma1/gamma2 + 1))/2 - (B1*exp(-d1*gamma1)*(gamma1/gamma2 - 1))/2)*(gamma2/gamma3 + 1))/2)*(gamma3/gamma4 + 1))/2)*(gamma4/gamma5 - 1))/2;
A5 = 0;

z1 = 0:0.5*nm:d1;
z2 = d1:0.5*nm:d2;
z3 = d2:0.5*nm:d3;
z4 = d3:0.5*nm:d4;
z5 = d4:0.5*nm:d5;

for j=1:length(z1)
    theta1(j) = A1*exp(gamma1*(z1(j)))+B1*exp(-gamma1*(z1(j)));
    theta(j) = theta1(j);
end
j=j+1;
for k = 1:length(z2)
    theta2(k) = A2*exp(gamma2*(z2(k)-d1))+B2*exp(-gamma2*(z2(k)-d1));
    theta(j) = theta2(k);
    j=j+1;
end
% 
for p = 1:length(z3)
    theta3(p) = A3*exp(gamma3*(z3(p)-d2))+B3*exp(-gamma3*(z3(p)-d2));
    theta(j) = theta3(p);
    j=j+1;
end
for p = 1:length(z4)
    theta4(p) = A4*exp(gamma4*(z4(p)-d3))+B4*exp(-gamma4*(z4(p)-d3));
    theta(j) = theta4(p);
    j=j+1;
end
for p = 1:length(z5)
    theta5(p) = A5*exp(gamma5*(z5(p)-d4))+B5*exp(-gamma5*(z5(p)-d4));
    theta(j) = theta5(p);
    j=j+1;
end

sum_sq_theta = sum(abs(theta).^2);
theta_norm = theta./sqrt(sum_sq_theta);
theta1_norm = theta1./sqrt(sum_sq_theta);
theta2_norm = theta2./sqrt(sum_sq_theta);
theta3_norm = theta3./sqrt(sum_sq_theta);
theta4_norm = theta4./sqrt(sum_sq_theta);
theta5_norm = theta5./sqrt(sum_sq_theta);


figure(2)
plot(z1/nm,abs(theta1_norm),'LineWidth',4)
hold on
plot(z2/nm,abs(theta2_norm),'LineWidth',4)
hold on
plot(z3/nm,abs(theta3_norm),'LineWidth',4)
hold on
plot(z4/nm,abs(theta4_norm),'LineWidth',4)
hold on
plot(z5/nm,abs(theta5_norm),'LineWidth',4)
xlabel('z')
ylabel('theta(z)')
set(gca,'linewidth',3)
set(gca,'fontsize',32)

%%

r = sqrt((ff*a^2)/pi); %radius of the air hole
x = -a/2:0.5*nm:a/2; 
y = x;
[X, Y] = meshgrid(x,y);
G = ones(length(x));

G(X.^2 + Y.^2 > r^2) = sqrt(eps_GaAs); %refractive index distribution

%figure(3)
%imagesc(x,y,G)

fG = fft2(G.^2)/length(x)^2; %taking fourier transfrom

%figure(4)
%imagesc(abs(fftshift(fG)))
fG(1,1)   %zeta_00
% fG(1,3)   %zeta_02
% fG(3,1)   %zeta_20
% fG(1,2)   %zeta_01
% fG(2,1)   %zeta_10
% fG(2,3)   %zeta_12
% fG(3,2)   %zeta_21

%%
% 1D coupling coefficients
zeta_20 = fG(3,1);
zeta_02 = fG(1,3);
theta3_n_sum = 0.5*nm*sum(abs(theta3_norm).^2); %theta in the PC layer

k_20 = - (k0^2/(2*B0))*zeta_20*theta3_n_sum;
k_02 = - (k0^2/(2*B0))*zeta_02*theta3_n_sum;
C1D = [0 k_20 0 0; k_20 0 0 0; 0 0 0 k_02; 0 0 k_02 0]; 

%%
% radiative coupling coefficients
zeta_10 = fG(2,1);
zeta_01 = fG(1,2);

Bz = k0*sqrt(eps3); %eps of PC region
k=1;
for i=1:length(z3)
    for j = 1:length(z3)
        if i == j
            G_zz(k) = -1i/(2*Bz);
        else G_zz(k) = (-1i/(2*Bz))*exp(-1i*Bz*abs(z3(i)-z3(j)));
        end

        xi(k) = G_zz(k)*theta3_norm(j)*theta3_norm(i)';
        k=k+1;
    end
end

xi_10 = sum(xi)*0.5*nm*zeta_10*zeta_10*(-k0^4/(2*B0));
xi_01 = sum(xi)*0.5*nm*zeta_01*zeta_01*(-k0^4/(2*B0));

Crad = [xi_10 xi_10 0 0; xi_10 xi_10 0 0; 0 0 xi_01 xi_01; 0 0 xi_01 xi_01];

%%
% 2D coupling coefficients
X_y_p10_p10 = -(k0^2/(2*B0))*(zeta_C_jmnrs(1, 0, 1, 0, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 1, 0, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_y_p10_n10 = -(k0^2/(2*B0))*(zeta_C_jmnrs(1, 0, -1, 0, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, -1, 0, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_y_p10_0p1 = -(k0^2/(2*B0))*(zeta_C_jmnrs(1, 0, 0, 1, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, 1, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_y_p10_0n1 = -(k0^2/(2*B0))*(zeta_C_jmnrs(1, 0, 0, -1, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(1, 0, 0, -1, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));

X_y_n10_p10 = -(k0^2/(2*B0))*(zeta_C_jmnrs(-1, 0, 1, 0, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 1, 0, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_y_n10_n10 = -(k0^2/(2*B0))*(zeta_C_jmnrs(-1, 0, -1, 0, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, -1, 0, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_y_n10_0p1 = -(k0^2/(2*B0))*(zeta_C_jmnrs(-1, 0, 0, 1, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, 1, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_y_n10_0n1 = -(k0^2/(2*B0))*(zeta_C_jmnrs(-1, 0, 0, -1, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(-1, 0, 0, -1, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));

X_x_0p1_p10 = -(k0^2/(2*B0))*(zeta_C_jmnrs(0, 1, 1, 0, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 1, 0, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_x_0p1_n10 = -(k0^2/(2*B0))*(zeta_C_jmnrs(0, 1, -1, 0, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, -1, 0, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_x_0p1_0p1 = -(k0^2/(2*B0))*(zeta_C_jmnrs(0, 1, 0, 1, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, 1, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_x_0p1_0n1 = -(k0^2/(2*B0))*(zeta_C_jmnrs(0, 1, 0, -1, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, 1, 0, -1, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));

X_x_0n1_p10 = -(k0^2/(2*B0))*(zeta_C_jmnrs(0, -1, 1, 0, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 1, 0, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_x_0n1_n10 = -(k0^2/(2*B0))*(zeta_C_jmnrs(0, -1, -1, 0, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, -1, 0, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_x_0n1_0p1 = -(k0^2/(2*B0))*(zeta_C_jmnrs(0, -1, 0, 1, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, 1, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));
X_x_0n1_0n1 = -(k0^2/(2*B0))*(zeta_C_jmnrs(0, -1, 0, -1, 1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, 1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, -1, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, -1, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, 2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, 1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, 2, -1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, 1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, -1, 2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, -2, 1, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, -1, -2, B0, k0, eps3, z3, theta3_norm, fG) + zeta_C_jmnrs(0, -1, 0, -1, -2, -1, B0, k0, eps3, z3, theta3_norm, fG));

C2D = [X_y_p10_p10 X_y_p10_n10 X_y_p10_0p1 X_y_p10_0n1;
       X_y_n10_p10 X_y_n10_n10 X_y_n10_0p1 X_y_n10_0n1;
       X_x_0p1_p10 X_x_0p1_n10 X_x_0p1_0p1 X_x_0p1_0n1;
       X_x_0n1_p10 X_x_0n1_n10 X_x_0n1_0p1 X_x_0n1_0n1];

% Define the matrices C, delta, and alpha
C = C1D + Crad + C2D;

%%
L = 70*10^(-6);


row = 16;
column = 16;
N = (row-1)*(column-1); %number of discrete points
del_x = L/row;
%A = eye(4*N); %derivative matrix
Rx = eye(N);
%Sx = eye(N);
Ry = eye(N);
%Sy = eye(N);

%Rx
j=1;
for i =column:N
    Rx(i,j) = -2;
    if i > 2*(column-1) 
        Rx(i,j-(column-1)) = 2;
    end
    if i > 3*(column-1)
        Rx(i,j-2*(column-1)) = -2;
    end
    if i > 4*(column-1)
        Rx(i,j-3*(column-1)) = 2;
    end
    if i > 5*(column-1)
        Rx(i,j-4*(column-1)) = -2;
    end
    if i > 6*(column-1)
        Rx(i,j-5*(column-1)) = 2;
    end
    if i > 7*(column-1)
        Rx(i,j-6*(column-1)) = -2;
    end
    if i > 8*(column-1)
        Rx(i,j-7*(column-1)) = 2;
    end
    if i > 9*(column-1)
        Rx(i,j-8*(column-1)) = -2;
    end
    if i > 10*(column-1)
        Rx(i,j-9*(column-1)) = 2;
    end
    if i > 11*(column-1)
        Rx(i,j-10*(column-1)) = -2;
    end
    if i > 12*(column-1)
        Rx(i,j-11*(column-1)) = 2;
    end
    if i > 13*(column-1)
        Rx(i,j-12*(column-1)) = -2;
    end
    if i > 14*(column-1)
        Rx(i,j-13*(column-1)) = 2;
    end
    if i > 15*(column-1)
        Rx(i,j-14*(column-1)) = -2;
    end
    if i > 16*(column-1)
        Rx(i,j-15*(column-1)) = 2;
    end
    if i > 17*(column-1)
        Rx(i,j-16*(column-1)) = -2;
    end
    if i > 18*(column-1)
        Rx(i,j-17*(column-1)) = 2;
    end
    if i > 19*(column-1)
        Rx(i,j-18*(column-1)) = -2;
    end
    if i > 20*(column-1)
        Rx(i,j-19*(column-1)) = 2;
    end
    j=j+1;
end

Sx = Rx';

%Ry
j=1;
i=1;
for p = 1:column-1
    for q = 1:column-1
        if q < p && mod(p+q,2) == 1
            Ry(i+p-1,i+q-1) = -2;
        elseif q<p && mod(p+q,2) == 0
            Ry(i+p-1,i+q-1) = 2;
        end
    end
end
for i = column:N
    if (mod(i,j*column-j+1)) == 0
        j=j+1;
        p=1;
        q=1;
        for p = 1:column-1
            for q = 1:column-1
                if q < p && mod(p+q,2) == 1
                    Ry(i+p-1,i+q-1) = -2;
                elseif q<p && mod(p+q,2) == 0
                    Ry(i+p-1,i+q-1) = 2;
                end
            end
        end
    end
end    


%Sy
Sy = Ry';
% A = derivative matrix
A = [Rx zeros(N) zeros(N) zeros(N);
     zeros(N) Sx zeros(N) zeros(N);
     zeros(N) zeros(N) Ry zeros(N);
     zeros(N) zeros(N) zeros(N) Sy];


C11_mat = eye(N) * C(1,1);
C12_mat = eye(N) * C(1,2);
C13_mat = eye(N) * C(1,3);
C14_mat = eye(N) * C(1,4);
C21_mat = eye(N) * C(2,1);
C22_mat = eye(N) * C(2,2);
C23_mat = eye(N) * C(2,3);
C24_mat = eye(N) * C(2,4);
C31_mat = eye(N) * C(3,1);
C32_mat = eye(N) * C(3,2);
C33_mat = eye(N) * C(3,3);
C34_mat = eye(N) * C(3,4);
C41_mat = eye(N) * C(4,1);
C42_mat = eye(N) * C(4,2);
C43_mat = eye(N) * C(4,3);
C44_mat = eye(N) * C(4,4);
    
% C = Coupling Matrix
C = [C11_mat C12_mat C13_mat C14_mat; C21_mat C22_mat C23_mat C24_mat; C31_mat C32_mat C33_mat C34_mat; C41_mat C42_mat C43_mat C44_mat];

% Compute the eigenvalues and eigenvectors of the coupled-wave equations
[V, D] = eig(0.5*C + (1i/del_x)*A);

for i = 1:length(V)
    matrixRx = sprintf('Rx_%d', i); % Create a matrix name based on the loop index
    eval([matrixRx ' = [reshape(V(1:length(V)/4,i),[row-1,column-1])]']);
    %eval([matrixName ' = [zeros(size(matrixName,1),1), matrixName]']);

    %Rx = reshape(V(1:size(V(:,1))/4,1),[row,column-1]);
    %Rx = [zeros(size(Rx,1),1), Rx];

    matrixSx = sprintf('Sx_%d', i); % Create a matrix name based on the loop index
    eval([matrixSx ' = [reshape(V(length(V)/4+1:2*length(V)/4,i),[row-1,column-1])]']);

    matrixRy = sprintf('Ry_%d', i); % Create a matrix name based on the loop index
    eval([matrixRy ' = [reshape(V(2*length(V)/4+1:3*length(V)/4,i),[row-1,column-1])]']);

    matrixSy = sprintf('Sy_%d', i); % Create a matrix name based on the loop index
    eval([matrixSy ' = [reshape(V(3*length(V)/4+1:4*length(V)/4,i),[row-1,column-1])]']);

%     figure(i) 
%     imagesc(abs(matrixRx).^2 + abs(matrixSx).^2 + abs(matrixRy).^2 + abs(matrixSy).^2)
end



% Print the eigenvalues and eigenvectors
%fprintf('Eigenvalues:\n');
%disp(diag(D));
%fprintf('Eigenvectors:\n');
%disp(V);
delta_L = L*real(2*diag(D));
alpha_L = L*imag(2*diag(D));

figure(5)
plot(delta_L,alpha_L,'ro',MarkerSize=12,LineWidth=4)
%title('Q factor for different in pane index','FontSize',28)
ylabel('alpha*L','FontSize',28)
xlabel('delta*L','FontSize',28)
set(gca,'linewidth',3)
set(gca,'fontsize',32)
grid on
xlim([-10 10])
ylim([0 4.9])

figure(6)
imagesc(abs(Rx_706)^2 + abs(Sx_706)^2 + abs(Ry_706)^2 + abs(Sy_706)^2)

figure(7)
imagesc(abs(Rx_640)^2 + abs(Sx_640)^2 + abs(Ry_640)^2 + abs(Sy_640)^2)

figure(8)
imagesc(abs(Rx_150)^2 + abs(Sx_150)^2 + abs(Ry_150)^2 + abs(Sy_150)^2)

figure(9)
imagesc(abs(Rx_650)^2 + abs(Sx_650)^2 + abs(Ry_650)^2 + abs(Sy_650)^2)

%%
function [out1] = zeta_C_jmnrs(p, q, r, s, m, n, B0, k0, eps3, z3, theta3_norm, fG)

% for 2D coupling coefficient
    Bzmn = sqrt((m^2+n^2)*B0^2-k0^2*eps3); %eps of PC 
    theta3_n_sum = 0.5*1e-9*sum(abs(theta3_norm).^2);
    
    v_p10 = -(1/eps3)*fG(abs(m-1)+1,abs(n-0)+1)*theta3_n_sum;
    v_n10 = -(1/eps3)*fG(abs(m+1)+1,abs(n-0)+1)*theta3_n_sum;
    v_0p1 = -(1/eps3)*fG(abs(m-0)+1,abs(n-1)+1)*theta3_n_sum;
    v_0n1 = -(1/eps3)*fG(abs(m-0)+1,abs(n+1)+1)*theta3_n_sum;
    
    k=1;
    for i=1:length(z3)
        for j = 1:length(z3)
            if i == j
                G_zz(k) = 1/(2*Bzmn);
            else G_zz(k) = (1/(2*Bzmn))*exp(-Bzmn*abs(z3(i)-z3(j)));
            end
    
            mu(k) = G_zz(k)*theta3_norm(j)*theta3_norm(i)';
    
            k=k+1;
        end
    end
    
    mu_p10 = k0^2*fG(abs(m-1)+1,abs(n-0)+1)*sum(mu)*0.5*1e-9;
    mu_n10 = k0^2*fG(abs(m+1)+1,abs(n-0)+1)*sum(mu)*0.5*1e-9;
    mu_0p1 = k0^2*fG(abs(m-0)+1,abs(n-1)+1)*sum(mu)*0.5*1e-9;
    mu_0n1 = k0^2*fG(abs(m-0)+1,abs(n+1)+1)*sum(mu)*0.5*1e-9;
    
    MU_V = [-m*mu_p10 -m*mu_n10 n*mu_0p1 n*mu_0n1; n*v_p10 n*v_n10 m*v_0p1 m*v_0n1];
    MN = [n m; -m n];
    C_xy = (1/(m^2+n^2))*MN*MU_V;
    
    cx_p10 = C_xy(1,1);
    cx_n10 = C_xy(1,2);
    cx_0p1 = C_xy(1,3);
    cx_0n1 = C_xy(1,4);
    cy_p10 = C_xy(2,1);
    cy_n10 = C_xy(2,2);
    cy_0p1 = C_xy(2,3);
    cy_0n1 = C_xy(2,4);
    
    cx_p10_fG = fG(abs(p-m)+1,abs(q-n)+1)*cx_p10;
    cx_n10_fG = fG(abs(p-m)+1,abs(q-n)+1)*cx_n10;
    cx_0p1_fG = fG(abs(p-m)+1,abs(q-n)+1)*cx_0p1;
    cx_0n1_fG = fG(abs(p-m)+1,abs(q-n)+1)*cx_0n1;
    cy_p10_fG = fG(abs(p-m)+1,abs(q-n)+1)*cy_p10;
    cy_n10_fG = fG(abs(p-m)+1,abs(q-n)+1)*cy_n10;
    cy_0p1_fG = fG(abs(p-m)+1,abs(q-n)+1)*cy_0p1;
    cy_0n1_fG = fG(abs(p-m)+1,abs(q-n)+1)*cy_0n1;
    if q == 0 && r == 1 && s == 0
        out1 = cy_p10_fG;
    elseif q == 0 && r == -1 && s == 0
        out1 = cy_n10_fG;
    elseif q == 0 && r == 0 && s == 1
        out1 = cy_0p1_fG;
    elseif q == 0 && r == 0 && s == -1
        out1 = cy_0n1_fG;
    elseif p == 0 && r == 1 && s == 0
        out1 = cx_p10_fG;
    elseif p == 0 && r == -1 && s == 0
        out1 = cx_n10_fG;
    elseif p == 0 && r == 0 && s == 1
        out1 = cx_0p1_fG;
    elseif p == 0 && r == 0 && s == -1
        out1 = cx_0n1_fG;
    end
end
