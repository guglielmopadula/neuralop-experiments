lapl=load("lapl_matrix.mat");
M=lapl.M;
L=lapl.L;
issymmetric(M)
issymmetric(L)
[v,d]=eigs(-L,1024);
I=norm(imag(v))/norm(real(v))
save("v.mat","v");
