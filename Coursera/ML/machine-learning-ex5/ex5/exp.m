load ('ex5data1.mat');

m = size(X, 1);

printf('size of X\n');
disp(m);
disp(size(X));
printf('\nX\n');
disp(X);
theta = [1 ; 1];
printf('\ntheta\n');
disp(theta);
printf('\n[ones(m, 1) X]\n');
disp([ones(m, 1) X]);
printf('\ny\n');
disp(y);
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
