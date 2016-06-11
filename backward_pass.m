function [ dL_db2 dL_dW2 dL_db1 dL_dW1] = backward_pass( X, W1, b1, W2, b2, y, u1, u2, H1, u3, u4, L )

dL_dy = 2*(y-u4); %scalar
dL_du4 = -2*(y-u4); %scalar

du4_db2 = 1.0; %scalar
du4_du3 = 1.0; %scalar

du3_dW2 = H1; %1x2
du3_dH1 = W2'; %1x2

% if u2_i > 0 ==> 1. if u2_i < 0 ==> 0
dH1_du2 = diag(max( 0, u2 ) > 0); %2x2

du2_db1 = eye(2,2); %2x2
du2_du1 = eye(2,2); %2x2



% finals
dL_db2 = dL_du4 * du4_db2;
dL_dW2 = dL_du4 * du4_du3 * du3_dW2;

dL_db1 = dL_du4 * du4_du3 * du3_dH1 * dH1_du2 * du2_db1;

dL_dW1 = X' * dL_du4 * du4_du3 * du3_dH1 * dH1_du2 * du2_du1;

end

