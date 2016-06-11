% From Deep Learning Book by Yoshua Bengio
% learn the xor function from itz for inputs
% network has 1 hidden layer. gradients computed with back-propagation
% (0,0) --> 0
% (0,1) --> 1
% (1,0) --> 1
% (1,1) --> 0
clear all;
% init
W1 = [0.5 1.5 ; 1.7 0.6 ];
b1 = [ 0.1, -1.1 ];
W2 = [1.1; -2.2 ];
b2 = 0.1;


% Data
X = [ 0 0 ; 0 1 ; 1 0 ; 1 1 ];
y = [  0  ;  1  ;  1  ;  0  ];




step = 0.03;
for itr = 1:300
    % computes the derivative of cost function wrt to each weights. sum for
    % every data
    all_dL_dW1 = zeros( 2,2 );
    all_dL_db1 = zeros( 1,2 );
    all_dL_dW2 = zeros( 2,1 );
    all_dL_db2 = 0;
    all_cost = 0;
    for data_indx = 1:4
        [u1, u2, H1, u3, u4, L ] = forward_pass( X(data_indx,:), W1, b1, W2, b2, y(data_indx) );
        [dL_db2, dL_dW2, dL_db1, dL_dW1] = backward_pass( X(data_indx,:), W1, b1, W2, b2, y(data_indx), u1, u2, H1, u3, u4, L );
        
        all_dL_dW1 = all_dL_dW1 + dL_dW1;
        all_dL_db1 = all_dL_db1 + dL_db1;
        all_dL_dW2 = all_dL_dW2 + dL_dW2';
        all_dL_db2 = all_dL_db2 + dL_db2;
        all_cost = all_cost + .5 * L^2;
    end
    
    display( sprintf('%d : %f', itr, all_cost) );
    
    % gradient descent step. +lambda is derivate of the regularized part
    lambda = 0.1;
    W1 = W1 - step*(all_dL_dW1 + lambda*W1);
    b1 = b1 - step*(all_dL_db1 + lambda*b1);
    W2 = W2 - step*(all_dL_dW2 + lambda*W2);
    b2 = b2 - step*(all_dL_db2 + lambda*b2); 
end


display( 'Learning Complete' )

display( 'Evaluate performance' );
eval_performance