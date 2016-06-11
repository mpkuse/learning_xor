function [u1 u2 H1 u3 s, L ] = forward_pass( X, W1, b1, W2, b2, y )

u1 = X * W1;
u2 = u1 + b1;
H1 = max(0, u2 ); %Element-wise max, (Also refered as ReLU).  
u3 = H1 * W2;
s = u3 + b2;
L = (y - s)^2; % Squared-Loss function

end

