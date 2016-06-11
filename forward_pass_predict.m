function [ prediction ] = forward_pass_predict( X, W1, b1, W2, b2 )

u1 = X * W1;
u2 = u1 + b1;
H1 = max(0, u2 );
u3 = H1 * W2;
s = u3 + b2;

prediction = s;
end

