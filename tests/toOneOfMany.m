function target = toOneOfMany(classes, nclasses, permutation, from_zero)

if nargin < 2
    error('classes and nClasses must be specified!');
end

% set the default permutation to 0.01
if nargin == 2
    permutation = 0.01;
end

if nargin < 4
    from_zero = 1;
end

% convert the array 'classes' to a row vector.
[m, n] = size(classes);
if m < n
    classes = classes';
    m = size(classes, 1);
end

% make sure the classes are labeled from 1.
if from_zero
    classes = classes + 1;
end

% convert the classes to 1-of-K style.
target = zeros(m, nclasses);
v = 1.0 - double(nclasses) * permutation;
for i = 1 : m
    target(i, :) = permutation;
    target(i, classes(i)) = target(i, classes(i)) + v;
end

end