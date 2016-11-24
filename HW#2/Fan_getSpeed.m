function s = Fan_getSpeed(t , c, resolution)

if nargin < 2
    error('test:Few arguments. This function needs 2 or 3 arguments.');
end
if nargin > 3
    error('test:Too many arguments. This function needs 2 or 3 arguments.');
end
if nargin == 2
    resolution = 100;
end
if length(t) ~= 2 || length(c) ~= 2
    error('test:Wrong input forma. t and c size must be equal to 2.');
end
if length(t) ~= length(c)
    error('test:Unbalanced arguments. t and c must have same size.');
end

c1 = t(1);
min_1 = c1 - t(2) / 2.0;
max_1 = c1 + t(2) / 2.0;

c2 = c(1);
min_2 = c2 - c(2) / 2.0;
max_2 = c2 + c(2) / 2.0;

sys = newfis('Cooler');

sys.input(1).name = 'Temperature';
sys.input(1).range = [0 50];

sys.input(1).mf(1).name = 'Cold';
sys.input(1).mf(1).type = 'trapmf';
sys.input(1).mf(1).params = [-inf -inf 5 15];

sys.input(1).mf(2).name = 'Cool';
sys.input(1).mf(2).type = 'trimf';
sys.input(1).mf(2).params = [5 15 25];

sys.input(1).mf(3).name = 'Medium';
sys.input(1).mf(3).type = 'trimf';
sys.input(1).mf(3).params = [15 25 35];

sys.input(1).mf(4).name = 'Warm';
sys.input(1).mf(4).type = 'trimf';
sys.input(1).mf(4).params = [25 35 45];

sys.input(1).mf(5).name = 'Hot';
sys.input(1).mf(5).type = 'trapmf';
sys.input(1).mf(5).params = [35 45 inf inf];

sys.input(2).name = 'Changes';
sys.input(2).range = [-50 50];

sys.input(2).mf(1).name = 'BigNegative';
sys.input(2).mf(1).type = 'trapmf';
sys.input(2).mf(1).params = [-Inf -inf -40 -25];

sys.input(2).mf(2).name = 'MediumNegative';
sys.input(2).mf(2).type = 'trimf';
sys.input(2).mf(2).params = [-40 -25 0];

sys.input(2).mf(3).name = 'ApproximatelyZero';
sys.input(2).mf(3).type = 'trimf';
sys.input(2).mf(3).params = [-5 0 5];

sys.input(2).mf(4).name = 'MediumPositive';
sys.input(2).mf(4).type = 'trimf';
sys.input(2).mf(4).params = [0 25 40];

sys.input(2).mf(5).name = 'BigPositive';
sys.input(2).mf(5).type = 'trapmf';
sys.input(2).mf(5).params = [25 40 Inf Inf];

sys.output(1).name = 'Speed';
sys.output(1).range = [0 100];

sys.output(1).mf(1).name = 'VerySlow';
sys.output(1).mf(1).type = 'trapmf';
sys.output(1).mf(1).params = [-Inf -Inf 5 25];

sys.output(1).mf(2).name = 'Slow';
sys.output(1).mf(2).type = 'trimf';
sys.output(1).mf(2).params = [5 25 50];

sys.output(1).mf(3).name = 'Medium';
sys.output(1).mf(3).type = 'trimf';
sys.output(1).mf(3).params = [25 50 75];

sys.output(1).mf(4).name = 'Fast';
sys.output(1).mf(4).type = 'trimf';
sys.output(1).mf(4).params = [50 75 95];

sys.output(1).mf(5).name = 'VeryFast';
sys.output(1).mf(5).type = 'trapmf';
sys.output(1).mf(5).params = [75 95 Inf Inf];

ruleList = [
    1 1 1 1 1;
    1 2 2 1 1;
    2 1 1 1 1;
    2 5 3 1 1;
    3 2 3 1 1;
    3 4 3 1 1;
    4 1 3 1 1;
    4 5 5 1 1;
    5 4 4 1 1;
    5 5 5 1 1;
];
sys = addrule(sys, ruleList);
rules = showrule(sys)

X = linspace( min_1, max_1, resolution );
% A = trimf( X, [min_1, c1, max_1]);

Y = linspace( min_2, max_2, resolution);
% B = trimf( Y, [min_2, c2, max_2]);

ANS = evalfis([X' Y'], sys, resolution);
s = sum(ANS) / length(ANS);

end
