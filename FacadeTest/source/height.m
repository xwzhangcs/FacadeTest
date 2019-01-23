%clear
%data = xlsread('height_info.xlsx');
nbins = 500;
h = histogram(data, nbins);
count = h.Values;
bins = h.BinEdges;