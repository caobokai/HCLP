classdef ExpDatasetSYN < ExpDataset
    
    methods
        function s = ExpDatasetSYN()
            s = s@ExpDataset('Synthetic', 'HIN');
        end
        
        function [train_data, train_label, test_data, test_label, ...
                schema] = load(varargin)
            num_node_type = randi([3, 10]);
            schema.typegraph = zeros(num_node_type);
            dim_per_type = [];
            for p = 1 : num_node_type
                dim_per_type = [dim_per_type randi([100, 1000])];
            end
            num_link_type = 0;
            for p = 1 : num_node_type
                for q = p : num_node_type
                    dice = rand(1);
                    if dice > 0.6
                        num_link_type = num_link_type + 1;
                        schema.typegraph(p, q) = 1;
                        schema.links{p, q} = ...
                            rand(dim_per_type(p), dim_per_type(q)) > 0.8;
                    else
                        schema.links{p, q} = [];
                    end
                end
            end
            fprintf('===== #node: %d, #link: %d =====\n', ...
                num_node_type, num_link_type);
            
            rng(5489, 'twister');
            num_fold = 10;
            for p = 1 : size(schema.typegraph, 1)
                for q = p : size(schema.typegraph, 1)
                    if schema.typegraph(p, q)
                        matrix = schema.links{p, q};
                        num = size(matrix, 1);
                        positive = find(matrix ~= 0);
                        negative = find(matrix == 0);
                        n = length(positive);
                        negative = negative(randperm(length(negative), n));
                        m = floor(n / num_fold);
                        train = randperm(n, m);
                        a = setdiff(1 : n, train);
                        b = randperm(n - m, m);
                        test = a(b);
                        
                        train_data.source{p, q} = zeros(2 * m, 1);
                        train_data.target{p, q} = zeros(2 * m, 1);
                        train_label.value{p, q} = zeros(2 * m, 1);
                        test_data.source{p, q} = zeros(2 * m, 1);
                        test_data.target{p, q} = zeros(2 * m, 1);
                        test_label.value{p, q} = zeros(2 * m, 1);
                        
                        index = positive(train);
                        train_data.source{p, q}(1 : m) = ...
                            1 + mod(index - 1, num);
                        train_data.target{p, q}(1 : m) = ...
                            1 + floor((index - 1) / num);
                        train_label.value{p, q}(1 : m) = ones(m, 1);
                        index = negative(train);
                        train_data.source{p, q}(m + 1 : end) = ...
                            1 + mod(index - 1, num);
                        train_data.target{p, q}(m + 1 : end) = ...
                            1 + floor((index - 1) / num);
                        train_label.value{p, q}(m + 1 : end) = zeros(m, 1);
                        index = positive(test);
                        test_data.source{p, q}(1 : m) = ...
                            1 + mod(index - 1, num);
                        test_data.target{p, q}(1 : m) = ...
                            1 + floor((index - 1) / num);
                        test_label.value{p, q}(1 : m) = ones(m, 1);
                        index = negative(test);
                        test_data.source{p, q}(m + 1 : end) = ...
                            1 + mod(index - 1, num);
                        test_data.target{p, q}(m + 1 : end) = ...
                            1 + floor((index - 1) / num);
                        test_label.value{p, q}(m + 1 : end) = zeros(m, 1);
                        
                        for k = 1 : m
                            schema.links{p, q}(...
                                train_data.source{p, q}(k), ...
                                train_data.target{p, q}(k)) = 0;
                            schema.links{p, q}(...
                                test_data.source{p, q}(k), ...
                                test_data.target{p, q}(k)) = 0;
                        end;
                    end;
                end;
            end;
            return;
        end
    end
end