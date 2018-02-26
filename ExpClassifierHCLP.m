classdef ExpClassifierHCLP < ExpClassifier
    
    properties
        baseLearner;
        num_iter = 5;
        max_len = 3;
        max_sample = 10000;
        threshold = 0.9;
    end
    
    methods
        function s = ExpClassifierHCLP()
            s = s@ExpClassifier('HCLP', 'link prediction');
            s.baseLearner = ExpClassifierSVM();
        end
        function [outputs, pre_labels, s] = classify(...
                s, train_data, train_label, test_data, schema)
            t = cputime;
            
            cla = cell(size(schema.typegraph, 1));
            for p = 1 : size(schema.typegraph, 1)
                for q = p : size(schema.typegraph, 1)
                    if schema.typegraph(p, q)
                        matrix = schema.links{p, q};
                        path1 = getMetaPath(...
                            p, size(matrix, 1), schema, s.max_len);
                        path2 = getMetaPath(...
                            q, size(matrix, 2), schema, s.max_len);
                        fea = getFeature(...
                            train_data.source{p, q}, ...
                            train_data.target{p, q}, ...
                            path1, path2, matrix);
                        cla{p, q} = s.baseLearner;
                        cla{p, q} = cla{p, q}.train(...
                            fea', train_label.value{p, q}');
                        schema.links{p, q} = updateSchema(...
                            schema.links{p, q}, ...
                            train_data.source{p, q}, ...
                            train_data.target{p, q}, ...
                            train_label.value{p, q}, 0);
                    end;
                end;
            end;
            
            for iter = 1 : s.num_iter
                fprintf('===== train %d/%d =====\n', iter, s.num_iter);
                for p = 1 : size(schema.typegraph, 1)
                    for q = p : size(schema.typegraph, 1)
                        if schema.typegraph(p, q)
                            matrix = schema.links{p, q};
                            idx = find(matrix == 0);
                            num_sample = length(idx);
                            if num_sample > s.max_sample
                                idx = idx(randperm(num_sample, s.max_sample));
                            end;
                            source = 1 + mod(idx - 1, size(matrix, 1));
                            target = 1 + floor((idx - 1) / size(matrix, 1));
                            path1 = getMetaPath(...
                                p, size(matrix, 1), schema, s.max_len);
                            path2 = getMetaPath(...
                                q, size(matrix, 2), schema, s.max_len);
                            fea = getFeature(...
                                source, target, ...
                                path1, path2, matrix);                        
                            [outputs, ~] = cla{p, q}.test(fea');
                            schema.links{p, q} = updateSchema(...
                                schema.links{p, q}, ...
                                source, target, outputs, s.threshold);
                        end;
                    end;
                end;
            end
            
            fprintf('===== test =====\n');
            outputs = cell(size(schema.typegraph, 1));
            pre_labels = cell(size(schema.typegraph, 1));
            for p = 1 : size(schema.typegraph, 1)
                for q = p : size(schema.typegraph, 1)
                    if schema.typegraph(p, q)
                        matrix = schema.links{p, q};
                        path1 = getMetaPath(...
                            p, size(matrix, 1), schema, s.max_len);
                        path2 = getMetaPath(...
                            q, size(matrix, 2), schema, s.max_len);
                        fea = getFeature(...
                            test_data.source{p, q}, ...
                            test_data.target{p, q}, ...
                            path1, path2, matrix);                         
                        [outputs{p, q}, pre_labels{p, q}] = ...
                            cla{p, q}.test(fea');
                        schema.links{p, q} = updateSchema(...
                            schema.links{p, q}, ...
                            test_data.source{p, q}, ...
                            test_data.target{p, q}, ...
                            outputs{p, q}, s.threshold);
                    end;
                end;
            end;
            
            s.time_train = cputime - t;
            s.time = cputime - t;  
            s.time_test = s.time - s.time_train;
            
            % save running state discription
            s.abstract = [
                s.name  '(' ...
                '-time:' num2str(s.time) ...
                '-time_train:' num2str(s.time_train) ...
                '-time_test:' num2str(s.time_test) ...
                '-base:' s.baseLearner.name ')'];
        end
    end
end

function metapath = getMetaPath(node, num_size, schema, max_len)
    num_path = 1;
    metapath{1} = sparse(1 : num_size, 1 : num_size, 1, num_size, num_size);
    p = 1;
    q = 1;
    bfs_node(1) = node;
    bfs_len(1) = 0;
    bfs_mat{1} = sparse(1 : num_size, 1 : num_size, 1, num_size, num_size);
    while p <= q
        node1 = bfs_node(p);
        for node2 = 1 : size(schema.typegraph, 1)
            if schema.typegraph(node1, node2) > 0
                if node1 < node2
                    link = schema.links{node1, node2};
                else
                    link = schema.links{node2, node1}';
                end;
                matrix = bfs_mat{p} * link;
                if node2 == node
                    num_path = num_path + 1;
                    metapath{num_path} = matrix;
                elseif bfs_len(p) < max_len - 1
                    q = q + 1;
                    bfs_node(q) = node2;
                    bfs_len(q) = bfs_len(p) + 1;
                    bfs_mat{q} = matrix;
                end
            end
        end;
        bfs_mat{p} = [];
        p = p + 1;
    end;
end

function fea = getFeature(source, target, path1, path2, mat)
    num_sample = length(source);
    num_path1 = length(path1);
    num_path2 = length(path2);
    fea = zeros(num_sample, num_path1 * num_path2);
    idx = 0;
    for i = 1 : num_path1
        for j = 1 : num_path2
            idx = idx + 1;
            matrix = path1{i} * mat * path2{j};
            sum1 = sum(path1{i}, 2);
            sum2 = sum(path2{j}, 1);
            for k = 1 : num_sample
                fea(k, idx) = matrix(source(k), target(k)) ...
                    / sum1(source(k)) / sum2(target(k));
            end;
        end;
    end;
    fea(isnan(fea)) = 0;
end

function links = updateSchema(links, source, target, outputs, threshold)
    len = length(outputs);
    for k = 1 : len
        if outputs(k) > threshold || threshold == 0
            links(source(k), target(k)) = outputs(k);
        end;
    end;
end