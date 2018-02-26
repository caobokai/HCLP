classdef ExpClassifierSVM < ExpClassifier
    
    properties
        para_train = '-t 0 -h 0';
        model;
    end
    
    methods
        function s = ExpClassifierSVM()
            s = s@ExpClassifier('SVM', 'classification');
        end
        
        function [outputs,pre_labels,s] = classify(...
                s, train_data, train_label, test_data, varargin)
            train_data = double(train_data);
            test_data = double(test_data);
            train_label(train_label <= 0) = -1;
            s = s.train(train_data, train_label);
            [outputs, pre_labels, s] = s.test(test_data);
            s.time = s.time_train + s.time_test;
            
            % save running state discription
            s.abstract = [
                s.name  '(' ...
                '-time:' num2str(s.time) ...
                '-time_train:' num2str(s.time_train) ...
                '-time_test:' num2str(s.time_test) ...
                '-para:' s.para_train ')'];
        end
        
        function s = train(s, train_data, train_label)
            paraStr = s.para_train;
            t = cputime;
            s.model = svmtrain(train_label', train_data', paraStr);
            s.time_train = cputime - t;
        end

        function [outputs, pre_labels, s] = test(s, test_data)
            t = cputime;
            num_test = size(test_data,2);
            [pre_labels, ~, outputs] = svmpredict(...
                zeros(num_test, 1), test_data', s.model);
            s.time_test = cputime - t;
        end
    end
end

