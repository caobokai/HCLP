classdef ExpClassifier

    properties
        name;
        type;
        abstract;
        time_train;
        time_test;
        time;
    end

    methods
        function obj = ExpClassifier(name, type)
            obj.name = name;
            obj.type = type;
        end
    end
   
    methods (Abstract)
        [outputs, pre_labels, obj] = classify(...
            obj, train_data, train_label, test_data, varargin)
    end
end 
