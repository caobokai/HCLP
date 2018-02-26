classdef ExpDataset

    properties
        name;
        type;       
    end

    methods
        function obj = ExpDataset(name, type)
            obj.name = name;
            obj.type = type;
        end
    end
   
    methods (Abstract)
        [train_data, train_label, test_data, test_label] = load(varargin)
    end
end 
