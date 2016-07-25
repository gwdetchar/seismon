function [calibrat_p_training_set] = calibrate_data(p_training_set,num_of_parameters)

for ind = 1:num_of_parameters
    calibrat_p_training_set(:,ind) = (p_training_set(:,ind) ...
        - mean(p_training_set(:,ind))) ./ ...
        (std(p_training_set(:,ind)));
end
end