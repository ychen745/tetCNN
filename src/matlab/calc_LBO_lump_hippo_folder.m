function calc_LBO_lump_hippo(include_dir, data_dir)
    function all_includes = all_include(folder_stack)
        all_includes = {};
        if isempty(folder_stack)
            return
        end
        folder = cell2mat(folder_stack(1));
        folder_stack = folder_stack(2:length(folder_stack));
        subfolders = split(genpath(folder), ':');
        subfolders = subfolders(2:length(subfolders)-1);
        all_includes = {folder};
        if ~isempty(subfolders)
            folder_stack = [folder_stack; subfolders];
        end
        all_includes = [all_includes; all_include(folder_stack)];
    end

    % include_dir = '/home/ychen855/tetCNN/src/matlab'
    allf = all_include({include_dir});
    for idx = 1:length(allf)
        % allf(idx)
        addpath(cell2mat(allf(idx)));
    end

    % addpath /home/ychen855/tetCNN/src/matlab;

    % data_dir = '/home/ychen855/tetCNN/data/328/rh/ad';

    dire = dir(data_dir);
    dire(~[dire.isdir]) = [];
    dire = dire(3:end);

    for i = 1:length(dire)
        subFolder_name{i} = dire(i).name;
        SubFolderDir{i} = [data_dir,'/',dire(i).name];
    end

    numberOfFolders = length(SubFolderDir);
    %%
    for n = 1:numberOfFolders
        cd(SubFolderDir{n});
        display(subFolder_name{n});
        if isfile('lh_hippo.1.node')
            [elem, node, face] = read_tetra('lh_hippo.1');
            L = cotmatrix(node,elem(:,1:4));
            M = massmatrix(node,elem(:,1:4), 'barycentric');
            m = diag(M);
            save('mass.mat','m')
            save('cot.mat','L')
            %weakform('weakform_diag.m', 'weakformM_other.m');
            disp('lump is completed!');
        end
    end
end
