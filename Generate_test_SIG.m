%% Initialization
clear all;
clc;
addpath(genpath('./Functions/'))

%% Parameters setting

angRes = 3; % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
angRes_label = 5;

sourceDataPath = 'F:\LFASR\Dataset\Test_mat\real\';
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);
idx = 0;

for DatasetIndex = 1 : datasetsNum
    DatasetName = sourceDatasets(DatasetIndex).name;
    SavePath = ['./Data/TestData', '_SIG2_' num2str(angRes), 'x', num2str(angRes), '_', 'ASR', '_', num2str(angRes_label), 'x', num2str(angRes_label),  '/', DatasetName];
    if exist(SavePath, 'dir')==0
        mkdir(SavePath);
    end
    
    sourceDataFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/'];
    folders = dir(sourceDataFolder); % list the scenes
    folders(1:2) = [];
    sceneNum = length(folders);
    
    for iScene = 1 : sceneNum
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating test data of Scene_%s in Dataset %s......\n', sceneName, sourceDatasets(DatasetIndex).name);
        dataPath = [sourceDataFolder, folders(iScene).name];
        data = load(dataPath);
        LF = data.LF;
        [U, V, H, W, ~] = size(LF);
        while mod(H, 2) ~= 0
            H = H - 1;
        end
        while mod(W, 2) ~= 0
            W = W - 1;
        end
        ind_all = linspace(1,angRes_label*angRes_label,angRes_label*angRes_label);
        ind_all = reshape(ind_all,angRes_label,angRes_label)';
        delt = (angRes_label-1)/(angRes-1);
        ind = ind_all(1:delt:angRes_label,1:delt:angRes_label);
        
        LF = LF(1:angRes_label, 1:angRes_label, 1:H, 1:W, 1:3);
        [~, ~, H_, W_, ~] = size(LF);
        LFlr = LF(1:delt:angRes_label,1:delt:angRes_label,:,:,:);
        [U, V, H, W, ~] = size(LF);
        label  = single(zeros(H, W, angRes_label*angRes_label-angRes*angRes));%single����ռ4���ֽ�angRes
        
        count = 0;
        for u = 1 : U
            for v = 1 : V
                k = (u-1)*U+v;
                if ismember(k,ind)
                    continue                
                else            
                    count = count + 1;
                    SAI_rgb = squeeze(LF(u, v, :, :, :));
                    SAI_ycbcr = rgb2ycbcr(double(SAI_rgb));
                    label(:,:,count) = SAI_ycbcr(:, :, 1);
                end
            end
        end
        label = permute(label,[2,1,3]); %[h,w,u,v] -> [w,h,v,u,N]  
%         label  = single(zeros(H, W, angRes_label, angRes_label));%single����ռ4���ֽ�angRes
%         
%         count = 0;
%         for u = 1 : U
%             for v = 1 : V
%                 SAI_rgb = squeeze(LF(u, v, :, :, :));
%                 SAI_ycbcr = rgb2ycbcr(double(SAI_rgb));
%                 label(:,:,u,v) = SAI_ycbcr(:, :, 1);
%             end
%         end
%         label = permute(label,[2,1,4,3]); %[h,w,u,v] -> [w,h,v,u,N]  
        
        
        [Ul, Vl, Hl, Wl, ~] = size(LFlr);
        data  = single(zeros(Hl, Wl, angRes, angRes));%single����ռ4���ֽ�angRes
        for u = 1 : Ul
            for v = 1 : Vl
                SAI_rgb = squeeze(LFlr(u, v, :, :, :));
                SAI_ycbcr = rgb2ycbcr(double(SAI_rgb));
                data(:,:,u,v) = SAI_ycbcr(:, :, 1); %h, w, u, v
            end
        end
        data = permute(data,[2,1,4,3]); %[h,w,u,v] -> [w,h,v,u]  
        %size(data)
        %size(label)        
        SavePath_H5 = [SavePath, '/', sceneName, '.h5'];
        h5create(SavePath_H5, '/data', size(data), 'Datatype', 'single');
        h5write(SavePath_H5, '/data', single(data), [1,1,1,1], size(data));
        h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
        h5write(SavePath_H5, '/label', single(label), [1,1,1,1], size(label));
        idx = idx + 1;
    end
end


