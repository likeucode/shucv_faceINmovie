clear
clc

folder_sor_name = '/home/byx/openface-master/test_acc/db/alignedtrain/';


dir_sor_Output=dir(folder_sor_name);
Sor_fileNames={dir_sor_Output.name}';
n = length(Sor_fileNames);


for i=1:n
    if  strcmp(Sor_fileNames{i},'.') || strcmp(Sor_fileNames{i},'..')
        continue
    end

    med_name = Sor_fileNames{i};
    sub_folder_name = fullfile([folder_sor_name,med_name,'/']);
    dir_sub_Output=dir(sub_folder_name);
    file_Names={dir_sub_Output.name}';
    m = length(file_Names);
    
    for j=1:m
        if  strcmp(file_Names{j},'.') || strcmp(file_Names{j},'..')
            continue
        end

        final=file_Names{j};      
        numlabel=j-2;
        system(['mv' ' "' sub_folder_name file_Names{j}  '" ' sub_folder_name med_name '_' num2str(numlabel) final(end-3:end)]);

    end   
end

