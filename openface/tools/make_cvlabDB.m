clear
clc
folder_sor_name='/home/byx/openface-master/test_acc/test/';
folder_des_name='/home/byx/openface-master/test_acc/db/';
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
    Sub_fileNames={dir_sub_Output.name}';
    m = length(Sub_fileNames);
    
    if m<10
        continue
    end
    mkdir([folder_des_name,'/train/'],med_name);
    mkdir([folder_des_name,'/val/'],med_name);
    mkdir([folder_des_name,'/test/'],med_name);
    
    train_folder_name = fullfile([folder_des_name,'/train/',med_name,'/']);
    val_folder_name = fullfile([folder_des_name,'/val/',med_name,'/']);
    test_folder_name = fullfile([folder_des_name,'/test/',med_name,'/']);
    
    for j=3:(ceil((m-2)/10)+3) 
%         copyfile([sub_folder_name,Sub_fileNames{j}],val_folder_name,'f');   
        copyfile([sub_folder_name,Sub_fileNames{j}],val_folder_name);   
    end   
    for k=(ceil((m-2)/10)+4):((ceil((m-2)/10)+3)*2) 
        copyfile([sub_folder_name,Sub_fileNames{k}],test_folder_name);   
    end   
    for l=(((ceil((m-2)/10)+3)*2)+1):m 
        copyfile([sub_folder_name,Sub_fileNames{l}],train_folder_name);   
    end   
    
end