clc;
clear all;



for split = 1:1:2 
    if split == 1
        file_in = '/home/zengwh/DeblurGANv2/GOPRO_Large/train/';
    else
        file_in = '/home/zengwh/DeblurGANv2/GOPRO_Large/test/';
    end

    subdir  = dir( file_in );
    finenum = 0;
    % 进入GOPR0372_07_00这一级别的每一层
    for i = 1 : length( subdir )

        if(     isequal( subdir( i ).name, '.' )||...
                isequal( subdir( i ).name, '..')||...
                ~subdir( i ).isdir)               % if not subdir
            continue;
        end

        subsubdirname = fullfile( file_in, subdir(i).name );
        subsubdir = dir(subsubdirname);

        % 进入GOPR0372_07_00\sharp这一级别的每一层
        for j = 1 : length( subsubdir )

            if(  ~isequal( subsubdir(j).name, 'sharp' ) )
                continue;
            end

            %writedir = fullfile(file_in,subdir(i).name,'sharp_noi');
            if split == 1
                write_shape_dir = ['../datasets/source/train/shape/',subdir(i).name];
                write_blur_dir = ['../datasets/source/train/blur/',subdir(i).name];
                write_dark_dir = ['../datasets/source/train/dark/',subdir(i).name];
            else
                write_shape_dir = ['../datasets/source/test2/shape/',subdir(i).name];
                write_blur_dir = ['../datasets/source/test2/blur/',subdir(i).name];
                write_dark_dir = ['../datasets/source/test2/dark/',subdir(i).name];
            end


            if ~exist(write_shape_dir,'dir')
                mkdir(write_shape_dir);
            end

            if ~exist(write_blur_dir,'dir')
                mkdir(write_blur_dir);
            end

            if ~exist(write_dark_dir,'dir')
                mkdir(write_dark_dir);
            end


            subsubdirimg =  fullfile( file_in, subdir(i).name, subsubdir(j).name , '*.png' );
            imgpath = dir(subsubdirimg);

            % 进入 GOPR0372_07_00\sharp\000047.png 这一级别的每一层
           for k = 1:length(imgpath)
            %for k = 1:1     % 暂时只遍历一次；
                k
                imgreadpath = fullfile( file_in, subdir(i).name, subsubdir(j).name, imgpath(k).name);
                blur_imgreadpath = fullfile( file_in, subdir(i).name, 'blur', imgpath(k).name);
                input_ori = imread(imgreadpath);

                write_shape_img = fullfile( write_shape_dir,imgpath(k).name);
                imwrite(uint8(input_ori),write_shape_img,'png');

                input_blur = imread(blur_imgreadpath);
                write_blur_img = fullfile(write_blur_dir,imgpath(k).name);
                imwrite(uint8(input_blur),write_blur_img,'png');



                input = input_ori;
                input=double(input/5) + (5)*randn(size(input));
                %input = double(input)./255.0;
    %             A = 1;
    %             gamma = 0.45;
    %             output = A*(input.^gamma);
                %output = input.*255;
                write_dark_img = fullfile(write_dark_dir,imgpath(k).name);
                imwrite(uint8(input),write_dark_img,'png');


                finenum = finenum + 1;

            end    

        end 

    end
end