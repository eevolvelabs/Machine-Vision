function [] = faceDetection(numberOfTranImage,imageSeqDirectory,testImagePath,nonFaceImages,debug)

%global gamma
%global phi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read images from the training set and collecting them in matrix gamma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read the squence of the images:
imageSeq = dir(imageSeqDirectory);

[total,dummy] = size(imageSeq);

i = 0;


for imageInSeq = 1 : total
    
    % checking the file name and if the file name has "subject" in it then that is
    % an desired image file and we need to proceed with it.
    
    [imgpath,imgname,imgext] = fileparts(imageSeq(imageInSeq).name);
    checkForImage = strfind(imgname,'subject');
    
    if(~isempty(checkForImage))
        i = i + 1;
        imagePath = strcat(imageSeqDirectory,'\',imageSeq(imageInSeq).name);
        
        image_utest = imread(imagePath);
        
        
        % converting color image to gray image
        [rowIn colIn ch] = size(image_utest);
        
        %         if(ch ~= 1)
        %             greyImage = rgb2gray(image_utest);
        %         end
        %
        % as the reshape function sorts the matrix coloumn wise hence I
        % shall take the transpose of the test image matrix and then will
        % use reshape function.
        image_utestTrans = image_utest';
        [rows,cols] = size(image_utestTrans);
        
        % converting rows x cols matrix into (rows*cols) x 1 matrix one
        % dimensional matrix
        oneDImageVectorMatrix = reshape(image_utestTrans,rows*cols,1);
        
        % gather the each one dimensional matrix representing of the images
        % into a matrix which will be used later in the algorithm
        gamma(:,i) = oneDImageVectorMatrix;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculating the average of all the one dimensional representation of
% the images under training set.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[row,col] = size(gamma);

psiTotal = zeros(row,1);

for j = 1:col
    psiTotal = double(gamma(:,j))+ psiTotal;
end

% averaging the phiTotal over total images in training set
psi = psiTotal/col;

% rounding the psi pixel values to display on the screen
psi = round(psi);
psi = uint8(psi);

% Displaying the average image
figure
axes
img=reshape(psi,colIn,rowIn);
img=img';
img = round(img);
img = uint8(img);
%subplot(ceil(sqrt(col)),ceil(sqrt(col)),i)
imshow(img)



% obtaining phi vector Obtained by subtracting average one dimensional
% vector obtained above (psi) from each image one dimensional vector stored
% in matrix gamma.
for k = 1:col
    A(:,k) = gamma(:,k) - psi;
end

A = double(A);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performing PCA in three different ways.
% a) eigenVectors of the XX', where X = A
% b) Use SVD
% c) EigenVector of X'X then using scheme proposed by Turk and Pentland in
% paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(debug == 1)
    tic;
    covarianceVector = A*A';
    % finding the eigen vector of the A*A'
    [eigVector1,eigValues1] = eig(covarianceVector);
    time1 = toc;
end

if(debug == 2)
    tic;
    [U,S,V] = svd(A);
    time2 = toc;
end


% As the covariance vector of A*A' will be of (col*row) * (col*row) size
% hence its egien values will also be that much large and it is difficult
% to store, hence to ease this we shall find eigen values of the A'*A so
% that eigen values will be of size MxM (M= number of the images in
% training set)
L = A'*A;

% calculate the eigen value
tic;
[v,d] = eig(L);
time3 = toc;

% Initilaizing the eigenfaces u
u = zeros(row,col);

% calculating the eigen faces from the above results:
% here 'v' eigenVectors of L are stored coloumn wise hence
% in below equation I am traversing the v coloumn wise.
for i = 1:col
    for j = 1:col
        u(:,i) = u(:,i) + v(j,i)*double(A(:,j));
    end
end

for i = 1 : col
    unormal(:,i) = u(:,i) / norm(u(:,i));
end

% as the eigen faces created in above step may have coordinate values less
% than zero or can be greater than 255. converting less than zero value to
% zero and max value to 255

% first step is to round the values in the matrix.
u = round(u);
[row , col] = size(u);

% now taking first 15 sorted eigenValues
n = 25;
% display the eigen faces
figure;
for i=1:size(u,2)
%for i=1:n
    image1DVec = u(:,i);
    img=reshape(image1DVec,colIn,rowIn);
    img=img';
    img = round(img);
    img = uint8(img);
    %img=histeq(img,255);
    subplot(ceil(sqrt(col)),ceil(sqrt(col)),i)
    imshow(img)
    drawnow;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding the n significant eigen vectors with largest associated
% eigenvalues
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[eigenrow,eignecol] = size(d);
for i = 1 : eigenrow
    eigenValues(i) = d(i,i);
end

% sorting the eigenValues
[sortEigenValues,index] = sort(eigenValues,'descend');



for i = 1 : n
    eigenVectorsSorted(:,i) = v(:,index(i));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculating the weights of the training images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1 : size(u,2)
    for j = 1 : size(u,2)
        w(i,j) = unormal(:,index(j))'*A(:,i);
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading the test image and projecting on to the Eigen faces
% here read image can be face or non face images. Hence every time 
% after reading new image, check for its size, because non face images
% requires the sizeing before it can be used aby further.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(nonFaceImages == 0)
    image_Test = imread(testImagePath);
    % converting color image to gray image
    [row col ch] = size(image_Test);
    
    if(ch ~= 1)
        greyImage = rgb2gray(image_Test);
    else
        greyImage = image_Test;
    end
    
    nonFaceImage = 0;
    % Check for read image. If size is not same as the images used for cal
    % culating the eigenfaces, then such image needs resizing.
    if (cols ~= col && rows ~= row)
        greyImage = imresize(greyImage,[rowIn colIn]);
        nonFaceImage = 1;
    end
    
    image_Test = greyImage';
    
    figure
    imshow(image_Test');
    
    [testRow,testCol] = size(image_Test);
    
    gamma_test = reshape(image_Test,(testRow*testCol),1);
    
    diffTestImage = gamma_test - psi;
    
    % Finding the weights.
    %for i = 1 : size(u,2)
    for j = 1 : n
        weightsTest(j) = unormal(:,index(j))' * double(diffTestImage);
    end
    %end
    
    % normalize the weights
    %for i = 1: n
    weightMax = max(weightsTest);
    %weight = weightsTest/weightMax;
    weight = weightsTest/1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Reconstructing the test image
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1 : n
        reconstTestImage(:,j) = weight(j) * unormal(:,index(j));
    end
    
    reconstImage = zeros(testRow*testCol,1);
    
    for j = 1: n
        reconstImage = reconstTestImage(:,j) + reconstImage;
    end
    
    
    reconstImage = reconstImage + double(psi);
    img=reshape(reconstImage,colIn,rowIn);
    img=img';
    img = round(img);
    img = uint8(img);
    figure
    imshow(img)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Finding the image in training set close to the image from test set.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% Calculating Euclidean distances
    % Euclidean distances between the projected test image and the projection
    % of all centered training images are calculated. Test image is
    % supposed to have minimum distance with its corresponding image in the
    % training database.
    
    %Euc_dist = [];
    for j = 1 : size(u,2)
        %for j = 1 : n
        weightsTestEuc(j) = unormal(:,index(j))' * double(diffTestImage);
    end
    for i = 1 : size(u,2)
        %for i = 1 : n
        q = w(i,:);
        Euc_dist(i) = ( norm( weightsTestEuc - q ) )^2;
    end
    
    [Euc_dist_min , Recognized_index] = min(Euc_dist);
    %OutputName = strcat(int2str(Recognized_index),'.jpg');
    
    % Displaying the the found image and the test image.
    i = 0;
    figure;
    for imageInSeq = 1 : total
        
        % checking the file name and if the file name has "subject" in it then that is
        % an desired image file and we need to proceed with it.
        
        [imgpath,imgname,imgext] = fileparts(imageSeq(imageInSeq).name);
        checkForImage = strfind(imgname,'subject');
        
        if(~isempty(checkForImage))
            i = i + 1;
            
            if(Recognized_index == i)
                imagePath = strcat(imageSeqDirectory,'\',imageSeq(imageInSeq).name);
                imshow(imagePath);
            end
        end
        
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculating the Frobenius norm of each non face images and also
% of the test images and plotting the values.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(nonFaceImages == 1)
    % Read the squence of the images:
    imageSeq = dir(testImagePath);
    
    [total,dummy] = size(imageSeq);
    
    i = 0;
    
    for imageInSeq = 1 : total
        
        % checking the file name and if the file name has "subject" in it then that is
        % an desired image file and we need to proceed with it.
        
        [imgpath,imgname,imgext] = fileparts(imageSeq(imageInSeq).name);
        checkForImage = strfind(imgname,'NonFace');
        checkForImage = strfind(imgname,'subject');
        if(~isempty(checkForImage))
            i = i + 1;
            imagePath = strcat(testImagePath,'\',imageSeq(imageInSeq).name);
            
            image_utest = imread(imagePath);
            
            % converting color image to gray image
            [row col ch] = size(image_utest);
            if(ch ~= 1)
                greyImage = rgb2gray(image_utest);
            else
                greyImage = image_utest;
            end
            % Check for read image. If size is not same as the images used for cal
            % culating the eigenfaces, then such image needs resizing.
            if (colIn ~= col && rowIn ~= row)
                greyImage = imresize(greyImage,[rowIn colIn]);
            
            end
            
            image_Test = greyImage';
            
            %figure
            %imshow(image_Test');
            
            [testRow,testCol] = size(image_Test);
            
            gamma_test = reshape(image_Test,(testRow*testCol),1);
            
            diffTestImage = gamma_test - psi;
            
            % Finding the weights.
            %for i = 1 : size(u,2)
            for j = 1 : n
                weightsTest(j) = unormal(:,index(j))' * double(diffTestImage);
            end
            %end
            
            % normalize the weights
            %for i = 1: n
            weightMax = max(weightsTest);
            %weight = weightsTest/weightMax;
            weight = weightsTest/1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Reconstructing the test image
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for j = 1 : n
                reconstTestImage(:,j) = weight(j) * unormal(:,index(j));
            end
            
            reconstImage = zeros(testRow*testCol,1);
            
            for j = 1: n
                reconstImage = reconstTestImage(:,j) + reconstImage;
            end
            
            reconstImage = reconstImage + double(psi);
            img=reshape(reconstImage,colIn,rowIn);
            img=img';
            
            % Find the difference between original image and reconstructed image
            diffImage = double(img) - double(greyImage);
            img = round(img);
            diffImage = round(diffImage);
            diffImage = uint8(diffImage);
%             img = uint8(img);
%             figure;
%             imshow(diffImage);
%             figure;
%             imshow(img);
            froNormNonFace(i) = norm(double(diffImage),'fro');
            
        end
    end
    figure;
    plot(froNormNonFace,'-.o','MarkerFaceColor','g','MarkerSize',10);
end

    
end




