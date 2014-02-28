% This is matlab file for back ground substraction, implementation of paper
% "BACKGROUND MODELING AND SUBTRACTION BY CODEBOOK CONSTRUCTION" by
% Kyungnam Kim, Thanarat H.. Chalidabhongse David Harwoor Larry Davis.

% Dhaval Malaviya
%

function [] = bgsubtraction(imageSeqDirectory,alpha,beta,epsilon1, epsilon2,generateCodeBook,bgsubtraction)

% creating empty structure for the codebook.
global codeBook
codeBook = struct([]);

global sequence
sequence = struct([]);

global foregroundPIndex

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read image frame of the given sequence of images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Below if condition is for determining the whether Code Book is generated
% or actual back ground subtraction is being performed.
% Flag generateCodeBook == 1 then code book is generated
% Flag bgsubtraction == 1 then actual back ground subtraction is performed.

if (generateCodeBook == 1)
    % Read the squence of the images:
    imageSeq = dir(imageSeqDirectory);
    
    [total,dummy] = size(imageSeq);
    
    imageFrame = 0;
    
    for imageInSeq = 1 : total
        
        % checking the file name and if the file name has "PetsD2Te" in it then that is
        % an desired image frame and we need to proceed with it.
        
        [imgpath,imgname,imgext] = fileparts(imageSeq(imageInSeq).name);
        checkForImage = strfind(imgname,'PetsD2Te');
        
        if(~isempty(checkForImage))
            imageFrame = imageFrame + 1;
            imagePath = strcat(imageSeqDirectory,'\',imageSeq(imageInSeq).name);
            
            image_utest = imread(imagePath);
            
            [row col] = size(image_utest);
            
            if(imageFrame == 1)
                % initializing the numberOfCodeWord for each pixel to zero
                for  i = 1 : row*col
                    codeBook(i).numberOfCodeWord = 0;
                end
            end
            
            % calling the function to calculate the code book for each pixel of
            % the frame under consideration.
            createCodeBook(image_utest,imageFrame,alpha,beta,epsilon1);
        end
    end
    
    % step 3 for each cordword wrap around lambda by setting lambda =
    % max(lambda,(N-qi+pi-1))
    for j = 1:row*col
        if(codeBook(j).numberOfCodeWord ~= 0)
            for k = 1 : codeBook(j).numberOfCodeWord
                codeBook(j).codeword(k).aux(4) = max(codeBook(j).codeword(k).aux(4),(imageFrame-codeBook(j).codeword(k).aux(6)+codeBook(j).codeword(k).aux(5)-1));
            end
        end
    end
    
    % Remove the unwanted code words and make codebook thin
    % remove the code words which have lambda value less than total training
    % frames / 2
    for k = 1 : row*col
        noOfCodeWord = codeBook(k).numberOfCodeWord;
        %     for l = 1 : noOfCodeWord
        %         if(codeBook(k).codeword(l).aux(4) > (imageFrame/2))
        %             codeBook(k).codeword(l) = [];
        %             codeBook(k).numberOfCodeWord = codeBook(k).numberOfCodeWord - 1;
        %             noOfCodeWord = noOfCodeWord - 1;
        %         end
        %     end
        while (noOfCodeWord > 0)
            if(codeBook(k).codeword(noOfCodeWord).aux(4) > (imageFrame/2))
                codeBook(k).codeword(noOfCodeWord) = [];
                codeBook(k).numberOfCodeWord = codeBook(k).numberOfCodeWord - 1;
                noOfCodeWord = noOfCodeWord - 2;
            else
                noOfCodeWord = noOfCodeWord - 1;
            end
        end
    end
    
    save('codeBook_video1_1.mat','codeBook');
    
    % Below else if is for bgsubtraction algorithm
elseif(bgsubtraction == 1)
    % loading the already calculated codeBook.
    load('codeBook_video1_1.mat');
    % Read the squence of the images:
    imageSeq = dir(imageSeqDirectory);
    
    [total,dummy] = size(imageSeq);
    
    imageFrame = 0;
    
    for imageInSeq = 1 : total
        
        % checking the file name and if the file name has "PetsD2Te" in it then that is
        % an desired image frame and we need to proceed with it.
        
        [imgpath,imgname,imgext] = fileparts(imageSeq(imageInSeq).name);
        checkForImage = strfind(imgname,'PetsD2Te');
        
        if(~isempty(checkForImage))
            imageFrame = imageFrame + 1;
            imagePath = strcat(imageSeqDirectory,'\',imageSeq(imageInSeq).name);
            
            image_utest = imread(imagePath);
            
            [row col] = size(image_utest);
            
            % calling the function which searches the code book of each respective pixel of
            % the frame under consideration and looks for similarity.
            % similarity is determined based on two functions:
            % a) colorDist(x,vm)
            % b) brightness(I,(Imin,Imax))
            searchCodeBook(image_utest,imageFrame,alpha,beta,epsilon2);
            
        end
    end
    %save('C:\dhaval\resultSequenceBinary.mat','sequence');
    
    % Morphological Operations on the binary image.
    % Use Dilation first and then erosion so that
    % small holes (noise) will be filled up.
    se = strel('square',10);
    
    [dummy total] = size(sequence);
    %hold on;
    for n = 1:total
        sequence(n).binaryI = imclose(sequence(n).binaryI,se);
        imshow(sequence(n).binaryI);
    end
    %hold off;
    
    %     hold on;
    %     % Generating the connected components using matlab functions
    %     for n = 1:total
    %         %markedMatrix = explore(sequence(n).binaryI);
    %         connectedComponents = bwconncomp(sequence(n).binaryI);
    %         for i = 1: connectedComponents.NumObjects
    %             pixelsInCC = cell2mat(connectedComponents.PixelIdxList(i));
    %             numPixel = size(pixelsInCC)
    %             if ( numPixel(1) > 10 )
    %                 % sort the pixel index which is linear and then
    %                 % find the mid pixel and draw the bounding box around it.
    %                 boundingBoxLeftUC = min(pixelsInCC);
    %                 boundingBoxRightLC = max(pixelsInCC);
    %                 if(boundingBoxLeftUC > row)
    %                     xUL = rem(boundingBoxLeftUC,row);
    %                     yUL = ceil(boundingBoxLeftUC/row);
    %                 else
    %                     xUL = boundingBoxLeftUC;
    %                     yUL = 1;
    %                 end
    %
    %                 if(boundingBoxRightLC > row)
    %                     xLR = rem(boundingBoxRightLC,row);
    %                     yLR = ceil(boundingBoxRightLC/row);
    %                 else
    %                     xLR = boundingBoxRightLC;
    %                     yLR = 1;
    %                 end
    %                 % plot the rectangle on the binary image.
    %                 figure;
    %                 imshow(sequence(n).binaryI);
    %
    %                 % find the other two coordinates to draw the rectangle around the
    %                 % object
    %                 xUR = xUL;
    %                 yUR = yLR;
    %                 xLL = xLR;
    %                 yLL = xUL;
    %
    %                 line([xUR xLR],[yUR yLR]);
    %                 line([xUR xUL],[yUR yUL]);
    %                 line([xUL xLR],[yUL yLR]);
    %                 line([xLR xLL],[yLR yLL]);
    %             end
    %
    %         end
    %
    %
    %         hold off;
    %         % calculating the area of the regions in the image
    %        % areas = regionprops(markedMatrix,'Area');
end

end





% Calling the function for creating code book.
function [] = createCodeBook(image_utest,t,alpha,beta,epsilon1)

global codeBook

% separating the three channels of the color image.
r = image_utest(:,:,1);
g = image_utest(:,:,2);
b = image_utest(:,:,3);

% Here I am converting the image matrix into the nx1 dimensinal array to
% ease the computation.
% Change r,g,b and image frame under consideration into nx1 dimensional
% array
[row col] = size(r);

r = r';
oneDred = reshape(r,row*col,1);
g = g';
oneDgreen = reshape(g,row*col,1);
b = b';
oneDblue = reshape(b,row*col,1);

% Here i will create multidimensional matrix. that is matrix of 10 rows, 9
% coloumns and N number of pixels.
% here 10 rows is assuming maximum code words that can be created for any
% particular pixel's code book.
% 9 coloumns represents the 9 variables viz R,G,B,Imax,Imin,freq,lambda
% maximum negative run length, p firt access times, q last access times of
% the code word.
% N represents the number of pixels or size of the images.


% initializing the variable representing the number of code word for
% particular pixel.

% for each pixel of the image, obtain the code word there by creating the
% code book for each pixel.
[rowI colI channel] = size(image_utest);



%epsilon1 = 20;
for i = 1 : rowI*colI
    
    % Initilizing the index which indicates the code word which is matched
    % with the generated code word of the current pixel
    matchedCodeWord = 0;
    
    %x(i,j) = [r(i,j),g(i,j),b(i,j)];
    % storing the intensity of the pixel
    I = oneDred(i) + oneDgreen(i) + oneDblue(i);
    
    % finding the codeword cm in C = {ci | 1 <= i <= L) matching to xt
    % based on two conditions (a) and (b)
    % (a) colordist(xt,vm) <= epsilon1
    % (b) brightness(I,(Imin,Imax)) = true
    if codeBook(i).numberOfCodeWord ~= 0
        for j = 1 : codeBook(i).numberOfCodeWord
            % As size of the code book is restricted to 10 maximum.
            % And it is possible that all the codw word of the book is not
            % filled and hence some next to last code word will be zeros.
            if ((epsilon1 >= colorDist(oneDred(i),oneDgreen(i),oneDblue(i),codeBook(i).codeword(j).v(1),codeBook(i).codeword(j).v(2),codeBook(i).codeword(j).v(3))) ...
                    && (brightness(I,codeBook(i).codeword(j).aux(1),codeBook(i).codeword(j).aux(2),alpha,beta) == 1))
                matchedCodeWord = j;
            end
        end
    end
    
    
    % iii) If C = null or there is no match then L = L+1 Create a new
    % codeword cL by setting
    % vL = (R,G,B)
    % auxL = (I,I,1,t-1,t,t)
    
    if codeBook(i).numberOfCodeWord == 0 || matchedCodeWord == 0
        % if the code word set is empty or no code was matching the
        % for the current frame data then do the following. Create the
        % new code word. t = frame Number
        % and also increment the numberOfCodeWord counter for
        % respective pixel by 1
        newCodewordIndex = codeBook(i).numberOfCodeWord + 1;
        codeBook(i).codeword(newCodewordIndex).v = [oneDred(i) oneDgreen(i) oneDblue(i)];
        codeBook(i).codeword(newCodewordIndex).aux = [I I 1 (t-1) t t];
        codeBook(i).numberOfCodeWord = codeBook(i).numberOfCodeWord + 1;
    else
        % otherwise update the existing code book.
        % update the matched code word by doing following
        % suppose existing code word is
        % vm = (rm, gm,bm) and auxm = (Imin,Imax,fm,lambdam,pm,qm)
        % update by following:
        % vm = ((fm*rm + r)/(fm+1),(fm*gm + g)/(fm+1),(fm*bm +
        % b)/(fm+1))
        % auxm = (min(I,Imin),max(I,Imax),fm+1,max(lambdam,t-qm),pm,t)
        
        codeBook(i).codeword(matchedCodeWord).v = ...
            [((codeBook(i).codeword(matchedCodeWord).aux(3) * codeBook(i).codeword(matchedCodeWord).v(1) + r(i))...
            / (codeBook(i).codeword(matchedCodeWord).aux(3) + 1)) ...
            ((codeBook(i).codeword(matchedCodeWord).aux(3)* codeBook(i).codeword(matchedCodeWord).v(2) + g(i))...
            / (codeBook(i).codeword(matchedCodeWord).aux(3) + 1)) ...
            ((codeBook(i).codeword(matchedCodeWord).aux(3)*codeBook(i).codeword(matchedCodeWord).v(3) + b(i))...
            / (codeBook(i).codeword(matchedCodeWord).aux(3) + 1))];
        
        codeBook(i).codeword(matchedCodeWord).aux = [min(I,(codeBook(i).codeword(matchedCodeWord).aux(1))) ...
            max(I,(codeBook(i).codeword(matchedCodeWord).aux(2))) (codeBook(i).codeword(matchedCodeWord).aux(3)+1) ...
            max(codeBook(i).codeword(matchedCodeWord).aux(4),(t-codeBook(i).codeword(matchedCodeWord).aux(6))) ...
            (codeBook(i).codeword(matchedCodeWord).aux(5)) t];
    end
end
end


% below function is used to calculate the color distortion between the
% pixel under consideration and to decide whether to create new code word
% or not. If distortion is tolerable then no need to create new code word
function [colorDistortionDelta] = colorDist(r,g,b,vmR,vmG,vmB)
xt = (r*r + g*g + b*b);
v  = (vmR*vmR + vmG*vmG + vmB*vmB);

combinedXtv = (r*vmR + g*vmG + b*vmB) * (r*vmR + g*vmG + b*vmB);

p = combinedXtv / v;

colorDistortionDelta = sqrt((double(xt) - double(p)));


end

% below function is used to calculate the brightness change direction
function [trueOrfalse] = brightness(I,Imin,Imax,alpha,beta)

Ilow = alpha*Imax;
Ihi = min((beta*Imax),(Imin/alpha));

if ((Ilow <= I) && (I <= Ihi))
    trueOrfalse = 1;
else
    trueOrfalse = 0;
end
end

% Search code book for matching codeword to decide the foreground and
% background pixels.
function [] = searchCodeBook(image_utest,imageFrame,alpha,beta,epsilon2)

global codeBook
global sequence
global foregroundPIndex

% separating the three channels of the color image.
r = image_utest(:,:,1);
g = image_utest(:,:,2);
b = image_utest(:,:,3);

% Here I am converting the image matrix into the nx1 dimensinal array to
% ease the computation.
% Change r,g,b and image frame under consideration into nx1 dimensional
% array
[row col] = size(r);

r = r';
oneDred = reshape(r,row*col,1);
g = g';
oneDgreen = reshape(g,row*col,1);
b = b';
oneDblue = reshape(b,row*col,1);

%epsilon2 = 22;

% Below is the binary image which stores the 0 for the foreground image and
% 255 for the background image.
foregroundPIndex = zeros(1,(row*col));

%foregroundPIndex = 255*foregroundPIndex;
%foregroundPIndex = reshape(foregroundPIndex,row*col,1);

for i = 1 : row*col
    
    % Initilizing the index which indicates the code word which is matched
    % with the generated code word of the current pixel
    foundCodeWord = 0;
    
    %x(i,j) = [r(i,j),g(i,j),b(i,j)];
    % storing the intensity of the pixel
    I = oneDred(i) + oneDgreen(i) + oneDblue(i);
    
    % finding the codeword cm in C = {ci | 1 <= i <= L) matching to xt
    % based on two conditions (a) and (b)
    % (a) colordist(xt,vm) <= epsilon1
    % (b) brightness(I,(Imin,Imax)) = true
    if codeBook(i).numberOfCodeWord ~= 0
        for j = 1 : codeBook(i).numberOfCodeWord
            % As size of the code book is restricted to 10 maximum.
            % And it is possible that all the codw word of the book is not
            % filled and hence some next to last code word will be zeros.
            if ((epsilon2 >= colorDist(oneDred(i),oneDgreen(i),oneDblue(i),codeBook(i).codeword(j).v(1),codeBook(i).codeword(j).v(2),codeBook(i).codeword(j).v(3))) ...
                    && (brightness(I,codeBook(i).codeword(j).aux(1),codeBook(i).codeword(j).aux(2),alpha,beta) == 1))
                foundCodeWord = j;
            end
        end
    end
    
    % if code word is found for the current pixel then it indicates that it
    % is the background pixel and if no match for the code word is found
    % then it is foreground pixel
    if(foundCodeWord == 0)
        oneDred(i) = 255;
        oneDgreen(i) = 255;
        oneDblue(i) = 255;
        foregroundPIndex(i) = 1;
    end
end


% Reshaping the one dimensional channel into rowxcol matrix
r = reshape(oneDred,col,row);
g = reshape(oneDgreen,col,row);
b = reshape(oneDblue,col,row);

foregroundPIndex = reshape(foregroundPIndex,col,row);
foregroundPIndex = foregroundPIndex';

image_result(:,:,1) = r';
image_result(:,:,2) = g';
image_result(:,:,3) = b';

sequence(imageFrame).image(:,:,:) = image_result;
sequence(imageFrame).binaryI(:,:) = foregroundPIndex;

%imshow(image_result);

end


% Connected component exploration algorithm
% Author: Will Chang
% Source: http://cseweb.ucsd.edu/classes/sp06/cse152/explore.m
function marked = explore(image)
%
%  image is a 2D array that contains values (zeros and ones)
%  corresponding to the binarized image.
%
s = size(image);
width = s(1); height = s(2);

marked = zeros(width, height);			% marked image
marker = 0;								% marker index

% keep a stack of pixel locations that we need to explore
stack = zeros(width*height, 2);
topStack = 0;

for i=1:width
    for j=1:height
        if ((image(i,j) == 1) && (marked(i,j) == 0))
            marker = marker + 1;
            
            % push the current location on the stack
            topStack = topStack + 1;
            stack(topStack, 1) = i;
            stack(topStack, 2) = j;
            
            % use a stack to emulate recursion
            while (topStack > 0)
                nx = stack(topStack, 1);	% x-coordinate of neighbor
                ny = stack(topStack, 2);	% y-coordinate of neighbor
                topStack = topStack - 1;
                
                if ((image(nx, ny) == 1) && (marked(nx, ny) == 0))
                    marked(nx, ny) = marker;
                    % iteratively explore the connected components
                    % I'm assuming 8-connectedness here
                    for x=-1:1
                        for y=-1:1
                            % push the neighbor on the stack... only if
                            % pixel coordinates are inside the boundaries
                            if ((nx+x >= 1) && (nx+x <= width) &&...
                                    (ny+y >= 1) && (ny+y <= height))
                                topStack = topStack + 1;
                                stack(topStack, 1) = nx+x;
                                stack(topStack, 2) = ny+y;
                            end
                        end
                    end
                end
            end
        end
    end
end
end




