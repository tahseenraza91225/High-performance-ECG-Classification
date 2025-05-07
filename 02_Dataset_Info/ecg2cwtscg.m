%Function ecg2cwtscg(): converting ecg to cwt scalogram and saving as image

function ecg2cwtscg(ecgdata,cwtfb,ecgtype)
nos=20; %number of signals
nol=500; %signal length
colormap=jet(128);
if ecgtype=="ARR"
folderpath=fullfile("ecgdataset",ecgtype);
findx=0;
    for i=1:30
        indx=0;
        for k=1:nos
            ecgsignal=ecgdata(i,indx+1:indx+nol);
            cfs =abs(cwtfb.wt(ecgsignal));
            im=ind2rgb(im2uint8(rescale(cfs)),colormap);
            filenameindex=findx+k;
            filename=fullfile(folderpath,sprintf("%s_%d.jpg",lower(ecgtype),filenameindex));
            imwrite(imresize(im,[227 227]),filename);
            indx=indx+nol;
        end
        findx=findx+nol;
    end
elseif ecgtype=="CHF"  
folderpath=fullfile("ecgdataset",ecgtype);
findx=0;
    for i=1:30
        indx=0;
        for k=1:nos
            ecgsignal=ecgdata(i,indx+1:indx+nol);
            cfs =abs(cwtfb.wt(ecgsignal));
            im = ind2rgb(im2uint8(rescale(cfs)),colormap);
            filenameindex=findx+k;
            filename=fullfile(folderpath,sprintf("%s_%d.jpg",lower(ecgtype),filenameindex));
            imwrite(imresize(im,[227 227]),filename);
            indx=indx+nol;
        end
        findx=findx+nol;
    end
elseif ecgtype=="NSR"   
folderpath=fullfile("ecgdataset",ecgtype);
findx=0;
    for i=1:30
        indx=0;
        for k=1:nos
            ecgsignal=ecgdata(i,indx+1:indx+nol);
            cfs =abs(cwtfb.wt(ecgsignal));
            im = ind2rgb(im2uint8(rescale(cfs)),colormap);
            filenameindex=findx+k;
            filename=fullfile(folderpath,sprintf("%s_%d.jpg",lower(ecgtype),filenameindex));
            imwrite(imresize(im,[227 227]),filename);
            indx=indx+nol;
        end
        findx=findx+nol;
    end
end