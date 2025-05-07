%Program to create CWT image database from ECG singls
load ("ECGData.mat");
data= ECGData.Data;
labels= ECGData.Labels;

ARR= data(1:30,:); %Taken first 30 samples
CHF= data(97:126,:);
NSR= data(127:156,:);

signallength = 500;

fb = cwtfilterbank("SignalLength", signallength, "Wavelet", "amor", "VoicesPerOctave", 12);

mkdir ("ecgdataset");
mkdir ("ecgdataset/arr");
mkdir ("ecgdataset/chf");
mkdir ("ecgdataset/nsr");

ecgtype=["ARR", "CHF", "NSR"];

ecg2cwtscg(ARR, fb, ecgtype{1});
ecg2cwtscg(CHF, fb, ecgtype{2});
ecg2cwtscg(NSR, fb, ecgtype{3});