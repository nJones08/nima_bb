clear 
close all

%% primary parameters
day     = 'P1';     % select T1, T3, P1, or P5 (the other file names are not tested yet)
type    = 'MVN';    % MVN or FMN
cntrl_N = '3';      % TOTAL number of control mice (set to 2 for just WT, set to 2+number of control [eg there is 1 MVN caspase control])
WT_ctlN = 2;        % how many WT controls are included

%% other important paramaters
cT      = 0.7;      % label confidence threshold
include = 'o';     % select 'on' to look at all mice, turn 'off' to leave out NiB001 
% (the mouse has no perturbation data ( thus we cant't tell when perturbation happens)                   
trial_portion = 'post';  % type 'pre' for before perturbation, and 'post' for after (angle will always be post[but we could change this easily]) 

%% angle analysis 
p       = 75;            % percentile of CI for angle analysis
mez     = 'mean angle'; % 'mean angle' for mean, 'med' angle for median, IN PROGRESS: 'min angle' to compute the minimum, 'max angle' if we want to find max angle

% in progress: angle point chooser
% vertex_point = '';
% principal_point = '';
% secondary_point = '';

%% set color order 
ord = [get(gca,'colororder');0,0,1;0,.5,0;1,0,0;0,.75,.75;.75,0,.75;.75,.75,0;.25,.25,.25];
% 'ord' variable default setting is set for current matlab default followed by old default,
% refer to table here:  http://math.loyola.edu/~loberbro/matlab/html/colorsInMatlab.html
% we use left column of table followed by right column of table
%% animal and session info 
if strcmpi(day,'T1') == 1
    day  = 'Training D1';
    spec = 'D1'
elseif strcmpi(day,'T2') == 1
    day  = 'Training D2';
    spec = 'D2'
elseif strcmpi(day,'T3') == 1
    day  = 'Training D3';
    spec = 'D3'    
elseif strcmpi(day,'P1') == 1
    day  = 'HBT D1';
    spec = 'DTe1'    
elseif strcmpi(day,'P2') == 1
    day  = 'HBT D2';
    spec = 'DTe2'
elseif strcmpi(day,'D3') == 1
    day  = 'HBT D3';
    spec = 'DTe3'
elseif strcmpi(day,'P4') == 1
    day  = 'HBT D4';
    spec = 'DTe4'    
elseif strcmpi(day,'P5') == 1
    day  = 'HBT D5';
    spec = 'DTe5'
end

if strcmpi(type,'MVN') == 1
    type = 'B.MVN-Caspase';
    ctrl = 'B.MVN-Control';
elseif strcmpi(type,'FMN') == 1
    type = 'B.FMN-Caspase';
    ctrl = 'B.FMN-Control';
end

storespec = spec;

%% assign mouse type to be analyzed
    MT_animals = {'NiB008M','NiB009M','NiB010M','NiB011M','Ni059','Ni063','Ni071M','Ni072M'};  % mvn test animals
    if strcmpi(day,'Training D1') == 1
        MT_animals = MT_animals(1:7);       %missing D1 from this mouse 'Ni072M'
    end
    MC_animals = {'Ni073M'};                                            %mvn control animals
    FT_animals = {'Ni067','Ni068'};                                     %fmn test animals % these animals can go in separate cohort? ==> 'Ni056','Ni057' since they are in a separate folder for HBT 5
    FC_animals = {'Ni069','Ni070'};                                     %fmn control animals
    WT_animals = {'NiB001','NiB005'};                                   % wt control animals
%     if strcmpi(day,'Training D1') == 1
%         FT_animals = FT_animals(1);         %only one trial for 'Ni068', so we drop from this analysis (variance will be 0, and skew data)
%     end
if strcmpi(include,'on') == 0                                   % we remove 'nib001' if include set to 'on'
    WT_anmials = WT_animals(2);
end

fprintf(['Day: ' num2str(day) '\n']);
    
mtsize = size(MT_animals,2);mcsize = size(MC_animals,2);        % number of animals depending on type 
ftsize = size(FT_animals,2);fcsize = size(FC_animals,2);

if strcmpi(type,'B.MVN-Caspase') == 1                           % sum for overall N to use later in SEM/plotting
    numanimals = mtsize + mcsize + 2;
    animals = [WT_animals,MC_animals,MT_animals];
elseif strcmpi(type,'B.FMN-Caspase') == 1
    numanimals = ftsize + fcsize + 2;
    animals = [WT_animals,FC_animals,FT_animals];
end

nmarker = numanimals +1;
% preallocate some arrays 
nt = [];
torder = [];

%% visualize # good trials per mouse
% we just plot the number of trials using a polar plot with relative sized
% dots at the end of the loop to visualize how many trials were contributed
% by each mouse in the current pool 
for a = 1 :nmarker
    R = mod(numanimals,2);
%     subplot(2,ceil(nmarker/2),a);
if a-nmarker==0 && R==0
        subplot(2,ceil(nmarker/2),[a a+1]);
        nt(nt==0) = NaN;
        th = linspace(0,2*pi,numanimals);
        sz = nt*200;
        r = linspace(0,10,numanimals);
        c = ord(1:size(th,2),:);
        pL=polarscatter(th,r,sz,c,'filled','MarkerFaceAlpha',.5);
        thetaticks([]);rticks([]);
        title('Trials per Mouse')
elseif a == nmarker && R==1
        subplot(2,ceil(nmarker/2),a);
        nt(nt==0) = NaN;
%         figure;
        th = linspace(0,2*pi,numanimals);
        sz = nt*200;
        r = linspace(0,10,numanimals);
        c = ord(1:size(th,2),:);
        pL =polarscatter(th,r,sz,c,'filled','MarkerFaceAlpha',.5);
        thetaticks([]);rticks([]);
        title('Trials per Mouse')
else
%% set up filename info
    nc = 0;
    animal_name = animals{a};
    if a <= WT_ctlN
        tp = 'WT-Control';
    elseif (str2double(cntrl_N) >= a) && (a > 2) 
        tp = ctrl;
    else
        tp = type;
    end
%% note: discuss consistent naming conventions w/ Nima
    if strcmpi(animal_name,'Ni071M') == 1 || strcmpi(animal_name,'Ni072M') ==1 || strcmpi(animal_name,'Ni073M') == 1 ...
            || strcmpi(animal_name,'Ni067') == 1 || strcmpi(animal_name,'Ni068') == 1 || strcmpi(animal_name,'Ni069') == 1 || strcmpi(animal_name,'Ni070') == 1 || strcmpi(animal_name,'Ni063') ==1
        if strcmpi(day,'HBT D5') == 1
            spec = 'Test D05';
        elseif strcmpi(day,'HBT D1') == 1
            spec = 'Test D01';
        elseif strcmpi(day,'Training D3') == 1
            spec = 'Training Day 3';
        elseif strcmpi(day,'T1') == 1
            spec = 'Training Day 1';
        end
    else
        spec = storespec; %keep original spec if it changes above, in case we need it later 
    end
    
    fprintf([animal_name '\n']);  
    
%% import coordinates and skeleton from Nima's deeplabcut results
% original path for Ni053(vidobj testing) %dis_pathone = ['D:\deep vids\Nima\'  animal_name filesep num2str(day)];
dis_pathone   = ['R:\Simon Chen Laboratory\Lab Members\Nima\Behaviour for DLC\' day ];
pturb_path    = ['R:\Simon Chen Laboratory\Lab Members\Nima\Behaviour for DLC\' day filesep tp '\' animal_name];

% s_files     = dir([dis_pathone filesep '*skeleton.csv']); % skeleton files
c_files     = dir([dis_pathone filesep 'Balance Control Training ' spec ' ' animal_name '*.csv']);  % coordinate files
p_files     = dir([pturb_path filesep '*.mat']);            % perturbation motor data files
% v_files     = dir([dis_pathone filesep '*.mp4']);         % video files 

%% trial order info (if we want to use perturbation time, we will need to order the trials correctly)
fnam_info = {c_files.name};
tnm_ind   = strfind(fnam_info,'DLC'); % DLC naming convention is such that .csv output files contain DLC, and due to nima's naming conventions, the characters immediately before DLC indicate trial number 
tRIAL     = [];
for c = 1:size(c_files,1)
    if fnam_info{c}(cell2mat(tnm_ind(c))-3) == '_'
        tRIAL     = fnam_info{c}(cell2mat(tnm_ind(c))-2:cell2mat(tnm_ind(c))-1);
    elseif fnam_info{c}(cell2mat(tnm_ind(c))-2) == '_'
        tRIAL     = fnam_info{c}(cell2mat(tnm_ind(c))-1);
    end
    torder = [torder;str2double(tRIAL)];% order of trial files 
end

%% perturbation info
 if strcmpi(animal_name,'Ni059')== 0 && strcmpi(animal_name,'NiB001')== 0
    P{a}               = load(fullfile(p_files.folder,p_files.name));    % trial information
    t_start            = {P{a}.data_trail.Start_Time};                   % trial start 
    p_start            = {P{a}.data_trail.Motor_Rotation_Time};          % perturbation time
    pturb_direction{a} = {P{a}.data_trail.Motor_Rotation};               % direction of perturbation
    if isfield(P{a},'ActualFR') ==1 % for some reason actual framerate is not always captured
        framrate{a}        = {P{a}.data_trail.ActualFR};                     % framerate in each trial (since its variable)
        trial_frnum{a}     = {P{a}.data_trail.framesaq};                     % frames per trial
    end
    t_dT = [];                                                          %#ok<NASGU>
%         if strcmpi(pturb_direction{a}{i},'none') == 0
    pt_dT        = datetime(p_start,'InputFormat','yyMMdd_HHmmss');
           if size(cell2mat(t_start(1)),2)> 16 % lol why are there different formats 
               t_dT     = datetime(t_start,'InputFormat','yyMMdd_HHmmss.SSS');
           else
               t_dT     = datetime(t_start,'InputFormat','yyMMdd_HHmmss');
           end    
    preP_elapsed{a} = seconds(pt_dT - t_dT); % time elapsed before perturbation
%        end 

    


 elseif strcmpi(animal_name,'NiB001')==1 && strcmpi(day,'T1')==1 || strcmpi(animal_name,'Ni059')==1
       preP_elapsed{a} = nan(1,size(c_files,1));
 end
 
 
 
 
 
 
    session_theta{a} = [];
    
%% prepare to loop through trials   
    thetA = [];
    pt_angles  = [];
    pt_measure    = [];
    angle_vars = [];
    angle_varmean =[];   
%% start loop
for i = 1: size(c_files,1)  %iterate through each trial for the animal
%     i = torder(ji);
    coords{i} = importdata([c_files(i).folder filesep c_files(i).name]);
    %if coords{i}.data(:,2)>0.8 %we could add confidence filter here,
    %consult with nima
    if strcmpi(animal_name,'NiB009M') == 1 && i == 5 && strcmpi(day,'HBT D5') == 1
        continue
    end    
    
    %% find low confidence frame indices for removal (overall points)
    low_All = unique([find(coords{i}.data(:,4)<cT);find(coords{i}.data(:,7)<cT);find(coords{i}.data(:,13)<cT);find(coords{i}.data(:,16)<cT);find(coords{i}.data(:,19)<cT)]);
    %[nose;nape;sacrum;hindL;hindR]
    %% beam start + end
    low_confbS  = find(coords{i}.data(:,20)<cT);
    low_confbE  = find(coords{i}.data(:,23)<cT);    
    %beam start
    beamS{i}.X  = coords{i}.data(:,20); %beamS{i}.X(low_confbS) = []; 
    beamS{i}.Y  = coords{i}.data(:,21); %beamS{i}.Y(low_confbS) = [];
    start_point = nanmean(beamS{i}.X);
    %beam end
    beamE{i}.X = coords{i}.data(:,23); %beamE{i}.X(low_confbE) = [];
    beamE{i}.Y = coords{i}.data(:,24); %beamE{i}.Y(low_confbE) = [];
    end_point  = mean(beamE{i}.X);
    %find midline 
    P1     = [mean(beamS{i}.X), mean(beamS{i}.Y)];
    P2     = [mean(beamE{i}.X), mean(beamE{i}.Y)];
    mid(i,1:2) = (P1(:) + P2(:)).'/2; 
    half_point = mid(1);
    
    %% nose
    nose{i}.X = coords{i}.data(:,2);  %#ok<*SAGROW>
    nose{i}.Y = coords{i}.data(:,3);
    lC_no     = find(coords{i}.data(:,4)<cT); 
    nose{i}.X(lC_no) = [];nose{i}.Y(lC_no) = [];
    %% nape
    nape{i}.X = coords{i}.data(:,5);
    nape{i}.Y = coords{i}.data(:,6);
    lC_na = find(coords{i}.data(:,7)<cT);
    nape{i}.X(lC_na) = [];nape{i}.Y(lC_na) = [];
    %% thorax
    thrx{i}.X = coords{i}.data(:,8);
    thrx{i}.Y = coords{i}.data(:,9);
    lC_tx     = find(coords{i}.data(:,10)<cT);
    thrx{i}.X(lC_tx) = [];thrx{i}.Y(lC_tx) = [];
    %% tailbase
    low_conf  = find(coords{i}.data(:,13)<cT);
    baseT{i}.X = coords{i}.data(:,11);  
    baseT{i}.Y = coords{i}.data(:,12);    
    %% remove low confidence points and pre beam-start points for tailbase
    baseT{i}.X(low_conf) = [];baseT{i}.Y(low_conf) = [];
    pre_beam_ind = find(baseT{i}.X < start_point); 
    baseT{i}.X(baseT{i}.X < start_point) = []; 
    baseT{i}.Y(pre_beam_ind) = [];
    
    rf = find(baseT{i}.X>end_point);
    baseT{i}.X(rf) = []; %rf is reflection points. DLC labels reflection of mouse in the cage, so we will cut all points past end of beam
    baseT{i}.Y(rf) = [];
    
    % separate coordinates into pre and post
    pre_ind = find(baseT{i}.X < half_point);
    pos_ind = find(baseT{i}.X > half_point); % & baseT{i}.X  600);
    if isempty(pos_ind) == 1
          %post 
        pos_displacement{a}(i) = nan;%max(pre_baseT{i}.Y) - min(pre_baseT{i}.Y);
        pos_line_variance{a}(i)= nan;%var(pre_baseT{i}.Y);
        pos_path_length{a}(i)  = nan;% sum(sqrt(sum(dpre.*dpre,2)))
        continue
    end
    pre_baseT{i}.X = baseT{i}.X(pre_ind); pre_baseT{i}.Y = baseT{i}.Y(pre_ind);
    pos_baseT{i}.X = baseT{i}.X(pos_ind); pos_baseT{i}.Y = baseT{i}.Y(pos_ind);
    if isempty(pre_ind) == 1
         %pre 
        pre_displacement{a}(i) = nan;%max(pre_baseT{i}.Y) - min(pre_baseT{i}.Y);
        pre_line_variance{a}(i)= nan;%var(pre_baseT{i}.Y);
%         dpre                   = 0%diff([pre_baseT{i}.X pre_baseT{i}.Y]);
        pre_path_length{a}(i)  = nan;% sum(sqrt(sum(dpre.*dpre,2)))
        continue
    end
    %% hind limbs
    %hindleft 
    hindL{i}.X = coords{i}.data(:,14);
    hindL{i}.Y = coords{i}.data(:,15);
    %hindright
    hindR{i}.X = coords{i}.data(:,17);
    hindR{i}.Y = coords{i}.data(:,18);
    % hind limb opposite from perturbation
    if strcmpi(animal_name,'Ni059')==0 && strcmpi(animal_name,'NiB001')== 0
        if strcmpi(day,'Training D1')==1 && strcmpi(day,'Training D3')==0 && strcmpi(pturb_direction{a}{i},'Right') == 1
            hindOP{i}.X = hindL{i}.X;
            hindOP{i}.Y = hindL{i}.Y;
            low_confH  = find(coords{i}.data(:,16)<cT);
        elseif strcmpi(day,'Training D1')==0 && strcmpi(day,'Training D3')==0 && strcmpi(pturb_direction{a}{i},'Left') == 1
            hindOP{i}.X = hindR{i}.X;
            hindOP{i}.Y = hindR{i}.Y;
            low_confH  = find(coords{i}.data(:,19)<cT);
        else 
               % randomly chose left or right hindlimb for no perturbation
               % trials
%         YourMatrix(randperm(msize, x))
            hindOP{i}.X = hindR{i}.X;
            hindOP{i}.Y = hindR{i}.Y;
            low_confH  = find(coords{i}.data(:,19)<cT);
        end    
        hindOP{i}.X(low_confH) = [];hindOP{i}.Y(low_confH) = [];
    end
    
%     if a>1 && (a==8)==0
%         hindOP{i}.X(low_All) = [];hindOP{i}.Y(low_All) = [];
%         pos_pturb_ind = find(hindOP{i}.X < half_point); 
%         hindOP{i}.X(hindOP{i}.X > half_point) = []; 
% %         hindOP{i}.Y(pre_pturb_ind) = [];
% %         nape{i}.X(pre_pturb_ind) = [];
% %         nape{i}.Y(pre_pturb_ind) = [];   
% %         nose{i}.X(pre_pturb_ind) = [];
% %         nose{i}.Y(pre_pturb_ind) = [];
%     end
    %% behavioral metrics
    %  calculate separate metrics 
    %% path length (need to add time component)
%     if strcmpi(animal_name,'Ni059')==0 
%         d = diff([baseT{i}.X baseT{i}.Y]);
% %         path_length{a}(i) = sum(sqrt(sum(d.*d,2)))/preP_elapsed{a}(i); % include p_start
%     else
%         path_length{a}(i) = 1;
%     end
    %% displacement 
    displacement{a}(i) = max(baseT{i}.Y) - min(baseT{i}.Y);    % simons wanted to see inclusive displacement idea
%     displacement{a}(i) = max(abs(baseT{i}.Y(1) - baseT{i}.Y(:))); % original
    %% variance
    line_variance{a}(i)= var(baseT{i}.Y,'omitnan');                          % variance
%     line_variance{a}(i)= nanvar(baseT{i}.Y)/abs(nanmean(baseT{i}.Y)); % relative variance 
    %% angle
    % make nose and nape equivalent matrices
    if strcmpi(animal_name,'Ni059')==0 && strcmpi(animal_name,'NiB001')== 0
%     nape{i}.X(low_All) = [];nape{i}.Y(low_All) = [];
    nape{i}.X(nape{i}.X> half_point) = [];nape{i}.Y(nape{i}.Y> half_point) = [];
%     nose{i}.X(low_All) = [];nose{i}.Y(low_All) = [];
    nose{i}.X(nose{i}.X> half_point) = [];nose{i}.Y(nose{i}.Y> half_point) = [];
%     hindOP{i}.X(low_All) = [];hindOP{i}.Y(low_All) = [];
    hindOP{i}.X(hindOP{i}.X> half_point) = [];hindOP{i}.Y(hindOP{i}.Y> half_point) = [];
    
    % some points have more frames because the sections of the body are not
    % always within the same quadrant of the screen, so we make them
    % equivalent matrices first in order to compare(we have to give up a
    % few frames to do this)
    mat_cut = min([size(hindOP{i}.X,1);size(thrx{i}.X,1);size(nose{i}.X,1);...
        size(hindOP{i}.Y,1);size(thrx{i}.Y,1);size(nose{i}.Y,1)]);
    % note: maybe angle of nose->nape->thorax would be good to try?
    hindOP{i}.X = hindOP{i}.X(1: mat_cut);hindOP{i}.Y = hindOP{i}.Y(1: mat_cut);
    nose{i}.X   = nose{i}.X(1:mat_cut);   nose{i}.Y  = nose{i}.Y(1:mat_cut);
%     nape{i}.X   = nape{i}.X(1:mat_cut);   nape{i}.Y  = nape{i}.Y(1:mat_cut);
    thrx{i}.X   = thrx{i}.X(1:mat_cut);   thrx{i}.Y  = thrx{i}.Y(1:mat_cut);
    beamE{i}.X  = beamE{i}.X(1:mat_cut);  beamE{i}.Y  = beamE{i}.Y(1:mat_cut);
    beamS{i}.X  = beamS{i}.X(1:mat_cut);  beamS{i}.Y  = beamS{i}.Y(1:mat_cut);
    %% set up points for angle computation
    % principal point
    p1X = nose{i}.X; %beamS{i}.X; 
    p1Y = nose{i}.Y; %beamS{i}.Y; 
    % second point (NOT MIDDLE POINT, THIS IS THE OTHER POINT THAT IS NOT
    % VERTEX)
    p2X = hindOP{i}.X;% thrx{i}.X;
    p2Y = hindOP{i}.Y;% thrx{i}.Y;
    % vertex point (THIS IS VERTEX, POINT WHERE YOU WANT THE ANGLE)
    vX = thrx{i}.X; % beamE{i}.X; %nape{i}.X;
    vY = thrx{i}.Y; % beamE{i}.Y; %nape{i}.Y;
    
    thetA{i} = atan2(abs((p1X-vX).*(p2Y-vY)-(p2X-vX).*(p1Y-vY)),...
    (p1X-vX).*(p2X-vX)+(p1Y-vY).*(p2Y-vX));
%     if  iscell(nape_theta{a}{i}) == 1
    pt_angles{i} = rad2deg(thetA{i});%*57.2958;    % angles in degrees
%     pt_angles{i}(pt_angles{i}<20) = [];
%     else
    if isempty(pt_angles{i})==0
        %% confidence interval of angles(method one, requires assumption of normal distribution)
        SEM_A = std(pt_angles{i})/sqrt(length(pt_angles{i}));     % SEM of angles
        tscrA = tinv([0.025  0.975],length(pt_angles{i})-1);      % T-Score
%         CIA = mean(pt_angles{i}) + tscrA*SEM_A;                    % Confidence Intervals
        %% CI computation (method two, uses percentile and does not require normal dist)
        ang = pt_angles{i};
        CIfn = @(x,p)prctile(x,abs([0,100]-(100-p)/2));
        CIA = CIfn(ang,p);
        angle_CI{i}      = CIA;
    
%         pt_angles{a}{i} = nape_theta{a}{i}*57.2958;   
    angle_vars{i}    = var(pt_angles{i},'omitnan'); %/abs(nanmean(pt_angles{i}));
    angle_varmean{i} = mean(angle_vars{i});
    
    % take average of angles in lower CI
        if strcmpi(mez,'mean angle') == 1 
            pt_measure(i)   = mean(ang(ang<CIA(1))); % here we take the mean of the angles in the lower CI
        elseif strcmpi(mez,'median angle') == 1
            pt_measure(i)   = median(ang(ang<CIA(1)));  % here we take the median of the angles in the lower CI
        elseif strcmpi(mez,'min angle') == 1
            pt_measure(i)   = min(pt_angles{i});
        elseif strcmpi(mez,'max angle') == 1
            pt_measure(i)   = max(pt_angles{i});%*57.2958);
        end
    end
    end
    %% pre 
    pre_displacement{a}(i) =  max(pre_baseT{i}.Y) - min(pre_baseT{i}.Y);
    pre_line_variance{a}(i)=  var(pre_baseT{i}.Y,'omitnan');                        %raw variance(matlab)
%     pre_line_variance{a}(i)= nanvar(pre_baseT{i}.Y)/abs(nanmean(pre_baseT{i}.Y)); %  relative
    %variance formula: sum((baseT{i}.Y-mean(baseT{i}.Y)).^2)/numanimals-1; %hand calculated
    dpre                      = diff([pre_baseT{i}.X pre_baseT{i}.Y]);
    pre_path_length{a}(i)     = sum(sqrt(sum(dpre.*dpre,2)));
    %% post 
    post_displacement{a}(i)  = max(pos_baseT{i}.Y) - min(pos_baseT{i}.Y);
%     post_line_variance{a}(i) = var(pos_baseT{i}.Y);
    post_line_variance{a}(i) = nanvar(pos_baseT{i}.Y)/abs(nanmean(pos_baseT{i}.Y));
    dpost                    = diff([pos_baseT{i}.X pos_baseT{i}.Y]);
    post_path_length{a}(i)   = sum(sqrt(sum(dpost.*dpost,2)));
    %% quick tailbase plot set up
    TailBase = subplot(2,ceil(nmarker/2),a);
    text(0.15,0.7,'No')
    text(0.25,0.5,'Good')
    text(0.4,0.3,'Trials')
    xticks([]);yticks([]);
    title(animal_name,'Color',ord(a,:))
    %% quick tailbase finish 
    x = linspace(1, size(baseT{i}.Y,1));
    if strcmpi(day,'HBT 5') == 1 || a <= str2double(cntrl_N)
        if strcmpi(animal_name,'NiB001')==1 && strcmpi(day,'Training D1')==1
            continue    
        else
            if strcmpi(trial_portion,'pre') == 1
                plot(pre_baseT{i}.X',pre_baseT{i}.Y','Color',uint8([128 128 128])); hold on                
            elseif strcmpi(trial_portion,'post') == 1
                plot(pos_baseT{i}.X',pos_baseT{i}.Y','Color',uint8([128 128 128])); hold on  
            elseif isempty(trial_portion) == 1
                plot(baseT{i}.X',baseT{i}.Y','Color',uint8([128 128 128])); hold on
            end
        end
    else
        if strcmpi(day,'Training D1')==1
            continue    
        else  
            if strcmpi(trial_portion,'pre') == 1
                plot(pre_baseT{i}.X',pre_baseT{i}.Y'); hold on
            elseif strcmpi(trial_portion,'post') == 1
                plot(pos_baseT{i}.X',pos_baseT{i}.Y'); hold on
            elseif isempty(trial_portion) == 1
                plot(baseT{i}.X',baseT{i}.Y'); hold on            
            end
        end
    end
    hold on

    if i == size(c_files,1)
        vL = nvline(half_point,{'r:','LineWidth',1.75},{'mid-line'}, [.15 .7], {'Rotation', 90});
%         vL = vline(mid(1),'r:','mid-line');
%         vL.LineWidth = 1.5;
    end
    
    if strcmpi(trial_portion,'pre') == 1
        xlim([0 half_point+50])
    else
        xlim([0 650])
    end
    
    ylim([180 420]);
    title(animals{a},'Color',ord(a,:))
    
%     if a==1 || a == ceil(nmarker/2)+1
        box off
%     else
%         set(gca, 'visible', 'off')
%         set(findall(gca, 'type', 'text'), 'visible', 'on')
%         xticklabels([]);xticks([]);
%         yticklabels([]);yticks([]);
%     end
    nc = nc+1;
end
     if strcmpi(animal_name,'Ni059')==0 
        session_theta{a} = pt_angles;
        session_mnT{a}   = nanmean(pt_measure); %cellfun(@nanmean,pt_measure);        
        session_medT{a}  = nanmedian(pt_measure);% cellfun(@nanmedian,pt_measure);
        session_minT{a}  = min(pt_measure);
        session_maxT{a}  = max(pt_measure);
        session_vars{a}  = angle_vars;
        session_vM{a}    = nanmedian(cell2mat(angle_varmean));
%     else
%         session_theta{a} = nan;
%         session_vars{a}  = nan;
    end
    nt(a) = nc;
%% unused code 
% skeletons (fix this to loop through num of trials found in folder, as with coordinates above)
%     skellz{1} = importdata([s_files(1).folder filesep s_files(1).name]);

% disfiles    = dir([dis_pathone filesep animal_name '_' num2str(day) '*.mat']); % used to point to mat file containing: video full filename, #frames acquired,time elapsed, and actual FR

% Calculate displacement by trial before perturbation (wobble)
% calculate displacement by trial after perturbation (spread/angle)
% first draft using Ni053, first vid (camera1_5DLC) is right perturbation
% second vid (camera1_9DLC) is left perturbation
% if i == 1
%     pturb = 'right';
% else
%     pturb = 'left' ; % define string 'left' or 'right' depending on trial perturnbation
% end
% % talk to nima about how videos are labeled so we may include the type of
% % perturbation in the video title for downstream 
% 
%     if strcmpi(pturb,'right') == 1
%         p2_temp = hindR;
%     elseif strcmpi(pturb,'left') == 1
%         p2_temp = hindL;
%     end
% get frames for each trial (and decide which ones are imoprtant
%      vidObj = VideoReader([v_files(i).folder filesep v_files(i).name]); %#ok<TNMLP>
%     numFrames = ceil(vidObj.FrameRate*vidObj.Duration);
%     % note, this number of  frames is unreliable as the videos raw are read
%     % as approximately 1 second for every three seconsd(i.e. 2.7-2.9x sped up; 9s vid --> 3.7s vid) 
%     % which results in a frame discrepancy
%     
%     for j = 1 : size(coords{i}.data,1)
% %         if coords{i}.data(j,13) > 0.9
% 
%     % mouse own body angle value
%         v  = [nape{i}.X(j), nape{i}.Y(j)];          % vertex of angle at nape
%         p2 = [p2_temp{i}.X(j), p2_temp{i}.Y(j)];    % point two at hind limb
%         p1 = [nose{i}.X(j), nose{i}.Y(j)];          % point one at nose
%         mouse_theta{i}(j)   = atan2(abs((p1(1)-v(1))*(p2(2)-v(2))-(p2(1)-v(1))*(p1(2)-v(2))),...
%             (p1(1)-v(1))*(p2(1)-v(1))+(p1(2)-v(2))*(p2(2)-v(1)));
%         
%     % nape from beam angle
%         bE = [beamE{i}.X(j), beamE{i}.Y(j)];        % vertex coord of beam end
%         bS = [beamS{i}.X(j), beamS{i}.Y(j)];        % coords of beam start
%         bNape_theta{i}(j)    = atan2(abs((bS(1)-bE(1))*(v(2)-bE(2))-(v(1)-bE(1))*(bS(2)-bE(2))),...
%             (bS(1)-bE(1))*(v(1)-bE(1))+(bS(2)-bE(2))*(v(2)-bE(1)));
%         
%     % thorax to hindlimbs
%         thx= [thrx{i}.X(j), thrx{i}.Y(j)];          % vertex of angle at thorax
%         tlb= [baseT{i}.X(j), baseT{i}.Y(j)];        % tailbase coords
%         hiL= [hindL{i}.X(j), hindL{i}.Y(j)];        % hindlimb coords L
%         hiR= [hindR{i}.X(j), hindR{i}.Y(j)];        % hindlimb coords R
%         
%         hindLx_theta{i}(j)    = atan2(abs((hiL(1)-thx(1))*(tlb(2)-thx(2))-(tlb(1)-thx(1))*(hiL(2)-thx(2))),...
%             (hiL(1)-thx(1))*(tlb(1)-thx(1))+(hiL(2)-thx(2))*(tlb(2)-thx(1)));
%  
%         hindLx_theta2{i}(j)   = atan2(abs(det([tlb-hiL;thx-hiL])),dot(tlb-hiL,thx-hiL))*180/pi;
%  
%         hindRx_theta{i}(j)    = atan2(abs((hiR(1)-thx(1))*(tlb(2)-thx(2))-(tlb(1)-thx(1))*(hiR(2)-thx(2))),...
%             (hiR(1)-thx(1))*(tlb(1)-thx(1))+(hiR(2)-thx(2))*(tlb(2)-thx(1)));
% 
% %     nape_displacement_overall = polyfit();
% %         end
%     end
% 
%     % displacement
%     y = nape{i}.Y;
%     z = beamE{i}.Y;
%     x = linspace(0,10,size(coords{i}.data,1))';
%     p = polyfit(x,z,7);
%     x1 = linspace(0,10,size(coords{i}.data,1));
%     y1 = polyval(p,beamE{i}.Y);
%     
%     displot = figure;
%     plot(x,smooth(y,'rloess'),'Color','r','LineStyle',':')
%     hold on
%     plot(x,smooth(z,'rloess'),'Color','r','LineStyle','--')
%     hold on
%     plot(x,smooth(nose{i}.Y,'rloess'),'Color','k')
%     hold on
%     plot(x,smooth(beamS{i}.Y,'rloess'),'Color','k','LineStyle','--')
%     hold on
%     plot(x,smooth(baseT{i}.Y,'rloess'),'Color','c','LineStyle','--')
%     legend('nape','beamEnd','nose','beamStart','tailbase','Location','southwest')
%     ylim([100 400])
%     xlim([3 9.5])
%     title('Trajectories')
%     movegui(displot,'north')
%     
%     %scatter
%     scatplot = figure;
%     scatter(nape{i}.X,nape{i}.Y,25,'m')
%     hold on
%     scatter(nose{i}.X,nose{i}.Y,25,'k')
%     hold on
%     scatter(baseT{i}.X,baseT{i}.Y,25,'c')
%     legend('nape','nose','tailbase')
%     ylim([100 400])
%     title('Coordinates')
%     movegui(scatplot,'south')
%     
%     %angle
%     thet = figure;plot(hindLx_theta2{i});
%     movegui(thet,'southeast')
end
movegui(TailBase,'center')

%% done looping throught trials, normalize if we want to
%normalize
% norm_displacement{a} = (displacement{a} - min(displacement{a}))/(max(displacement{a})-min(displacement{a}));
% Lnorm_variance{a}    = (line_variance{a} - min(line_variance{a}))/(max(line_variance{a}) - min(line_variance{a}));
end

%% tail base displacement set-up 
assdis = figure;
alldis = [];
predis = [];
posdis = [];
normdis= [];
allvar = [];
prevar = [];
posvar = [];
allpath= [];
prepath= [];
pospath= [];

%% take values by animal
    for m = 1:size(displacement,2)
        if isempty(pre_displacement{m}) == 1 || isempty(pre_line_variance{m}) == 1 
            pre_displacement{m} = NaN;
            pre_line_variance{m}= NaN;
        end
    alldis    = [alldis;max(displacement{m})]; %#ok<*AGROW>
    predis    = [predis;max(pre_displacement{m})];
    posdis    = [posdis;max(post_displacement{m})];
%     normdis   = [normdis;nanmean(norm_displacement{m})];%;nanmean(norm_displacement{2});nanmean(norm_displacement{3});nanmean(norm_displacement{4});nanmean(norm_displacement{5});nanmean(norm_displacement{6})];
    allvar    = [allvar;nanmean(line_variance{m})];%nanmean(line_variance{2});nanmean(line_variance{3});nanmean(line_variance{4});nanmean(line_variance{5});nanmean(line_variance{6});];
    prevar    = [prevar;nanmean(pre_line_variance{m})];%nanmean(pre_line_variance{2});nanmean(pre_line_variance{3});nanmean(pre_line_variance{4});nanmean(pre_line_variance{5});nanmean(pre_line_variance{6});];
    posvar    = [posvar;nanmean(post_line_variance{m})];%nanmean(post_line_variance{2});nanmean(post_line_variance{3});nanmean(post_line_variance{4});nanmean(post_line_variance{5});nanmean(post_line_variance{6});];
%     allpath   = [allpath;nanmean(path_length{m})];
    prepath   = [prepath;nanmean(pre_path_length{m})];
    pospath   = [pospath;nanmean(post_path_length{m})];
    end
% end

if strcmpi(trial_portion,'pre') == 1
    disview  = predis;
    varview  = prevar;
    pathview = prepath;
elseif strcmpi(trial_portion,'post') == 1
    varview = posvar;
    disview = posdis;
    pahtview= pospath;
elseif isempty(trial_portion) == 1
    varview = allvar;
    disview = alldis;
%     pathview= allpath;
end

%% displacement plot
D = bar(disview);hold on
% m = length(disview);
for k = str2double(cntrl_N)+1 : length(disview)
    i = mod(k-1,m);
    i = i +1;
    D2 = bar(k,disview(k));
    set(D2,'FaceColor',[0.8500    0.3250    0.0980]);
end
legendflex([D, D2],{'WT/Control',type},'anchor',{'nw','nw'},'box','off');
xlabel('Mice')
ylabel('Tailbase: Max Displacement(pixels)');
% ylim([0 300]);
title([day ':tailbase displacement, ' trial_portion]); %': max displacement'])
box off
[hd,pd]=ttest2(disview(1:str2num(cntrl_N)),disview(str2num(cntrl_N)+1:end))
DAVE = figure;
D3 = bar(1,nanmean(disview(1:str2double(cntrl_N)))); hold on;
SEM_WTD = nanstd(disview(1:str2double(cntrl_N)))/sqrt(str2double(cntrl_N)); er1 = errorbar(1,nanmean(disview(1:str2double(cntrl_N))),SEM_WTD);
er1.Color = [0 0 0];
er1.LineStyle = 'none'; er1.LineWidth = 2;
D4 = bar(2,nanmean(disview(str2double(cntrl_N)+1:end))); set(D4,'FaceColor',[0.8500    0.3250    0.0980]); hold on
SEM_BCD = nanstd(disview(str2double(cntrl_N)+1:end))/sqrt(size(animals,2)-str2double(cntrl_N));
er2 = errorbar(2,nanmean(disview(str2double(cntrl_N)+1:end)),SEM_BCD); er2.Color = [0 0 0]; er2.LineStyle = 'none';
er2.LineWidth = 2;
% ylim([0 200]);
legendflex([D3, D4],{'WT/Control',type},'anchor',{'nw','nw'},'box','off');
xlabel('Mouse type');xticklabels([]);
ylabel('Average Max Displacement(pixels)');
title([day ': tailbase displacement averaged'])
box off
D3.BarWidth = 0.5;
D4.BarWidth = 0.5;

%% variance
VARD = figure;
V = bar(varview); hold on
for k = str2double(cntrl_N)+1 : length(varview)
    i = mod(k-1,m);
    i = i +1;
    V2 = bar(k,varview(k));
    set(V2,'FaceColor',[0.8500    0.3250    0.0980]);
end
legendflex([V, V2],{'WT/Control',type},'anchor',{'nw','nw'},'box','off');
xlabel('Mice')
ylabel('Tailbase Variance: y-axis (px^2)')
title([day ': tail base variance, ' trial_portion]); %': variance']);
box off

[hv,pv]=ttest2(varview(1:str2num(cntrl_N)),varview(str2num(cntrl_N)+1:end))
VAVE = figure;
V3 = bar(1,nanmean(varview(1:str2double(cntrl_N)))); hold on;
SEM_WTV = nanstd(varview(1:str2double(cntrl_N)))/sqrt(str2double(cntrl_N)); er1 = errorbar(1,nanmean(varview(1:str2double(cntrl_N))),SEM_WTV);
er1.Color = [0 0 0];
er1.LineStyle = 'none'; er1.LineWidth = 2;
V4 = bar(2,nanmean(varview(str2double(cntrl_N)+1:end))); set(V4,'FaceColor',[0.8500    0.3250    0.0980]); hold on
SEM_BCV = nanstd(varview(str2double(cntrl_N)+1:end))/sqrt(size(animals,2)-str2double(cntrl_N));
er2 = errorbar(2,nanmean(varview(str2double(cntrl_N)+1:end)),SEM_BCV); er2.Color = [0 0 0]; er2.LineStyle = 'none';
er2.LineWidth = 2;
legendflex([V3, V4],{'WT/Control',type},'anchor',{'nw','nw'},'box','off');
xlabel('Mouse type');xticklabels([]);
ylabel('Average Variance (px^2)');
title([day ': tailbase variance averaged'])
box off
V3.BarWidth = 0.5;
V4.BarWidth = 0.5;

%% path length
% PARD = figure;
% p = bar(pathview); hold on
% for k = str2double(cntrl_N)+1 : length(pathview)
%     i = mod(k-1,m);
%     i = i +1;
%     p2 = bar(k,pathview(k));
%     set(p2,'FaceColor',[0.8500    0.3250    0.0980]);
% end
% legendflex([p, p2],{'WT/Control',type},'anchor',{'ne','ne'},'box','off');
% xlabel('Mice')
% ylabel('Tailbase Path Length: y-axis (px)')
% title([day ': path length, ' trial_portion]); %': variance']);
% box off
% 
% PAVE = figure;
% p3 = bar(1,nanmean(pathview(1:str2double(cntrl_N)))); hold on;
% SEM_WTP = nanstd(pathview(1:str2double(cntrl_N)))/sqrt(str2double(cntrl_N)); er1 = errorbar(1,nanmean(pathview(1:str2double(cntrl_N))),SEM_WTP);
% er1.Color = [0 0 0];
% er1.LineStyle = 'none'; er1.LineWidth = 2;
% p4 = bar(2,nanmean(pathview(str2double(cntrl_N)+1:end))); set(p4,'FaceColor',[0.8500    0.3250    0.0980]); hold on
% SEM_BCP = nanstd(pathview(str2double(cntrl_N)+1:end))/sqrt(size(animals,2)-str2double(cntrl_N));
% er2 = errorbar(2,nanmean(pathview(str2double(cntrl_N)+1:end)),SEM_BCP); er2.Color = [0 0 0]; er2.LineStyle = 'none';
% er2.LineWidth = 2;
% legendflex([p3, p4],{'WT/Control',type},'anchor',{'ne','ne'},'box','off');
% xlabel('Mouse type');xticklabels([]);
% ylabel('Average Path Length (px)'); ylim([0 1000])
% title([day ': path length averaged'])
% box off
% p3.BarWidth = 0.5;
% p4.BarWidth = 0.5;

movegui(D,'north')
movegui(DAVE,'south')
movegui(VARD,'northeast')
movegui(VAVE,'southeast')
% movegui(PARD,'northwest')
% movegui(PAVE,'southwest')

%% angle
session_mnT  = cell2mat(session_mnT);
session_medT  = cell2mat(session_medT);
session_vM    = cell2mat(session_vM);

%% metric chooser
if strcmpi(mez,'median angle') == 1
    session_metric= session_medT;
    smet = 'Median';
elseif strcmpi(mez,'mean angle') == 1
    session_metric= session_mnT;
    smet = 'Mean';
end

if iscell(session_metric) == 1
    if isempty(session_metric{1}) == 1
        session_metric{1} = NaN;
    elseif isempty(session_metric{2}) == 1
        session_metric{2} = NaN;
    end
    
    session_metric = cell2mat(session_metric);
end

%% begin plotting angles
S             = figure;

b = bar(session_metric); hold on

for k = str2double(cntrl_N)+1 : length(session_metric)
    i = mod(k-1,m);
    i = i +1;
    b2 = bar(k,session_metric(k));
    set(b2,'FaceColor',[0.8500    0.3250    0.0980]);
end

legendflex([b, b2],{'WT/Control',type},'anchor',{'n','n'},'box','off');
xlabel('Mice')
title([day ': mean angle per trial ']); %': variance']);
box off
movegui('northwest')
%% ylable marker 
if strcmpi(mez,'max angle') == 1
    ylabel('Maximum Angle (degrees)');
elseif strcmpi(mez,'min angle') == 1
    ylabel('Minimum Angle (degrees)');
elseif strcmpi(mez,'median angle') == 1
    ylabel('Median Angle (degrees)');
elseif strcmpi(mez,'mean angle') == 1
    ylabel('Mean Angle (degrees)');
end

title([day ': mean angle per trial ']); %': variance']);
box off
movegui('northwest')
%% next plot
% subplot(2,1,2)
figure;
b3 = bar(1,nanmean(session_metric(1:str2double(cntrl_N)))); hold on;
SEM_ANGW = nanstd(session_metric(1:str2double(cntrl_N)))/sqrt(str2double(cntrl_N)); 
era1 = errorbar(1,nanmean(session_metric(1:str2double(cntrl_N))),SEM_ANGW);
% if SEM_ANGW>0
    era1.Color = [0 0 0];
    era1.LineStyle = 'none'; era1.LineWidth = 2;
% end
b4 = bar(2,nanmean(session_metric(str2double(cntrl_N)+1:end))); set(b4,'FaceColor',[0.8500    0.3250    0.0980]); hold on
SEM_mANG = nanstd(session_metric(str2double(cntrl_N)+1:end))/sqrt(numanimals-str2double(cntrl_N)); 
era2 = errorbar(2,nanmean(session_metric(str2double(cntrl_N)+1:end)),SEM_mANG); era2.Color = [0 0 0]; 
era2.LineStyle = 'none'; era2.LineWidth = 2;
legendflex([b3, b4],{'WT/Control',type},'anchor',{'n','n'},'box','off');
xlabel('Mouse type');xticklabels([]);

title([day ': angles averaged (by mouse type)'])
box off
b3.BarWidth = 0.3;
b4.BarWidth = 0.3;

%% ylabel marker
if strcmpi(mez,'max angle') == 1
    ylabel('Ave Maximum Angle (degrees)');
elseif strcmpi(mez,'min angle') == 1
    ylabel('Ave Minimum Angle (degrees)');
elseif strcmpi(mez,'median angle') == 1
    ylabel('Median Angle (degrees)');
elseif strcmpi(mez,'mean angle') == 1
    ylabel('Mean Angle (degrees)');
end


% mA = session_medT;
% % r(r==0) = [];
% r = linspace(0,1,numanimals-1);
% sz = sz(1:size(mA,2));
% ord_v2 = [0 0 0;0 0 0;ord(3:end,:)];
% c = ord_v2(1:size(mA,2),:);
% meanAngles = figure;
% polarscatter(mA,r,sz,c,'filled','MarkerFaceAlpha',.5);
% thetalim([0 180]);rticks([]);

movegui('southwest')