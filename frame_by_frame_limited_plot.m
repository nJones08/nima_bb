% plot trial + averaged angles
nAnimals = 2; %animals to plot the angles
Ani      = [4 7];% array of animals Id by number out of all animals (in this order) i.e. [wt,experimental,+control] 
tot_nAni = size(session_theta,2); % total animal number
[ani1,ani2] = session_theta{Ani};
allAni_angl_nf_t = [size(ani1,2),size(ani2,2)]; % animal angles number of trials
a1=cell2mat(cellfun(@size,ani1,'uni',false));a1=a1(a1>1);
a2=cell2mat(cellfun(@size,ani2,'uni',false));a2=a2(a2>1);
allAni_thetaf    = [a1,a2];% animal angles number of frames per trial

% plot quick
figure;
for i1 = 1:allAni_angl_nf_t(1)
    subplot(allAni_angl_nf_t(1),1,i1);
    plot(cell2mat(ani1(i1)))
%     hold on
%     pause(2)
end

xlim([0 max(a1)])
ylabel('angle at thorax (degrees)')

figure;
for i2 = 1:allAni_angl_nf_t(2)
    subplot(allAni_angl_nf_t(2),1,2);
    plot(cell2mat(ani2(i2)))
%     hold on    
%     pause(2)

end

xlim([0 max(a2)])
ylabel('angle at thorax')
xlabel('frames')


