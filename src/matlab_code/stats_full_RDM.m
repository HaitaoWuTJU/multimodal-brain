%%
nproc = artemis_setup();

%% load data
totalsubs = 50;
subject_mean_accuracy = [];
sum_RDM = zeros(275,1854,1854);
loweridx = find(tril(ones(1854),-1));
cc=clock();mm='';n=0;
fprintf('\nLoading data\n')
for s=1:totalsubs
    fn = sprintf('../data/derivatives/RDM/sub-%02i_RDM_full.mat',s);
    try
        x=load(fn);
        n = n+1
        subject_mean_accuracy(n,:) = mean(x.RDM(:,loweridx),2);
        sum_RDM = sum_RDM+x.RDM;
        timevec = x.timevec;
    catch
    end
    mm=cosmo_show_progress(cc,s/totalsubs,sprintf('%i/%i',s,totalsubs),mm);
end
mean_RDM = sum_RDM./n;
fprintf('Finished\n')

%% stack
fprintf('Saving\n')
save('../data/derivatives/RDM/stats_RDM_full.mat','mean_RDM','subject_mean_accuracy','timevec','-v7.3');
fprintf('Finished\n')