%%
nproc = artemis_setup();

%% load data
totalsubs = 50;
res_cell = cell(1,totalsubs);
cc=clock();mm='';
fprintf('\nLoading data\n')
for s=1:totalsubs
    fn = sprintf('../data/derivatives/RDM/sub-%02i_rdm_test_images.mat',s);
    try
        x=load(fn,'res');
        x.res.sa.subject = s+0*x.res.sa.target1;
        res_cell{s} = x.res;
    catch
    end
    mm=cosmo_show_progress(cc,s/totalsubs,sprintf('%i/%i',s,totalsubs),mm);
end
fprintf('Finished\n')

%% stack
fprintf('Stacking data\n')
rdm_all = cosmo_stack(res_cell(~cellfun(@isempty,res_cell)));
res_all = cosmo_fx(rdm_all,@mean,{'subject'});
rdm_mean = cosmo_fx(rdm_all,@mean,{'target1','target2'});
res_cell = [];
fprintf('Finished\n')

%% noise ceiling
fprintf('Computing noise ceiling\n')
noise_up_cell = cosmo_split(res_all,{'subject'});
noise_lo_cell = cosmo_split(res_all,{'subject'});
rdm_mean_left_out_cell = cell(size(noise_up_cell));
corrtype = 'Spearman';
cc=clock();mm='';
for s=1:numel(noise_up_cell)
    rdm_sub = cosmo_slice(rdm_all,rdm_all.sa.subject==noise_up_cell{s}.sa.subject);
    r_slice = cosmo_slice(rdm_all,rdm_all.sa.subject~=noise_up_cell{s}.sa.subject);
    rdm_mean_left_out = cosmo_fx(r_slice,@mean,{'target1','target2'});
    
    noise_up = zeros(1,275);
    noise_lo = zeros(1,275);
    for t=1:275
        noise_up(t) = corr(rdm_sub.samples(:,t),rdm_mean.samples(:,t),'type',corrtype);
        noise_lo(t) = corr(rdm_sub.samples(:,t),rdm_mean_left_out.samples(:,t),'type',corrtype);
    end
    noise_up_cell{s}.samples = noise_up;
    noise_lo_cell{s}.samples = noise_lo;
    r_slice = [];
    rdm_sub = [];
    mm=cosmo_show_progress(cc,s/numel(noise_up_cell),sprintf('%i/%i',s,numel(noise_up_cell)),mm);
end
noise_up = cosmo_stack(noise_up_cell);
noise_lo = cosmo_stack(noise_lo_cell);
fprintf('Finished\n')

%%
fprintf('Saving\n')
save('../data/derivatives/RDM/stats_test_images.mat','res_all','rdm_mean','noise_up','noise_lo','-v7.3');
fprintf('Finished\n')



