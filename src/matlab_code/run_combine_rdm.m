function run_combine_rdm(subjectnr)

    %% init pool
    nproc = artemis_setup();
    
    %% load
    fprintf('Loading RDM...\n')
    res_cell=cell(1,10);
    parfor b=1:10
        tic;
        fn = sprintf('../data/derivatives/RDM/RDM_full_blocks/sub-%02i_b%02i_rdm.mat',subjectnr,b);
        x=load(fn,'res');
        res_cell{b} = x.res;
        fprintf('Loading RDM_full_blocks/sub-%02i_b%02i_rdm finished in %.2fs\n',subjectnr,b,toc)
    end
    fprintf('Stacking RDM...');tic
    RDM_all = cosmo_stack(res_cell);
    fprintf('Finished in %.2fs\n',toc)
    
    %% unflatten
    fprintf('Unflatten RDM...\n');
    X = cosmo_unflatten(RDM_all);
    timevec = RDM_all.a.fdim.values{1};
    RDM = zeros(numel(timevec),1854,1854);
    cc=clock();mm='';
    for t=1:numel(timevec)
        RDM(t,:,:) = squareform(X(:,t));
        mm=cosmo_show_progress(cc,t/numel(timevec),sprintf('%i/%i',t,numel(timevec)),mm);
    end
    
    %% save
    fprintf('Saving RDM...');tic
    
    outfn = sprintf('../data/derivatives/RDM/sub-%02i_RDM_full.mat',subjectnr);
    save(outfn,'RDM','timevec','-v7.3');
    fprintf('Finished in %.2fs\n',toc)
    
    
    
    
