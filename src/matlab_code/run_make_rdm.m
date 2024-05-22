function run_make_rdm(varargin)
    
    %% This code makes the full 275x1854x1854 RDMs for one subject
    % It involves looping over all possible pairs of concepts in THINGS.
    % To make this faster, the pairs are divided in 10 blocks which are run
    % separately on a high performance computer (artemis). Within each of
    % these blocks, sub-blocks of pairs are also done in parallel to
    % maximise cpu usage. Using 10 separate HPC jobs with 12 cores and 
    % 50GB RAM each, the RDM for one subject took ~35 hours

    %% init pool
    nproc = artemis_setup();
    
    %% get params
    defaults = struct();
    defaults.subject=1;
    defaults.block=1;
    opt = cosmo_structjoin(defaults,varargin);
    subjectnr = opt.subject;
    blocknr = opt.block;
    
    %% load data
    fn = sprintf('../data/derivatives/cosmomvpa/sub-%02i_task-rsvp_cosmomvpa.mat',subjectnr);
    outfn = sprintf('../data/derivatives/RDM/RDM_full_blocks/sub-%02i_b%02i_rdm.mat',subjectnr,blocknr);
    fprintf('loading %s\n',fn);tic
    load(fn,'ds')
    fprintf('loading data finished in %i seconds\n',ceil(toc))
    if subjectnr>1
        ds = cosmo_slice(ds,~ds.sa.isteststim,1);
    end
    
    %% decode things: set up combs
    ds.sa.targets = 1+ds.sa.objectnumber; %(add one for matlab)
    ds.sa.chunks = 1+ds.sa.blocksequencenumber; %(add one for matlab)
    nh = cosmo_interval_neighborhood(ds,'time','radius',0);
    
    %% all pairwise combinations
    ut = unique(ds.sa.targets);
    combs = combnk(ut,2);
    % split into blocks
    step = ceil(length(combs)/10);
    s = 1:step:length(combs);
    blocks = cell(length(s),1);
    for b = 1:length(s)
        blocks{b} = combs(s(b):min(s(b)+step-1,length(combs)),:);
    end
    
    % grab to the block to run here
	combs = blocks{blocknr};
    
    % all chunks to leave out
    uc = unique(ds.sa.chunks);
        
    %% create RDM
    % find the items belonging to the exemplars
    target_idx = cell(1,length(ut));
    for j=1:length(ut)
        target_idx{j} = find(ds.sa.targets==ut(j));
    end
    % for each chunk, find items belonging to the test set
    test_chunk_idx = cell(1,length(uc));
    for j=1:length(uc)
        test_chunk_idx{j} = find(ds.sa.chunks==uc(j));
    end
    
    %% make blocks for parfor loop
    step = ceil(length(combs)/nproc);
    s = 1:step:length(combs);
    comb_blocks = cell(1,length(s));
    for b = 1:nproc
        comb_blocks{b} = combs(s(b):min(s(b)+step-1,length(combs)),:);
    end
    
    %arguments for searchlight and crossvalidation
    ma = struct();
    ma.classifier = @cosmo_classify_lda;
    ma.output = 'accuracy';
    ma.check_partitions = false;
    ma.nproc = 1;
    ma.progress = 0;
    ma.partitions = struct();

    % set options for each worker process
    worker_opt_cell = cell(1,nproc);
    for p=1:nproc
        worker_opt=struct();
        worker_opt.ds=ds;
        worker_opt.ma = ma;
        worker_opt.uc = uc;
        worker_opt.worker_id=p;
        worker_opt.nproc=nproc;
        worker_opt.nh=nh;
        worker_opt.combs = comb_blocks{p};
        worker_opt.target_idx = target_idx;
        worker_opt.test_chunk_idx = test_chunk_idx;
        worker_opt_cell{p}=worker_opt;
    end
    %% run the workers
    result_map_cell=cosmo_parcellfun(nproc,@run_block_with_worker,...
                                    worker_opt_cell,'UniformOutput',false);
    %% cat the results
    res=cosmo_stack(result_map_cell);
    res.sa.target1stim = ds.sa.stim(res.sa.target1);
    res.sa.target2stim = ds.sa.stim(res.sa.target2);
    
    %% save
    fprintf('Saving...');tic
    save(outfn,'res','-v7.3')
    fprintf('Saving finished in %i seconds\n',ceil(toc))
end

function res_block = run_block_with_worker(worker_opt)
    ds=worker_opt.ds;
    nh=worker_opt.nh;
    ma=worker_opt.ma;
    uc=worker_opt.uc;
    target_idx=worker_opt.target_idx;
    test_chunk_idx=worker_opt.test_chunk_idx;
    worker_id=worker_opt.worker_id;
    nproc=worker_opt.nproc;
    combs=worker_opt.combs;
    res_cell = cell(1,length(combs));
    cc=clock();mm='';
    for i=1:length(combs)
        idx_ex = [target_idx{combs(i,1)}; target_idx{combs(i,2)}];
        [ma.partitions.train_indices,ma.partitions.test_indices] = deal(cell(1,length(uc)));
        for j=1:length(uc)
            ma.partitions.train_indices{j} = setdiff(idx_ex,test_chunk_idx{j});
            ma.partitions.test_indices{j} = intersect(test_chunk_idx{j},idx_ex);
        end
        res_cell{i} = cosmo_searchlight(ds,nh,@cosmo_crossvalidation_measure,ma);
        res_cell{i}.sa.target1 = combs(i,1);
        res_cell{i}.sa.target2 = combs(i,2);
        if ~mod(i,10)
            mm=cosmo_show_progress(cc,i/length(combs),sprintf('%i/%i for worker %i/%i\n',i,length(combs),worker_id,nproc),mm);
        end
    end
    res_block = cosmo_stack(res_cell);
end