function nproc = artemis_setup()

    if ismac
        if isempty(which('cosmo_wtf'))
            addpath('~/CoSMoMVPA/mvpa')
        end
        addpath('~/fieldtrip');ft_defaults;
        nproc = 1;
    else %on HPC
        addpath('../../CoSMoMVPA/mvpa');
        addpath('../../fieldtrip');ft_defaults;
        % start cluster, give it a unique directory
        % starting a pool can fail when 2 procs are requesting simultaneous
        % thus try again after a second until success
        pool=[];
        while isempty(pool) 
            try
                pc = parcluster('local');
                pc.JobStorageLocation=tempdir;
                pool=parpool(pc);
            catch err
                disp(err)
                delete(gcp('nocreate'));
                pause(1)
            end
        end
        nproc=cosmo_parallel_get_nproc_available();
    end