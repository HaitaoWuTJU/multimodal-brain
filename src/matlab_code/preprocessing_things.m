function preprocessing_things(partid)

    %% add eeglab and cosmo on HPC
    if ~ismac
        addpath('../../CoSMoMVPA/mvpa')
        addpath('../../eeglab')
    else
        addpath('~/CoSMoMVPA/mvpa')
        addpath('~/Dropbox-usyd/Matlabtoolboxes/eeglab')
    end
    eeglab
    
    %% get files
    datapath = '../data';
    mkdir(sprintf('%s/derivatives/eeglab',datapath));
    mkdir(sprintf('%s/derivatives/cosmomvpa',datapath));
    
    contfn = sprintf('%s/derivatives/eeglab/sub-%02i_task-rsvp_continuous.set',datapath,partid);
    if isfile(contfn)
        fprintf('Using %s\n',contfn)
    	EEG_cont = pop_loadset(contfn);
    else
        % load EEG file
        EEG_raw = pop_loadbv(sprintf('%s/sub-%02i/eeg/',datapath,partid), sprintf('sub-%02i_task-rsvp_eeg.vhdr',partid));
        EEG_raw = eeg_checkset(EEG_raw);
        EEG_raw.setname = partid;
        EEG_raw = eeg_checkset(EEG_raw);

        % re-reference
        if ismember(partid,[49 50]) %these were recorded with a 128 workspace and different ref, so remove the extra channels
            EEG_raw = pop_select(EEG_raw,'channel',1:63);
            EEG_raw = pop_chanedit(EEG_raw, 'append',1,'changefield',{2 'labels' 'FCz'},'setref',{'' 'FCz'});
            EEG_raw = pop_reref(EEG_raw, [],'refloc',struct('labels',{'FCz'},'type',{''},'theta',{0},'radius',{0.1278},'X',{0.3907},'Y',{0},'Z',{0.9205},'sph_theta',{0},'sph_phi',{67},'sph_radius',{1},'urchan',{[]},'ref',{''},'datachan',{0}));
        else
            EEG_raw = pop_chanedit(EEG_raw, 'append',1,'changefield',{2 'labels' 'Cz'},'setref',{'' 'Cz'});
            EEG_raw = pop_reref(EEG_raw, [],'refloc',struct('labels',{'Cz'},'type',{''},'theta',{0},'radius',{0},'X',{0},'Y',{0},'Z',{85},'sph_theta',{0},'sph_phi',{90},'sph_radius',{85},'urchan',{[]},'ref',{''},'datachan',{0}));
        end

        % high pass filter
        EEG_raw = pop_eegfiltnew(EEG_raw, 0.1,[]);

        % low pass filter
        EEG_raw = pop_eegfiltnew(EEG_raw, [],100);

        % downsample
        EEG_cont = pop_resample(EEG_raw, 250);
        EEG_cont = eeg_checkset(EEG_cont);
        
        % save
        pop_saveset(EEG_cont,contfn);
    end
    
    %% add eventinfo to events
    eventsfncsv = sprintf('%s/sub-%02i/eeg/sub-%02i_task-rsvp_events.csv',datapath,partid,partid);
    eventsfntsv = strrep(eventsfncsv,'.csv','.tsv');
    eventlist = readtable(eventsfncsv);
    
    idx = find(strcmp({EEG_cont.event.type},'E  1'));
    onset = vertcat(EEG_cont.event(idx).latency)*4-3; % adjust onset times for the downsampling
    duration = 50*ones(size(onset)); %stim was oon for 50 msst

    neweventlist = [table(onset,duration,'VariableNames',{'onset','duration'}) eventlist(1:numel(onset),:)];
    
    writetable(neweventlist,eventsfntsv,'filetype','text','Delimiter','\t')
    
    %% create epochs
    EEG_epoch = pop_epoch(EEG_cont, {'E  1'}, [-0.100 1.000]);
    EEG_epoch = eeg_checkset(EEG_epoch);
    
    %% convert to cosmo
    ds = cosmo_flatten(permute(EEG_epoch.data,[3 1 2]),{'chan','time'},{{EEG_epoch.chanlocs.labels},EEG_epoch.times},2);
    ds.a.meeg=struct(); %or cosmo thinks it's not a meeg ds 
    ds.sa = table2struct(eventlist,'ToScalar',true);
    cosmo_check_dataset(ds,'meeg');
    
    %% save epochs    
    fprintf('Saving.\n');
    save(sprintf('%s/derivatives/cosmomvpa/sub-%02i_task-rsvp_cosmomvpa.mat',datapath,partid),'ds','-v7.3')
    fprintf('Finished.\n');
end
