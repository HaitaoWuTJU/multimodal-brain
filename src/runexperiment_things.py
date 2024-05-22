#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 2019

@author: tgro5258
"""

from psychopy import core, event, visual, parallel, gui
import os,random,sys,math,json,requests
from glob import glob
import pandas as pd
import numpy as np

# debug things
debug_testsubject = 1
debug_usedummytriggers = 1
debug_windowedmode = 1
debug_save_screenshots = 0

objects = sorted(glob('stimuli/*'))
stimuli=[]
for e in objects:
    stimuli.append(sorted(glob(e+'/*.jpg'))[:12])
    
assert(len(objects)==1854)

if debug_testsubject:
    subjectnr = 0
else:
    # Get subject info
    subject_info = {'Subject number':''}
    if not gui.DlgFromDict(subject_info,title='Enter subject info:').OK:
        print('User hit cancel at subject information')
        exit()
    try:
        subjectnr = int(subject_info['Subject number'])
    except:
        raise

outfn = 'sub-%02i_task-rsvp_events.csv'%subjectnr
if not debug_testsubject and os.path.exists(outfn):
    raise Exception('%s exists'%outfn)

nobjects = len(objects)
nrepeats = 12 #repeats per stimulus
nstimpersequence = 309 #stim per sequences

random.seed(subjectnr)

refreshrate = 60
fixationduration = 1 - .5/refreshrate
stimduration = .05 - .5/refreshrate
isiduration = .1 - .5/refreshrate

trigger_stimon = 1
trigger_stimoff = 2
trigger_sequencestart = 3
trigger_duration = 0.010
trigger_port = 0xcff8

webhook_url='https://hooks.slack.com/services/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'

objectnumber=[]
for i in range(nrepeats):
    objectnumber += random.sample(range(nobjects),nobjects)

stimorders = [random.sample(range(12),12) for e in objects]
    
eventlist = pd.DataFrame(objectnumber,columns=['objectnumber'])
eventlist['object']=[os.path.split(objects[x])[-1] for x in eventlist['objectnumber']]
eventlist['sequencenumber'] = [math.floor(x/nstimpersequence) for x in range(len(eventlist))]
eventlist['presentationnumber'] = [x%nstimpersequence for x in range(len(eventlist))]
eventlist['blocksequencenumber'] = [math.floor(x/6) for x in eventlist['sequencenumber']]
eventlist['withinsequencenumber'] = [x%6 for x in eventlist['sequencenumber']]
eventlist['stimnumber'] = [stimorders[i][j] for i,j in zip(eventlist['objectnumber'],eventlist['blocksequencenumber'])]
eventlist['stim'] = [stimuli[i][j] for i,j in zip(eventlist['objectnumber'],eventlist['stimnumber'])]
eventlist['isteststim'] = [0 for x in eventlist['sequencenumber']]
eventlist['teststimnumber'] = [-1 for x in eventlist['sequencenumber']]

#add 10 sequences at the end with test images
with open('test_images.csv') as f:
    if sys.platform == 'win32':
        teststimuli = ['stimuli\\'+x.replace('/','\\') for x in f.read().splitlines()]
    else:
        teststimuli = ['stimuli/'+x for x in f.read().splitlines()]
nteststim = len(teststimuli)
assert(nteststim==200)

teststimnumber=[]
for i in range(nrepeats):
    teststimnumber += random.sample(range(nteststim),nteststim)
    
eventlisttest = pd.DataFrame(teststimnumber,columns=['teststimnumber'])
eventlisttest['isteststim'] = [1 for x in eventlisttest['teststimnumber']]
eventlisttest['sequencenumber'] = [72+math.floor(x/200) for x in range(len(eventlisttest))]
eventlisttest['presentationnumber'] = [x%200 for x in range(len(eventlisttest))]
eventlisttest['objectnumber']=[-1 for x in eventlisttest['teststimnumber']]
eventlisttest['object']=[-1 for x in eventlisttest['teststimnumber']]
eventlisttest['blocksequencenumber']=[-1 for x in eventlisttest['teststimnumber']]
eventlisttest['withinsequencenumber']=[-1 for x in eventlisttest['teststimnumber']]
eventlisttest['stimnumber']=[-1 for x in eventlisttest['teststimnumber']]
eventlisttest['stim'] = [teststimuli[i] for i in eventlisttest['teststimnumber']]

#combine sequences
eventlist = pd.concat([eventlist, eventlisttest],ignore_index=1)

#add targets
nsequences = eventlist['sequencenumber'].iloc[-1]+1
istarget=[]
for i in range(nsequences):
    ntargets = random.randint(2,6) #2 to 5 targets in a seq
    targetpos=[1, 1]
    nr = nstimpersequence if i<72 else 200
    t = [0 for x in range(nr)]
    while len(targetpos)>1 and any(np.diff(targetpos)<20):
        targetpos = sorted(random.sample(range(10,nr-10),ntargets))
    for p in targetpos:
        t[p]=1
    istarget += t
eventlist['istarget'] = istarget

def writeout(eventlist):
    with open(outfn,'w') as out:
        eventlist.to_csv(out,index_label='eventnumber')

writeout(eventlist)

# =============================================================================
# %% START
# =============================================================================
try:
    if debug_windowedmode:
        win=visual.Window([700,700],units='pix')
    else:
        win=visual.Window(units='pix',fullscr=True)
    mouse = event.Mouse(visible=False)

    fixation = visual.ImageStim(win, 'Fixation.png', size=20,
         name='fixation', autoLog=False)
    fixationtarget = visual.ImageStim(win, 'Fixationtarget.png', size=20,
         name='fixationtarget', autoLog=False)
    
    querytext = visual.TextStim(win,text='',pos=(0,200),name='querytext')
    progresstext = visual.TextStim(win,text='',pos=(0,100),name='progresstext')
    sequencestarttext = visual.TextStim(win,text='',pos=(0,50),name='sequencestarttext')

    filesep='/'
    if sys.platform == 'win32':
        filesep='\\'
        
    def check_abort(k):
        if k and k[0][0]=='q':
            raise Exception('User pressed q')
            
    screenshotnr = 0
    def take_screenshot(win):
        global screenshotnr 
        screenshotnr += 1
        win.getMovieFrame()
        win.saveMovieFrames('screenshots/screen_%05i.png'%screenshotnr)
                    
    def loadstimtex(stimname):
        return visual.ImageStim(win,stimname,size=375,name=stimname.split(filesep)[-1]);

    def send_dummy_trigger(trigger_value):
        core.wait(trigger_duration)
            
    def send_real_trigger(trigger_value):
        trigger_port.setData(trigger_value)
        core.wait(trigger_duration)
        trigger_port.setData(0)
    
    if debug_usedummytriggers:
        sendtrigger = send_dummy_trigger
    else:
        trigger_port = parallel.ParallelPort(address=trigger_port)
        trigger_port.setData(0)
        sendtrigger = send_real_trigger

    nevents = len(eventlist)
    sequencenumber = -1
    for eventnr in range(nevents):
        first = eventlist['sequencenumber'].iloc[eventnr]>sequencenumber
        if first: #start of sequence
            writeout(eventlist)
            
            sequencenumber = eventlist['sequencenumber'].iloc[eventnr]
            last_target = -99
            correct=0
            
            fixation.draw()
            win.flip()
            nstimthissequence = sum(eventlist['sequencenumber']==sequencenumber)
            stimtex = []
            for i in range(nstimthissequence):
                s = loadstimtex(eventlist['stim'].iloc[eventnr+i])
                stimtex.append(s)
                
            if sequencenumber:
                idx = eventlist['sequencenumber']==(sequencenumber-1)
                nt = sum(eventlist['istarget'][idx])
                nc = sum(eventlist['correct'][idx])
                slack_data={'text':'sub-%02i seq %i/%i (things) hits %i/%i <@tijlgrootswagers> <@amanda> <@U9C24ECQ7> <@UGGG7UD9S>'%(subjectnr,sequencenumber,nsequences,nc,nt),'channel':'#eeglab','username':'python'}
                if debug_testsubject:
                    print(slack_data)
                else:
                    try:
                        response = requests.post(webhook_url, data=json.dumps(slack_data),headers={'Content-Type': 'application/json'})
                    except:
                        pass
                
            progresstext.text = '%i / %i'%(1+sequencenumber,nsequences)
            progresstext.draw()
            sequencestarttext.text = 'Press any key to start the sequence'
            sequencestarttext.draw()
            fixation.draw()
            win.flip()
            k=event.waitKeys(keyList='afq', modifiers=False, timeStamped=True)
            check_abort(k)
            fixation.draw()
            time_fixon = win.flip()
            sendtrigger(trigger_sequencestart)
            while core.getTime() < time_fixon + fixationduration:pass
        
        response=0
        rt=0
        istarget = eventlist['istarget'].iloc[eventnr]
        stimnum = eventlist['presentationnumber'].iloc[eventnr]
        stim = stimtex[stimnum]
        stimname = stim.name
        stim.draw()
        if istarget:
            fixationtarget.draw()
        else:
            fixation.draw()
        time_stimon=win.flip()
        sendtrigger(trigger_stimon)
        if debug_save_screenshots:take_screenshot(win)
        
        if istarget:
            last_target=time_stimon
        
        while core.getTime() < time_stimon + stimduration:pass
        if istarget:
            fixationtarget.draw()
        else:
            fixation.draw()
        time_stimoff=win.flip()
        sendtrigger(trigger_stimoff)
        if debug_save_screenshots:take_screenshot(win)
        
        #get response
        correct=0
        k=event.getKeys(keyList='sdq', modifiers=False, timeStamped=True)
        if k:
            check_abort(k)
            response=1
            rt=k[0][1]
            correct = rt-last_target < 1
            
        eventlist.at[eventnr, 'stimname'] = stimname
        eventlist.at[eventnr, 'response'] = int(response)
        eventlist.at[eventnr, 'rt'] = rt-last_target if correct else 0
        eventlist.at[eventnr, 'correct'] = int(correct)
        eventlist.at[eventnr, 'time_stimon'] = time_stimon
        eventlist.at[eventnr, 'time_stimoff'] = time_stimoff
        eventlist.at[eventnr, 'stimdur'] = time_stimoff-time_stimon;
        while core.getTime() < time_stimon + isiduration:pass
            
finally:
    writeout(eventlist)
    print(str(sys.exc_info()))
    sequencestarttext.text='Experiment finished!'
    sequencestarttext.draw()
    win.flip()
    core.wait(1)
    win.close()
    exit()



