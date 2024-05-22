%%
f=figure(1);clf
f.Position = [f.Position(1:2) 800 900];f.Resize='off';

% panel A: images
rng(2);
fns = dir('../public_domain_images/stim*.png');
fns = randsample(fns,numel(fns));
IM={};
for f=1:numel(fns)
    x=im2double(imread(fullfile(fns(f).folder,fns(f).name)));
    IM{end+1}=x;
    IM{end+1}=.5+0.*x;
end

[fix,~,fixa] = imread('../Fixation.png');

a = axes('Position',[.01 .5 .5 .45]);
montage(randsample(IM(1:2:end),numel(fns)),'Size',[6,4],'BorderSize',10,'BackgroundColor','w')


a = axes('Position',[.51 .5 .45 .45]);
imshow('../eeg_setup.png')

a = axes('Position',[.1 .1 .85 .4]);
aw=1;
for i=14:-1:1
    image('XData',i*1.2+[-aw aw],'YData',i*.3+[-aw aw],'CData',flipud(IM{i}))
    if i<14
        text(i*1.2+aw,i*.3-aw,' 50 ms','FontSize',12)
    end
    image('XData',i*1.2+.1*[-aw aw],'YData',i*.3+.1*[-aw aw],'CData',fix,'AlphaData',fixa)
end
axis equal
axis off

annotation('textbox',[.05,.95,.4,.04],'String','A   Example images','FontSize',20,'LineStyle','none','FontWeight','bold')
annotation('textbox',[.5,.95,.4,.04],'String','B   EEG setup','FontSize',20,'LineStyle','none','FontWeight','bold')
annotation('textbox',[.05,.4,.4,.04],'String','C   RSVP design','FontSize',20,'LineStyle','none','FontWeight','bold')


%% save
fn = '../figures/figure_design';
tn = tempname;
print(gcf,'-dpng','-r500',tn)
im=imread([tn '.png']);
[i,j]=find(mean(im,3)<255);margin=2;
imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');



