%%
addpath('~/Repository/CommonFunctions/matplotlib/')
addpath('~/CoSMoMVPA/mvpa')

%% load data
load('../data/derivatives/RDM/stats_test_images.mat','res_all')

%% sort and plot
tv = res_all.a.fdim.values{1};

f=figure(1);clf
f.Position = [f.Position(1:2) 1300 700];

[~,idx]=sort(max(res_all.samples,[],2));
res = cosmo_slice(res_all,idx);


a=subplot(2,2,1);hold on
a.FontSize=16;
mu = 100*mean(res.samples);
se = 100*std(res.samples)./sqrt(size(res.samples,1));
plot(tv,50+0*tv,'k-')
fill([tv fliplr(tv)],[mu-se,fliplr(mu+se)],'k','FaceAlpha',.2,'LineStyle','none')
plot(tv,mu,'k','LineWidth',2)
xlim(tv([1 end]));
xlabel('time (ms)')
ylabel('pairwise decoding accuracy')
%title('pairwise decoding (200 validation images)')

a=subplot(2,2,3);hold on
a.FontSize=16;
plot(tv,0*tv,'k-')
fill([tv fliplr(tv)],[mean(noise_up.samples) fliplr(mean(noise_lo.samples))],...
    'k','FaceAlpha',.2,'LineStyle','none')
xlim(tv([1 end]));

xlabel('time (ms)')
ylabel('Spearman''s \rho')
%title('noise ceiling (200 validation images)')

a=subplot(1,2,2);hold on
a.FontSize=16;
co = plasma(size(res.samples,1)+10);
tv = res.a.fdim.values{1};
yv = .03*(1:size(res.samples,1));

for i=1:size(res.samples,1)
    plot(tv,yv(i)+.5+0*res.samples(i,:),'-','Color',.8*[1 1 1],'LineWidth',1)
end
for i=1:size(res.samples,1)
    
    plot(tv,yv(i)+res.samples(i,:),'Color',co(i,:),'LineWidth',1)
    
end
xlim(tv([1 end]));
ylim([.49 max(yv)+.65]);
a.YTick = yv+.5;
a.YTickLabel = arrayfun(@(x) sprintf('sub-%02i',x), res.sa.subject,'UniformOutput',0);
a.YAxis.FontSize=11;
xlabel('time (ms)')
%title('pairwise decoding per subject (200 validation images)')

annotation('textbox',[.08,.94,.4,.04],'String','A','FontSize',22,'LineStyle','none','FontWeight','bold')
annotation('textbox',[.08,.5,.4,.04],'String','C','FontSize',22,'LineStyle','none','FontWeight','bold')
annotation('textbox',[.51,.94,.4,.04],'String','B','FontSize',22,'LineStyle','none','FontWeight','bold')

%% save
fn = '../figures/figure_validation_images';
tn = tempname;
print(gcf,'-dpng','-r500',tn)
im=imread([tn '.png']);
[i,j]=find(mean(im,3)<255);margin=1;
imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');


