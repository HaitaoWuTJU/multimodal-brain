%%
addpath('~/Repository/CommonFunctions/matplotlib/')
addpath('~/CoSMoMVPA/mvpa')

%% load data
load('../data/derivatives/RDM/stats_RDM_full.mat','mean_RDM','timevec')

%% subset to a timewindow
mean_RDM_t = squeeze(mean(mean_RDM(timevec>=180 & timevec<220,:,:)));

%% load THINGS table to make 4 example models
T = readtable('../object_concepts.xlsx');

models = {};modelnames={};
models{end+1} = squareform(pdist(T.Category_manual__Animal_excl_Human_==1,'euclidean'));
modelnames{end+1} = 'Animal';
models{end+1} = squareform(pdist(T.Category_manual__Clothing==1,'euclidean'));
modelnames{end+1} = 'Clothing';
models{end+1} = squareform(pdist(T.Category_manual__Food_Drink==1,'euclidean'));
modelnames{end+1} = 'Food/Drink';
models{end+1} = squareform(pdist(T.Category_manual__Natural_incl_EdibleItems_==1,'euclidean'));
modelnames{end+1} = 'Natural';

idx = find(tril(ones(1854),-1));
for m=1:numel(models)
    modelvec(:,m) = models{m}(idx);
end  
fprintf('correlating models ');tic
modelcorr = corr(modelvec,mean_RDM(:,idx)','type','Spearman');
fprintf('finished in %.2fs\n',toc)

%% plot
tv = timevec;

f=figure(1);clf
f.Position = [f.Position(1:2) 700 900];

f.Resize='off';

a=axes('Position',[.1 .29 .85 .7]);
a.FontSize=12;

X = mean_RDM_t;
X(:) = tiedrank(X(:));
X = 100*X./numel(X);

[sc,idx]=sort(sum([T.Category_manual__Natural_incl_EdibleItems_==1 ...
    10*(T.Category_manual__Food_Drink==1) ...
    100*(T.Category_manual__Animal_excl_Human_==1) ...
    1000*(T.Category_manual__Clothing==1) ...
],2));

X = X(idx,idx);

imagesc(X,[0 100]);a.FontSize = 14;hold on
c=colorbar();c.Label.String='dissimilarity (percentile)';
axis square
colormap inferno
a.YDir = 'normal';
xlabel('concept #')
ylabel('concept #')
idx2 = (-15:15)+round(sum(sc<1));
f=fill(idx2([1 end end 1]),idx2([end end 1 1]),'b','FaceColor','none',...
    'EdgeColor','c','LineWidth',1.5);

a=axes('Position',[.1 .05 .2 .2]);
imagesc(idx2,idx2,X(idx2,idx2),[0 100]);a.FontSize = 14;hold on
axis square
colormap inferno
a.YDir = 'normal';
f=fill(idx2([1 end end 1]),idx2([end end 1 1]),'b','FaceColor','none',...
    'EdgeColor','c','LineWidth',7);
a.XTick=idx2(2:end-1);
a.XTickLabel = T.Word(idx(a.XTick));
a.XTickLabelRotation=90;
a.XAxis.FontSize=6;
a.YTick=a.XTick;
a.YTickLabel=a.XTickLabel;
a.YAxis.FontSize=a.XAxis.FontSize;

a=axes('Position',[.39 .05 .25 .18]);
a.FontSize = 12;hold on
mu = 100*mean(subject_mean_accuracy);
se = 100*std(subject_mean_accuracy)./sqrt(size(subject_mean_accuracy,1));
plot(tv,50+0*tv,'k-')
fill([tv fliplr(tv)],[mu-se,fliplr(mu+se)],'k','FaceAlpha',.2,'LineStyle','none')
plot(tv,mu,'k','LineWidth',2)
xlim(tv([1 end]));ylim([49.8,51.6]);
a.YTick = [50 51];
xlabel('time (ms)')
ylabel('decoding accuracy')

a=axes('Position',[.73 .05 .25 .18]);
a.ColorOrder=tab10();hold on
a.FontSize = 12;
h=plot(tv,modelcorr,'LineWidth',1);
plot(tv,0*tv,'k-')
xlim([-100,600]);ylim([-.03 .12])
a.YTick = [0 .1];
xlabel('time (ms)')
ylabel('model correlation (\rho)')
leg=legend(h,modelnames);leg.Position = [.85 .17 .14 .07];

annotation('textbox',[.01,.92,.4,.05],'String','A','FontSize',22,'LineStyle','none','FontWeight','bold')
annotation('textbox',[.01,.21,.4,.05],'String','B','FontSize',22,'LineStyle','none','FontWeight','bold')
annotation('textbox',[.32,.21,.4,.05],'String','C','FontSize',22,'LineStyle','none','FontWeight','bold')
annotation('textbox',[.66,.21,.4,.05],'String','D','FontSize',22,'LineStyle','none','FontWeight','bold')


%% save
fn = '../figures/figure_full_RDM';
tn = tempname;
print(gcf,'-dpng','-r500',tn)
im=imread([tn '.png']);
[i,j]=find(mean(im,3)<255);margin=1;
imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');


