%%
clc
T = readtable('../data/participants.tsv','FileType','text');

mean_age = mean(T.age)
sd_age = std(T.age)
age_range = [min(T.age) max(T.age)]

n_female = sum(strcmpi(T.gender,'F'))
n_male = sum(strcmpi(T.gender,'M'))

n_native = sum(strcmpi(T.native_english,'Yes'))
n_non_native = sum(strcmpi(T.native_english,'No'))

n_monolingual = sum(strcmpi(T.language_profile,'Monolingual'))
n_bilingual = sum(strcmpi(T.language_profile,'Bilingual'))
n_trilingual = sum(strcmpi(T.language_profile,'Trilingual'))



