"""Readout x head learning rate x re-uploading at 300 episodes (160 runs).

The eight factorial cells x head lr 0.1 (skolik_protocol_1000, first 300 episodes) / 0.01 (head_lr_300), seeds 0-9.
Final-20 at episode 300. Per seed, the gain from the lower head lr, averaged over the two initial ranges, for each
readout and re-uploading level. Confirmatory, Holm over four: the lr x re-uploading interaction for Zi and for ZZ,
the three-way interaction (one-sample t), and equivalence of the readout gap at lr 0.01 with re-uploading within
+-50 (TOST). The greedy scores at episode 300 (results/greedy/) are described.
-> analysis/head_lr_300_reuploading.txt (JSON)
"""
import json
import os

from ..data import greedy as stored_greedy, output as output_file, path as result_path
from ..io import durable_write
from .head_lr_300_readout import contrast_summary

SEEDS = list(range(10))
BOUND = 50.0
MATCHES = {
    False: {'low': {'Zi':'QDQN-R_no-reup','ZZ':'QDQN-R_global_no-reup_lo-scale'},
            'high': {'Zi':'QDQN-R_no-reup_hi-scale','ZZ':'QDQN-R_global_no-reup'}},
    True: {'low': {'Zi':'QDQN-R_full','ZZ':'QDQN-R_global_lo-scale'},
           'high': {'Zi':'QDQN-R_full_hi-scale','ZZ':'QDQN-R_global-obs'}},
}


def load_histories():
    histories,missing = {},[]
    for seed in SEEDS:
        for reup,initials in MATCHES.items():
            for initial,cells in initials.items():
                for readout,cell in cells.items():
                    for lr in (0.1,0.01):
                        experiment = 'skolik_protocol_1000' if lr == 0.1 else 'head_lr_300'
                        path = result_path(experiment, f'{cell}_seed{seed}')
                        if not os.path.exists(path):
                            missing.append(path);continue
                        with open(path, encoding='utf-8') as stream:
                            data = json.load(stream)
                        assert data['label']==cell and data['seed']==seed
                        assert data['env']=='CartPole-v1' and data['protocol']=='SKP'
                        assert len(data['history'])>=300
                        if lr==0.01:
                            assert data['episodes']==300 and len(data['history'])==300
                            assert data['lr_classical']==0.01
                            if reup:
                                assert data['config']['reuploading'] is True
                                assert data['config']['readout']==readout
                                assert data['config']['training']['lr']==0.001
                        histories[seed,initial,readout,reup,lr]=data['history'][:300]
    if missing:
        raise RuntimeError('Analysis waits for all registered runs; missing %d completed files:\n%s' %
                           (len(missing),'\n'.join(missing)))
    assert len(histories)==160
    for seed in SEEDS:
        for reup in MATCHES:
            for initial in ('low','high'):
                for readout in ('Zi','ZZ'):
                    assert histories[seed,initial,readout,reup,0.01][0]==histories[seed,initial,readout,reup,0.1][0]
    return histories

def holm(pvalues):
    import numpy as np
    pvalues = np.asarray(pvalues,dtype=float)
    order = np.argsort(pvalues)
    result = np.empty(len(pvalues))
    highest = 0.0
    for rank,index in enumerate(order):
        highest=max(highest,min(1.0,(len(pvalues)-rank)*pvalues[index]))
        result[index]=highest
    return result.tolist()

def summary(values,confidence=0.95):
    return contrast_summary(values,confidence)

def zero_p(values):
    import numpy as np
    from scipy import stats
    values=np.asarray(values,dtype=float)
    return float(stats.ttest_1samp(values,0).pvalue) if values.std(ddof=1)>0 else (1.0 if values.mean()==0 else 0.0)

def primary_statistics(scores):
    import numpy as np
    from scipy import stats
    gains={readout:{reup:np.array([np.mean([scores[seed,initial,readout,reup,0.01]-scores[seed,initial,readout,reup,0.1]
                                           for initial in ('low','high')]) for seed in SEEDS])
                     for reup in MATCHES} for readout in ('Zi','ZZ')}
    contrasts={'lr_reup_interaction_Zi':gains['Zi'][True]-gains['Zi'][False],
               'lr_reup_interaction_ZZ':gains['ZZ'][True]-gains['ZZ'][False]}
    contrasts['three_way_interaction']=contrasts['lr_reup_interaction_ZZ']-contrasts['lr_reup_interaction_Zi']
    low_gap=np.array([np.mean([scores[seed,initial,'ZZ',True,0.01]-scores[seed,initial,'Zi',True,0.01]
                               for initial in ('low','high')]) for seed in SEEDS])
    se=low_gap.std(ddof=1)/np.sqrt(len(low_gap))
    if se==0:
        lower_p=0.0 if low_gap.mean()>-BOUND else 1.0
        upper_p=0.0 if low_gap.mean()<BOUND else 1.0
    else:
        lower_p=float(stats.t.sf((low_gap.mean()+BOUND)/se,len(low_gap)-1))
        upper_p=float(stats.t.cdf((low_gap.mean()-BOUND)/se,len(low_gap)-1))
    raw=[zero_p(value) for value in contrasts.values()]+[max(lower_p,upper_p)]
    adjusted=holm(raw)
    result={name:dict(summary(values),p_raw=raw[index],p_holm=adjusted[index],
                      shapiro_p_descriptive=float(stats.shapiro(values).pvalue) if values.std()>0 else None,
                      wilcoxon_p_descriptive=float(stats.wilcoxon(values).pvalue) if np.any(values) else 1.0)
            for index,(name,values) in enumerate(contrasts.items())}
    result.update({'lr_gains':{readout:{'on' if reup else 'off':summary(values) for reup,values in group.items()}
                               for readout,group in gains.items()},
                   'readout_gap_low_lr_reup_on':summary(low_gap),
                   'low_gap_ci90':summary(low_gap,0.90)['ci'],
                   'equivalence_bounds':[-BOUND,BOUND],
                   'tost_p_lower':lower_p,'tost_p_upper':upper_p,'tost_p_raw':raw[-1],
                   'tost_p_holm':adjusted[-1],'equivalent_within_50_reup_on':adjusted[-1]<0.05,
                   'seed_contrasts':[{'seed':seed,**{name:float(values[index]) for name,values in contrasts.items()},
                                      'readout_gap_low_lr_reup_on':float(low_gap[index])}
                                     for index,seed in enumerate(SEEDS)]})
    return result

def greedy_scores(keys):
    import numpy as np
    cache=stored_greedy('head_lr_300_reuploading')   # "seed|range|readout|reup|lr" -> greedy returns at episode 300
    return {key:float(np.mean(cache['|'.join(map(str,key))])) for key in keys}

def main(evaluate_greedy=True):
    import numpy as np
    histories=load_histories()
    scores={key:float(np.mean(history[280:300])) for key,history in histories.items()}
    table=[]
    for reup in MATCHES:
        for initial in ('low','high'):
            for readout in ('Zi','ZZ'):
                for lr in (0.1,0.01):
                    chosen=[histories[seed,initial,readout,reup,lr] for seed in SEEDS]
                    final=[np.mean(history[-20:]) for history in chosen]
                    table.append({'reuploading':reup,'initial_range':initial,'readout':readout,'head_lr':lr,
                                  'n':10,'final20_mean':float(np.mean(final)),'final20_sd':float(np.std(final,ddof=1)),
                                  'collapse':sum(bool(np.mean(h[-20:])<0.5*np.mean(h[100:120])) for h in chosen),
                                  'auc_mean':float(np.mean([np.mean(h) for h in chosen]))})
    result={'scope':'SKP, 300 episodes; reuploading extension after seeing full Zi no-reup results',
            'primary_endpoint':'mean return episodes 281..300','registered_observations':160,
            'reproducibility_episode0':'80/80 same-cell/seed comparisons',
            'descriptives':table,'primary':primary_statistics(scores),'greedy_evaluated':evaluate_greedy}
    if evaluate_greedy:
        greedy=greedy_scores(list(histories))
        result['greedy_secondary_descriptive']={
            f'{readout}|reup={reup}|lr={lr}':float(np.mean([greedy[seed,initial,readout,reup,lr]
                                                        for seed in SEEDS for initial in ('low','high')]))
            for readout in ('Zi','ZZ') for reup in MATCHES for lr in (0.1,0.01)}
    output=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    durable_write(output_file('head_lr_300_reuploading.txt'),lambda stream:stream.write(output),'w')
    print(output,flush=True)

