"""Seven-biomolecule mean-spectrum matrices with explicit seeded RNA/lipid selection."""
from pathlib import Path
from dataclasses import asdict
from importlib.metadata import version
import json
import platform
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from threadpoolctl import threadpool_limits
import spectral_diagnostics as sd
import raw_lipid_rna as lr

LABELS = ('Adenine', 'DL-alanine', 'DL-phenylalanine', 'DL-tyrosine', 'Glutamic acid', 'RNA', 'Lipid')
OLD_PERCENTILES = {('Renishaw','RNA'):85,('Renishaw','Lipid'):92,('Horiba','RNA'):60,('Horiba','Lipid'):65}

def select_top_counts(counts, n, seed, key):
    """Keep all counts above cutoff; uniformly sample cutoff ties without replacement."""
    counts = np.asarray(counts)
    if counts.ndim != 1 or n < 1 or len(counts) < n:
        raise ValueError(f'{key}: need at least {n} distinct spectra; found {len(counts)}')
    cutoff = int(np.partition(counts,len(counts)-n)[len(counts)-n])
    above, tied = np.flatnonzero(counts>cutoff), np.flatnonzero(counts==cutoff)
    take = n-len(above)
    chosen_ties = tied[sd.stable_indices(len(tied),take,seed,f'peak-count-ties/{key}')]
    indices = np.sort(np.r_[above,chosen_ties])
    assert len(indices)==n and len(np.unique(indices))==n
    return indices, cutoff, len(above), len(tied), take

def plot_matrix_grid(matrices, metric, out, panels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = panels or (('Renishaw','Renishaw'),('Horiba','Horiba'),('Renishaw','Horiba'))
    fig,axes=plt.subplots(1,len(panels),figsize=(6.4*len(panels),6.4),layout='constrained',squeeze=False)
    vmin,vmax=(-1.,1.) if metric=='pearson_r' else (0.,max(matrices[p]['rmse'].max() for p in matrices))
    cmap=plt.get_cmap('RdBu_r' if metric=='pearson_r' else 'Blues')
    for ax,(a,b) in zip(axes[0],panels):
        values=matrices[a,b][metric]
        im=ax.imshow(values,vmin=vmin,vmax=vmax,cmap=cmap,interpolation='nearest')
        ax.set(xticks=range(7),yticks=range(7),xticklabels=LABELS,yticklabels=LABELS,
               xlabel=f'{b} molecule',ylabel=f'{a} molecule',title=f'{a} × {b}')
        plt.setp(ax.get_xticklabels(),rotation=45,ha='right',fontsize=9)
        plt.setp(ax.get_yticklabels(),fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        for i in range(7):
            for j in range(7):
                rgba=cmap((values[i,j]-vmin)/(vmax-vmin))
                brightness=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if brightness<.52 else '#111111')
    fig.colorbar(im,ax=axes[0].tolist(),shrink=.72,label='Pearson r (higher is closer)' if metric=='pearson_r' else 'RMSE in normalized intensity (lower is closer)')
    subtitle = '750 spectra per molecule/instrument · ASLS → min–max · 451–1450 cm⁻¹' if len(panels)>1 else '750 spectra per group · ASLS → min–max\n451–1450 cm⁻¹ · seed 42'
    fig.suptitle(('Pearson correlation' if metric=='pearson_r' else 'Spectral RMSE')+' of molecule mean spectra\n'+subtitle,fontsize=14 if len(panels)>1 else 12)
    stem=metric+('_cross_instrument' if len(panels)==1 else '_matrices')
    fig.savefig(out/f'{stem}.png',dpi=180)
    fig.savefig(out/f'{stem}.pdf')
    plt.close(fig)

def run(data_dir, manifest_path, downloads_dir, output_dir, seed=42, n=750):
    data_dir,manifest_path,downloads_dir,out=map(Path,(data_dir,manifest_path,downloads_dir,output_dir))
    out.mkdir(parents=True,exist_ok=True)
    cfg=sd.DiagnosticConfig(seed=seed,max_per_class=n,low=451,high=1450)
    grid=np.arange(451,1451,dtype=float)
    loaded=lr.load_raw_lipid_rna(manifest_path,grid,downloads_dir=downloads_dir,seed=seed,max_per_class=None)
    data={}; records=[]; selections=[]; hashes=list(loaded['inputs']); all_filter_rows=[]
    # Analysis never extrapolates. Reproduce the original filter grid separately:
    # 450 lies below native support and np.interp holds its endpoint, as in lipid_rna.py.
    filter_grid=np.arange(450,1451,dtype=float)
    raw_counts={}
    for inv in loaded['inventory'].to_dict('records'):
        path=downloads_dir/inv['file']; parsed=lr.read_raw_export(path,inv['instrument'])
        if lr.file_hash(path)!=inv['sha256']: raise ValueError(f'Input changed: {path.name}')
        filtering=np.stack([np.interp(filter_grid,parsed['wavenumbers'],row) for row in parsed['spectra']])
        smooth=savgol_filter(filtering,11,3,axis=1)
        raw_counts[inv['file']]=np.array([len(find_peaks(row,prominence=30,distance=25)[0]) for row in smooth])
    for instrument in sd.INSTRUMENTS:
        for molecule in sd.CLASSES:
            key=(instrument,molecule)
            if molecule in ('RNA','Lipid'):
                meta=loaded['rows'].loc[(loaded['rows'].instrument==instrument)&(loaded['rows']['class']==molecule)].reset_index(drop=True).copy()
                counts=np.array([raw_counts[f][r] for f,r in zip(meta.file,meta.source_row_zero_based)])
                indices,cutoff,above,ties,taken=select_top_counts(counts,n,seed,f'{instrument}/{molecule}')
                old_threshold=float(np.percentile(counts,OLD_PERCENTILES[key]))
                selected=np.zeros(len(meta),dtype=bool); selected[indices]=True
                assert np.all(selected[counts>old_threshold]), 'Old accepted spectra must be retained'
                meta['peak_count']=counts; meta['selected']=selected
                meta['old_threshold']=old_threshold; meta['passed_original_filter']=counts>old_threshold
                all_filter_rows.append(meta)
                records.append(meta.iloc[indices].copy())
                data[key]=loaded['data'][key][indices]
                selections.append({'instrument':instrument,'class':molecule,'available':len(meta),'selected':len(indices),'method':'top_peak_count','peak_cutoff':cutoff,'strictly_above_cutoff':above,'tied_at_cutoff':ties,'ties_selected':taken,'original_filter_kept':int((counts>old_threshold).sum()),'additional_rows':n-int((counts>old_threshold).sum())})
            else:
                path=data_dir/f'{molecule}_{instrument.lower()}.txt'; digest=sd.file_hash(path)
                wn,X=sd.read_amino_file(path,instrument)
                if sd.file_hash(path)!=digest: raise ValueError(f'Input changed: {path.name}')
                if len(X)<n or not np.isfinite(X).all(): raise ValueError(f'{path.name}: insufficient or invalid rows')
                indices=sd.stable_indices(len(X),n,seed,f'{instrument}/{molecule}')
                data[key]=sd.interpolate_checked(wn,X[indices],grid,path.name)
                hashes.append({'name':path.name,'sha256':digest,'bytes':path.stat().st_size})
                records.append(pd.DataFrame({'instrument':instrument,'class':molecule,'file':path.name,'source_row_zero_based':indices,'selected':True,'source_id':[f'{path.name}::spectrum:{r}' for r in indices]}))
                selections.append({'instrument':instrument,'class':molecule,'available':len(X),'selected':n,'method':'seeded_random'})
    selected_rows=pd.concat(records,ignore_index=True)
    assert len(selected_rows)==14*n and not selected_rows.source_id.duplicated().any()
    selected_rows.to_csv(out/'selected_rows.csv',index=False)
    selection=pd.DataFrame(selections); selection.to_csv(out/'selection_summary.csv',index=False)
    pd.concat(all_filter_rows,ignore_index=True).to_csv(out/'rna_lipid_all_peak_counts.csv',index=False)
    loaded['inventory'].to_csv(out/'raw_file_inventory.csv',index=False)
    with threadpool_limits(limits=1):
        views,audit=sd.preprocess(data,cfg)
        processed=views['asls_minmax']
        means,sds=sd.moments(processed)
        matrices={}
        for a,b in [('Renishaw','Renishaw'),('Horiba','Horiba'),('Renishaw','Horiba')]:
            matrices[a,b]={'pearson_r':sd.correlation_matrix(means[a],means[b]),'rmse':sd.rmse_matrix(means[a],means[b])}
        # Full 14x14 exports allow every within-/between-instrument comparison.
        full=np.concatenate([means[i] for i in sd.INSTRUMENTS]); labels=[f'{i} | {m}' for i in sd.INSTRUMENTS for m in LABELS]
        full_r,full_rm=sd.correlation_matrix(full,full),sd.rmse_matrix(full,full)
    assert np.allclose(full_r,full_r.T) and np.allclose(np.diag(full_r),1)
    assert np.array_equal(full_rm,full_rm.T) and np.all(np.diag(full_rm)==0)
    assert np.isfinite(full_r).all() and np.isfinite(full_rm).all()
    audit.to_csv(out/'preprocessing_audit.csv',index=False)
    pd.DataFrame(full_r,index=labels,columns=labels).to_csv(out/'pearson_r_all14.csv')
    pd.DataFrame(full_rm,index=labels,columns=labels).to_csv(out/'rmse_all14.csv')
    for (a,b),values in matrices.items():
        for metric,M in values.items(): pd.DataFrame(M,index=LABELS,columns=LABELS).to_csv(out/f'{metric}_{a}_vs_{b}.csv')
    cross=matrices['Renishaw','Horiba']
    matches=[]
    for j,label in enumerate(LABELS):
        matches.append({'class':label,'same_molecule_pearson_r':cross['pearson_r'][j,j],'same_molecule_rmse':cross['rmse'][j,j],'closest_Horiba_by_correlation':LABELS[np.argmax(cross['pearson_r'][j])],'closest_Horiba_by_RMSE':LABELS[np.argmin(cross['rmse'][j])]})
    matched=pd.DataFrame(matches); matched.to_csv(out/'cross_instrument_summary.csv',index=False)
    arrays={}
    for i in sd.INSTRUMENTS:
        arrays[f'{i}_means']=means[i]; arrays[f'{i}_sd']=sds[i]
        for m in sd.CLASSES: arrays[f'{i}_{m}_processed']=processed[i,m]
    arrays['grid']=grid; arrays['pearson_r_all14']=full_r; arrays['rmse_all14']=full_rm
    np.savez_compressed(out/'numerical_results.npz',**arrays)
    for metric in ('pearson_r','rmse'):
        plot_matrix_grid(matrices,metric,out)
        plot_matrix_grid(matrices,metric,out,panels=[('Renishaw','Horiba')])
    run_manifest={'configuration':asdict(cfg),'selection':'RNA/Lipid top 750 peak counts; seeded cutoff ties; other molecules seeded random 750','filter':{'grid':'450–1450; endpoint hold at 450 reproduces original selection method','SG_window':11,'SG_order':3,'prominence':30,'distance':25},'analysis':'interpolate 451–1450, ASLS once, per-spectrum min–max once, then mean per class/instrument','input_files':hashes,'source_manifest_sha256':sd.file_hash(manifest_path),'code_sha256':sd.file_hash(__file__),'diagnostics_sha256':sd.file_hash(sd.__file__),'raw_loader_sha256':sd.file_hash(lr.__file__),'versions':{p:version(p) for p in ['numpy','pandas','scipy','pybaselines','matplotlib','threadpoolctl']},'python':platform.python_version(),'array_hashes':{k:sd.array_hash(v) for k,v in arrays.items()}}
    (out/'reproducibility.json').write_text(json.dumps(run_manifest,indent=2)+'\n')
    return {'output_dir':out,'counts':selection,'preprocessing_audit':audit,'matches':matched,'means':means,'matrices':matrices,'array_hashes':run_manifest['array_hashes']}
