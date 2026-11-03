"""Plot every stored forcing curve for all events, without selecting flood times."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    (root/'events').mkdir()
    config=json.loads(Path('results/stage2_multilead_local/20261023_fold0/seed_45/run_config.json').read_text())
    split={event:part for part,events in config['split_events'].items() for event in events}
    with Path('processed_data/timestamp_stage1/m1_events/events.csv').open() as handle:
        events=sorted(csv.DictReader(handle),key=lambda row:row['event_id'])
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    summaries=[]; sources=[]; arrays=[]
    with PdfPages(root/'all_events_300_locations.pdf') as pdf:
        for row in events:
            path=Path(row['path_to_X_event']); values=np.load(path)
            if values.shape != (480,300) or not np.isfinite(values).all():
                raise ValueError(f'Invalid forcing array: {path}')
            if values.min()<0:raise ValueError('Negative forcing needs a revised display scale')
            hours=.5+.5*np.arange(values.shape[0]);event=row['event_id']; arrays.append((event,values))
            median=np.median(values,axis=1);p95=np.percentile(values,95,axis=1);maximum=values.max(axis=1)
            peak=float(values.max()); early=hours<=24
            loc_peak=values.max(axis=0)
            relative=np.divide(values,loc_peak[None,:],out=np.zeros_like(values),where=loc_peak[None,:]>0)
            cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#cccccc')
            relative[:,loc_peak==0]=np.nan
            fig,axes=plt.subplots(3,1,figsize=(13,10),layout='constrained',height_ratios=[1,1,1.25])
            for ax,indices,title in [(axes[0],np.ones(len(hours),dtype=bool),'Full event: all 300 location curves'),
                                     (axes[1],early,'First 24 hours: all 300 location curves (independent vertical scale)')]:
                ax.plot(hours[indices],values[indices],color='#397caf',alpha=.15,lw=.45,rasterized=True)
                ax.plot(hours[indices],maximum[indices],color='#d15b22',lw=1.2,label='Maximum across locations')
                ax.plot(hours[indices],p95[indices],color='#823f98',lw=1.3,label='95th percentile across locations')
                ax.plot(hours[indices],median[indices],color='#161616',lw=1.3,label='Median across locations')
                ax.set(title=title,ylabel='Source discharge (cfs*)',xlabel='Event-relative time (hours)',
                       xlim=(0,24 if ax is axes[1] else 240),ylim=(0,None))
                ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8,ncol=3)
            image=axes[2].imshow(np.ma.masked_invalid(relative.T),origin='lower',aspect='auto',
                                extent=[.25,240.25,.5,300.5],cmap=cmap,norm=Normalize(0,1),interpolation='nearest')
            axes[2].set(title='Every location separately: forcing divided by that location’s event maximum',
                        xlabel='Event-relative time (hours)',ylabel='Location ID (Loc1–Loc300)',xlim=(0,240),
                        yticks=[1,50,100,150,200,250,300])
            fig.colorbar(image,ax=axes[2],label='Fraction of each location’s peak (not flood classification)',pad=.01)
            fig.suptitle(f'{event} · {split[event]} · 300 inflow-location series ×480 half-hour samples',fontsize=15)
            fig.supxlabel('*cfs is the source-file unit label; simulator conversion is not established. Raw stored forcing, before model standardization.\n'
                          'No event/time filtering. Gray heatmap rows: all-zero series. Nonzero forcing does not by itself establish flooding.',fontsize=9)
            fig.savefig(root/'events'/f'{event}_all_300_locations.png',dpi=150)
            pdf.savefig(fig,dpi=130);plt.close(fig)
            sources.append(dict(event_id=event,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            summary=dict(event_id=event,split=split[event],time_samples=480,locations=300,
                         all_zero_locations=int((loc_peak==0).sum()),peak_source_cfs=peak,
                         initial_max_source_cfs=float(maximum[0]),peak_max_across_locations_hour=float(hours[maximum.argmax()]),
                         first6h_peak_fraction=float(values[hours<=6].max()/peak) if peak else 0,
                         first12h_peak_fraction=float(values[hours<=12].max()/peak) if peak else 0,
                         first24h_peak_fraction=float(values[early].max()/peak) if peak else 0)
            summaries.append(summary)
            with (root/'events'/f'{event}_location_summary.csv').open('w') as handle:
                writer=csv.writer(handle);writer.writerow(['location_id','initial_source_cfs','peak_source_cfs','peak_hour'])
                for i in range(300):writer.writerow([i+1,float(values[0,i]),float(loc_peak[i]),float(hours[values[:,i].argmax()])])
            print(f'Plotted {event} ({split[event]})',flush=True)
    with (root/'event_summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    with PdfPages(root/'all_events_overview.pdf') as pdf:
        for early_only in [False,True]:
            fig,axes=plt.subplots(8,5,figsize=(20,22),layout='constrained')
            for ax,(event,v) in zip(axes.flat,arrays):
                hours=.5+.5*np.arange(len(v));idx=hours<=24 if early_only else np.ones(len(v),dtype=bool)
                peak=float(v.max()); scale=peak if peak else 1
                ax.plot(hours[idx],v[idx].max(axis=1)/scale,color='#d15b22',lw=1)
                ax.plot(hours[idx],np.percentile(v[idx],95,axis=1)/scale,color='#823f98',lw=1)
                ax.plot(hours[idx],np.median(v[idx],axis=1)/scale,color='#161616',lw=1)
                ax.set(title=f'{event} · {split[event]}',xlim=(0,24 if early_only else 240),ylim=(0,1.03))
                ax.grid(alpha=.2)
            title='First 24 hours' if early_only else 'Full 240-hour events'
            fig.suptitle(f'{title}: all 40 events · max (orange), spatial p95 (purple), median (black)\n'
                         'Each event divided by its own full-event maximum; compare timing, not physical amplitudes',fontsize=16)
            fig.supxlabel('Event-relative time (hours)');fig.supylabel('Fraction of full-event maximum')
            stem='overview_first24h' if early_only else 'overview_full240h'
            fig.savefig(root/f'{stem}.png',dpi=130);pdf.savefig(fig,dpi=130);plt.close(fig)
    manifest=dict(status='complete',events=len(events),series_per_event=300,time_samples_per_series=480,
                  first_hour=.5,last_hour=240,step_hours=.5,source_unit_label='cfs',
                  unit_caveat='Source comments only; simulator input conversion not independently established.',sources=sources)
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    notes=['# All-event forcing time-series review','',
           '**40 events ×300 inflow locations ×480 half-hour samples (0.5–240 h).** These are source forcing hydrographs, not 300 verified field gauges and not depth/velocity raster histories.',
           '', '[All 40 event pages (PDF)](all_events_300_locations.pdf) · [Overview PDF](all_events_overview.pdf) · [Event statistics](event_summary.csv)',
           '', '![Full events](overview_full240h.png)','', '![First 24 h](overview_first24h.png)','',
           'Each event page shows all 300 raw source curves, an independently scaled first-24-hour zoom, and a location-by-time heatmap. Each heatmap row is divided by its own maximum; this is for timing visualization, not a model preprocessing change or flood threshold. All-zero rows are gray. Overview panels are divided by the event maximum. Orange/purple/black denote spatial maximum/p95/median, not three distinct measured series.',
           'Source comments label discharge as cfs; simulator input conversion is not independently established. Curves are the exact stored X_event.npy inputs before training-event standardization. Relative timing/order were previously audited against all 40 source archives; see docs/stage2/TARGET_DATA_AUDIT.md.',
           '', '## Training interpretation','',
           'Training uses 25 events, validation seven, testing eight. Plotting all events here is descriptive; choose subsequent training/onset rules using training events and validate them without tuning to the test events.',
           'Target times are sampled rather than exhaustively used. The configured phase proportions are quiet 20%, rising 25%, peak 25%, recession 30%; these are sampler requests, not measured realized proportions. Spatial targets are dry 12.5%, boundary 25%, wet 31.25%, deep 31.25%, with a 15% batch wet-cell target. Candidate availability, feasibility checks and minimum history change actual eligibility.',
           'Fixed-5-hour B excludes target indices below 11 (targets earlier than 6 h). The multi-lead model excludes indices below 25 (targets earlier than 13 h), so every selected target can support a 12 h lead and two observed states. Earlier forcing values still supply the causal history. Quiet time is not necessarily synonymous with the beginning of an event.',
           'Do not infer flood presence or absence from low discharge alone: compare changes relative to each location’s baseline and inspect hydraulic depth/extent. The current plots do not establish that quiet examples caused the observed errors. Audit actual sampled phase/depth/change coverage before changing training.', '']
    for event,_ in arrays:notes+= [f'## {event} ({split[event]})','',f'![All 300 forcing series](events/{event}_all_300_locations.png)','']
    (root/'README.md').write_text('\n'.join(notes)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/stage2_event_forcing_review/20261103'))
    run(parser.parse_args().output_dir)
