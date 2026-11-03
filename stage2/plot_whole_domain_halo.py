"""Plot native-coordinate, north-up Halo depth mosaics against reference maps."""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
import netCDF4
import numpy as np

from stage2.whole_domain_halo import BOUNDARY, DEFAULT_OUTPUT, verify_orientation
from stage2.prepare_dynamic_stream import digest_file


DEPTH_BREAKS = [0.05, .1, .25, .5, 1, 2, 4, 8]


def max_pool(grid, factor):
    """Retain narrow wet channels for display; native data remain unchanged."""
    values=np.where(np.isfinite(grid),grid,-np.inf)
    reduced=np.maximum.reduceat(np.maximum.reduceat(values,np.arange(0,len(values),factor),axis=0),
                                np.arange(0,values.shape[1],factor),axis=1)
    reduced[~np.isfinite(reduced)]=np.nan
    return reduced


def decorate(axis,extent,boundaries,tile_grid=False):
    boundaries.boundary.plot(ax=axis,color='#71362c',linewidth=.45,alpha=.85)
    xmin,ymin,xmax,ymax=boundaries.total_bounds
    axis.set_xlim(min(extent[0],xmin)-.15,max(extent[1],xmax)+.15)
    axis.set_ylim(min(extent[2],ymin)-.15,max(extent[3],ymax)+.15);axis.set_aspect('equal')
    if tile_grid:
        spacing=80*9.506082243286073/1000
        segments=[]
        for x in np.arange(extent[0],extent[1],spacing):segments.append([(x,extent[2]),(x,extent[3])])
        for y in np.arange(extent[3],extent[2],-spacing):segments.append([(extent[0],y),(extent[1],y)])
        axis.add_collection(LineCollection(segments,colors='#777777',linewidths=.13,alpha=.28))
    axis.set_xlabel('UTM easting (km)');axis.set_ylabel('UTM northing (km)')
    axis.ticklabel_format(useOffset=False,style='plain')
    axis.annotate('N',xy=(.94,.96),xytext=(.94,.87),xycoords='axes fraction',
                  ha='center',va='center',fontweight='bold',arrowprops=dict(arrowstyle='-|>',lw=1.3))
    x=extent[0]+.06*(extent[1]-extent[0]);y=extent[2]+.045*(extent[3]-extent[2])
    axis.plot([x,x+10],[y,y],color='#222222',lw=2)
    axis.text(x+5,y+.7,'10 km',ha='center',fontsize=7,
              bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))


def masked_prediction(prediction, probability, depth_threshold, probability_threshold):
    mask = np.isfinite(prediction) & (prediction >= depth_threshold)
    if probability_threshold > 0:
        if probability is None:
            raise ValueError('Wet probability is required for a gated display')
        mask &= np.isfinite(probability) & (probability >= probability_threshold)
    return np.where(mask,prediction,np.nan)


def signed_error_pool(error, factor):
    positive=max_pool(error,factor)
    negative=-max_pool(-error,factor)
    return np.where(np.abs(positive)>=np.abs(negative),positive,negative)


def draw_triplet(axes,truth,prediction,extent,boundaries,factor,tile_grid=False,
                 settings=None,probability=None):
    # Both depth panels use maximum pooling of the same native 10 m footprint.
    # Error is the difference between these displayed maxima, not native RMSE.
    t=max_pool(truth,factor);p=max_pool(prediction,factor)
    if settings:
        candidate=settings['recommended_mask']
        visible=masked_prediction(prediction,probability,candidate['depth_threshold_m'],candidate['probability_threshold'])
        p=max_pool(visible,factor)
    cmap=ListedColormap(plt.get_cmap('Blues')(np.linspace(.22,1,len(DEPTH_BREAKS)-1)))
    cmap.set_bad('#f8f8f6');cmap.set_over('#210d48')
    norm=BoundaryNorm(DEPTH_BREAKS,cmap.N)
    images=[]
    for axis,grid,title in zip(axes[:2],[t,p],['Reference depth','Halo predicted depth']):
        shown=np.ma.masked_where(~np.isfinite(grid)|(grid<.05),grid)
        images.append(axis.imshow(shown,origin='upper',extent=extent,cmap=cmap,norm=norm,interpolation='nearest'))
        axis.set_title(title,fontsize=11)
    union=(t>=.05)|(p>=.05)
    error=np.ma.masked_where(~union|~np.isfinite(t)|~np.isfinite(p),p-t)
    if settings:
        valid=np.isfinite(truth)&np.isfinite(prediction)
        native_union=valid&((truth>=.05)|(prediction>=.05))
        error=signed_error_pool(np.where(native_union,prediction-truth,np.nan),factor)
        error=np.ma.masked_invalid(error)
    error_cmap=plt.get_cmap('RdBu' if settings else 'RdBu_r').copy();error_cmap.set_bad('#f8f8f6')
    images.append(axes[2].imshow(error,origin='upper',extent=extent,cmap=error_cmap,
        norm=TwoSlopeNorm(vmin=settings['error_limits_m'][0] if settings else -2,vcenter=0,
                          vmax=settings['error_limits_m'][1] if settings else 2),interpolation='nearest'))
    axes[2].set_title('Halo − reference (raw depth error)\nLargest absolute native error per display cell' if settings else 'Halo − reference (displayed depth)',fontsize=11)
    for axis in axes:decorate(axis,extent,boundaries,tile_grid)
    return images


def orientation_preview(source_path,metadata_path,output_path,time_index):
    with netCDF4.Dataset(source_path) as source:
        metadata=json.loads(metadata_path.read_text())
        orientation=verify_orientation(source.variables['x'][:],source.variables['y'][:],metadata)
        truth=np.ma.filled(source.variables['output_depth'][time_index,:,:],np.nan)
    boundaries=gpd.read_file(BOUNDARY).to_crs(orientation['crs'])
    boundaries.geometry=boundaries.geometry.scale(xfact=.001,yfact=.001,origin=(0,0))
    extent=np.array(orientation['extent_m'])/1000
    fig,axis=plt.subplots(figsize=(8,11))
    grid=max_pool(truth,4);shown=np.ma.masked_where(~np.isfinite(grid)|(grid<.05),grid)
    image=axis.imshow(shown,origin='upper',extent=extent,cmap='Blues',vmin=.05,vmax=4,interpolation='nearest')
    decorate(axis,extent,boundaries,True)
    axis.set_title('Orientation audit: native reference depth, north up\nConasauga HUC12 boundaries and model tile grid')
    fig.colorbar(image,ax=axis,label='Reference depth (m)',shrink=.7,extend='max')
    fig.tight_layout();output_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=160);plt.close(fig)
    output_path.with_suffix('.json').write_text(json.dumps(orientation,indent=2)+'\n')


def run(args):
    root=args.input_dir
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if manifest['status'] not in ['predictions_complete','complete']:
        raise ValueError('Predictions are not complete')
    settings_path=getattr(args,'settings',None)
    settings=json.loads(settings_path.read_text()) if settings_path else None
    if settings and settings['status']!='ready_for_new_plots':
        raise ValueError('Mask calibration is not ready')
    display_only=bool(settings and settings.get('selection_method')=='display_only_not_calibrated')
    model_label='Halo B (local forcing, depth/extent)' if manifest.get('model')=='halo_local' else 'Halo'
    if manifest.get('model')=='halo_local' and len(manifest.get('trained_horizons_steps') or [])>1:
        model_label='Halo multi-lead (local forcing, depth/extent)'
    plots=getattr(args,'output_dir',None) or root/('plots_calibrated' if settings else 'plots')
    plots.mkdir(parents=True,exist_ok=True)
    targets=getattr(args,'target_hours',None)
    indexed_frames=[(i,r) for i,r in enumerate(manifest['frames']) if targets is None or r['target_hours'] in targets]
    if targets and len(indexed_frames)!=len(set(targets)):
        raise ValueError('Some requested target frames are unavailable')
    boundaries=gpd.read_file(BOUNDARY).to_crs(manifest['orientation']['crs'])
    boundaries.geometry=boundaries.geometry.scale(xfact=.001,yfact=.001,origin=(0,0))
    extent=np.array(manifest['orientation']['extent_m'])/1000
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    with netCDF4.Dataset(root/'depth_maps.nc') as maps, PdfPages(plots/'whole_domain_comparison.pdf') as pdf:
        for i,record in indexed_frames:
            truth=np.ma.filled(maps.variables['true_depth'][i],np.nan)
            prediction=np.ma.filled(maps.variables['halo_depth'][i],np.nan)
            fig,axes=plt.subplots(1,3,figsize=(16,9),layout='constrained')
            probability=np.ma.filled(maps.variables['halo_wet_probability'][i],np.nan) if settings else None
            images=draw_triplet(axes,truth,prediction,extent,boundaries,args.display_factor,args.tile_grid,
                                settings=settings,probability=probability)
            fig.colorbar(images[0],ax=list(axes[:2]),label='Depth (m)',shrink=.65,extend='max')
            fig.colorbar(images[2],ax=axes[2],label='Depth difference (m)',shrink=.65,extend='both')
            fig.suptitle(f"{model_label} · {manifest['event_id']} · origin {record.get('origin_hours',manifest['origin_hours']):g} h · target {record['target_hours']:g} h · lead +{record['lead_hours']:g} h\n"
                f"Native wet-cell RMSE {record['depth_rmse_true_wet_m']:.3f} m · all-valid-cell RMSE {record['depth_rmse_all_valid_m']:.3f} m",fontsize=14)
            footer = (f"North up · NAD83 / UTM 16N · brown lines: HUC12 boundaries · depths <0.05 m hidden\n"
                f"Display: maximum depth within {args.display_factor}×{args.display_factor} native cells; native predictions/metrics retained · known forcing through target")
            if settings:
                candidate=settings['recommended_mask']
                mask_label=f"predicted depth ≥{candidate['depth_threshold_m']:g} m"
                mask_label+=f", wet probability ≥{candidate['probability_threshold']:g}" if candidate['probability_threshold']>0 else " (no wet-probability mask)"
                footer=(f"North up · reference wet depth ≥0.05 m · {mask_label}\n"
                        f"Blue: raw overprediction; red: raw underprediction · range {settings['error_limits_m'][0]:g} to {settings['error_limits_m'][1]:g} m; larger errors saturated · raw RMSE unchanged\n"
                        f"Depth: max within {args.display_factor}×{args.display_factor} cells; error: signed native maximum absolute error, including hidden wet predictions")
            fig.supxlabel(footer,fontsize=9)
            stem=f"target_{record['target_hours']:06.1f}h_lead_{record['lead_hours']:04.1f}h"
            fig.savefig(plots/f'{stem}.png',dpi=170);pdf.savefig(fig,dpi=170);plt.close(fig)
            print(f'Plotted {stem}',flush=True)
    lines=['# Whole-domain Halo depth visualization','',
        f"Event **{manifest['event_id']}**; {manifest['forecast_mode']}; {len(manifest['frames'])} target frames.",
        'All valid cells in all model tiles are included. This differs from the selected stream-tile evaluation.',
        'Event-relative hours are used; source CF dates are not verified calendar dates.',
        '', '[Multipage comparison PDF](plots/whole_domain_comparison.pdf) · [Native maps](depth_maps.nc) · [Native-resolution metrics](metrics.csv)',
        '', 'Coordinates are verified against the original tile-raster affine transform. Native y coordinates decrease with row index; plots use origin=upper, increasing northing upward and easting to the right. HUC12 boundaries are reprojected into EPSG:26916.',
        f'Depth panels share fixed 0.05–8 m bins (values above 8 m use the over-range color). A {args.display_factor}×{args.display_factor} maximum reduction preserves narrow channels for display. The error panel subtracts the displayed maxima, with fixed ±2 m colors; quantitative metrics use native cells.',
        'Prediction depth is not gated by the wet classifier; both truth and prediction are hidden below the same 0.05 m display threshold. Dry valid cells and missing pixels are visually light; the NetCDF retains their distinct values/masks.',
        'These are direct predictions from observed origin states, with known forcing through each target. No intermediate reference hydraulic states are supplied to the model.', '']
    for record in manifest['frames']:
        stem=f"target_{record['target_hours']:06.1f}h_lead_{record['lead_hours']:04.1f}h"
        lines.extend([f"## Target {record['target_hours']:g} h, lead {record['lead_hours']:g} h",'',f'![Reference, Halo and error](plots/{stem}.png)',''])
    if settings:
        notes=['# Whole-domain display' if display_only else '# Validation-selected mask previews','',
            'Display settings only; no threshold calibration or model tuning. Reference and prediction both use the original 0.05 m wet definition.' if display_only else 'Reference wet depth remains >=0.05 m. Prediction-only extent mask is selected on validation events.',
            f"Model: {model_label}. Event: {manifest['event_id']}. Frames: {len(indexed_frames)}. Native maps and metrics are in the parent directory.",
            f"Mask: predicted depth >={settings['recommended_mask']['depth_threshold_m']:g} m, wet probability >={settings['recommended_mask']['probability_threshold']:g}.",
            f"Raw error is Halo minus reference. Blue is positive overprediction; red is negative underprediction. Fixed range {settings['error_limits_m']} m with overflow arrows.",
            'The error is computed before masking, so hidden false wet predictions remain visible as errors. Each display cell shows the signed native error with largest absolute value, rather than the difference between independently pooled depths.',
            'RMSE and native prediction files are unchanged. Displaying depth maxima can visually widen narrow streams; quantitative metrics use native cells.',
            'North up, east right: native decreasing y coordinates use origin=upper; boundaries are reprojected to NAD83 / UTM 16N. Event-relative hours; direct forecasts use observed origin states and known forcing through target.',
            f'Settings SHA-256: {digest_file(settings_path)}','']
        for _,record in indexed_frames:
            stem=f"target_{record['target_hours']:06.1f}h_lead_{record['lead_hours']:04.1f}h"
            notes.extend([f"## Target {record['target_hours']:g} h",'',f'![Calibrated display]({stem}.png)',''])
        (plots/'README.md').write_text('\n'.join(notes)+'\n')
        if display_only:
            manifest.update(status='complete',display_settings_path=str(settings_path),
                            display_settings_sha256=digest_file(settings_path),plots_directory=str(plots))
            (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:
        (root/'README.md').write_text('\n'.join(lines)+'\n')
        manifest['status']='complete';(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--display-factor',type=int,default=4)
    p.add_argument('--tile-grid',action='store_true')
    p.add_argument('--settings',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--target-hours',type=float,nargs='+')
    args=p.parse_args()
    if args.display_factor<1:raise ValueError('Display factor must be positive')
    run(args)
