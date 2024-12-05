from pathlib import Path
import numpy as np
import pandas as pd
import time 
import gc 
import os

# xsuite
import xtrack as xt
import xfields as xf
import xobjects as xo

# Some custom tools
import xsuiteplus as xPlus





# ==================================================================================================
# --- Main script
# ==================================================================================================
def prepare_line(config,collider = None,full_configuration = None):
    # Parameters
    #-------------------------------------
    seq         = config['sequence']
    disable_ho  = config['disable_ho']

    beam_name = seq[-2:]
    s_marker  = config['s_marker']
    e_marker  = config['e_marker']
    #-------------------------------------

    # Loading collider
    #-------------------------------------
    if collider is None:
        collider = xPlus.load_collider(config['collider_path']) # For zip files
    elif not isinstance(collider,xt.Multiline):
        collider = xPlus.load_collider(collider)
    
    # Selecting sequence and saving emittance
    line0 = collider[seq]
    if full_configuration is None:
        line0.metadata['nemitt_x'] = collider.metadata['config_collider']['config_beambeam']['nemitt_x']
        line0.metadata['nemitt_y'] = collider.metadata['config_collider']['config_beambeam']['nemitt_y']
    else:
        line0.metadata['nemitt_x'] = full_configuration['config_collider']['config_beambeam']['nemitt_x']
        line0.metadata['nemitt_y'] = full_configuration['config_collider']['config_beambeam']['nemitt_y']

    # Extraction name of BB elements
    #-------------------------------------
    _direction  = 'clockwise' if seq == 'lhcb1' else 'anticlockwise'
    bblr_names_all  = collider._bb_config['dataframes'][_direction].groupby('label').get_group('bb_lr').index.to_list()
    bbho_names_all  = collider._bb_config['dataframes'][_direction].groupby('label').get_group('bb_ho').index.to_list()


    # Keeping only the active bblr
    active_bblr     = [nn for nn in bblr_names_all if line0.element_refs[nn].scale_strength._value != 0]
    active_strength = [line0.element_refs[nn].scale_strength._value for nn in bblr_names_all if line0.element_refs[nn].scale_strength._value != 0]


    # Ref Twiss
    #===========================================
    for nn in bblr_names_all:
        line0.element_refs[nn].scale_strength = 0
    for nn in bbho_names_all:
        line0.element_refs[nn].scale_strength = 0
    
    
    twiss0      = line0.twiss4d()
    twiss_init  = twiss0.get_twiss_init(at_element=s_marker)

    # Restoring active bblr and bbho
    #--------------------------------
    for nn,ss in zip(active_bblr,active_strength):
        line0.element_refs[nn].scale_strength = ss
    
    if not disable_ho:
        for nn in bbho_names_all:
            line0.element_refs[nn].scale_strength = 1
    #===========================================


    # twiss0      = line0.twiss4d()
    # twiss_init  = twiss0.get_twiss_init(at_element=s_marker)

    return line0,twiss0,twiss_init,beam_name,s_marker,e_marker


def compute_residual(config_file = 'config.yaml', collider = None, full_configuration = None):
    
    # Parameter space
    #=======================================================================
    config = xPlus.read_YAML(config_file)

    # Tori
    #---------------
    n_part = int(config['n_per_torus'])
    n_turns= int(config['n_turns'])
    #------------------
    r_min   = config['r_min']
    r_max   = config['r_max']
    n_r     = config['n_r']
    n_angles= config['n_angles']
    #--------------------
    radial_list = np.linspace(r_min, r_max, n_r)
    theta_list  = np.linspace(0, np.pi/2, n_angles + 2)[1:-1]
    rr,tt       = np.meshgrid(radial_list, theta_list)
    #--------------------
    rx_vec, ry_vec = rr*np.cos(tt), rr*np.sin(tt)
    #--------------------------------
    fx  = 1/2/np.sqrt(2)
    fy  = 1/2/np.sqrt(3)
    fz  = -1/2/np.sqrt(5)/100
    #=======================================================================


    # GENERATING TORI
    #=======================================================================
    Tx= 2*np.pi*fx*np.arange(n_part)
    Ty= 2*np.pi*fy*np.arange(n_part)
    init_coord = {'x_n':[],'px_n':[],'y_n':[],'py_n':[],'zeta_n':[],'pzeta_n':[]}
    for rx,ry in zip(rx_vec.flatten(),ry_vec.flatten()):
        Gx  = rx*np.exp(1j*Tx)
        Gy  = ry*np.exp(1j*Ty)

        init_coord[f'x_n']  += list(np.real(Gx))
        init_coord[f'px_n'] += list(-np.imag(Gx))
        init_coord[f'y_n']  += list(np.real(Gy))
        init_coord[f'py_n'] += list(-np.imag(Gy))
    #=======================================================================


    # Prepare line
    #=======================================================================
    context = xo.ContextCpu(omp_num_threads=config['num_threads'])
    line0,twiss0,twiss_init,beam_name,s_marker,e_marker = prepare_line(config,collider,full_configuration)

    # select section of the line
    line        = line0.select(s_marker,e_marker)

    # Monitor
    #-------------------------------------
    monitor_name = 'buffer_monitor'
    n_torus = len(rx_vec.flatten())  
    monitor = xt.ParticlesMonitor(  _context      = context,
                                    num_particles = int(n_torus*n_part) ,
                                    start_at_turn = 0, 
                                    stop_at_turn  = n_turns)
    line.insert_element(index=line.element_names[-1], element=monitor, name=monitor_name)
    #-------------------------------------
    line.discard_tracker()
    line.build_tracker(_context=context)
    twiss = line.twiss4d(start=line.element_names[0],end=line.element_names[-1],init=twiss_init)
    #=======================================================================


    # Buffer
    #=======================================================================
    buffer  = xPlus.TORUS_Buffer(complex2tuple=False,skip_naff=True)
    #---------------------------------------------------------
    buffer.n_torus      = n_torus
    buffer.n_points     = n_part
    buffer.twiss        = twiss.get_twiss_init(at_element=monitor_name)
    buffer.nemitt_x     = line0.metadata['nemitt_x']    
    buffer.nemitt_y     = line0.metadata['nemitt_y']    
    buffer.nemitt_zeta  = None # To avoid any rescaling
    #---------------------------------------------------------
    #=======================================================================


    # TRACKING
    #=======================================================================
    particles = line.build_particles(   x_norm   = init_coord['x_n'],
                                        px_norm  = init_coord['px_n'],
                                        y_norm   = init_coord['y_n'],
                                        py_norm  = init_coord['py_n'],
                                        method   = '4d',
                                        nemitt_x = line0.metadata['nemitt_x'],
                                        nemitt_y = line0.metadata['nemitt_y'],
                                        nemitt_zeta     = None,
                                        W_matrix        = twiss.W_matrix[0],
                                        particle_on_co  = twiss.particle_on_co.copy(),
                                        _context        = context)

    monitor.reset(start_at_turn = 0,stop_at_turn = n_turns)
    line.track(particles, num_turns= n_turns,turn_by_turn_monitor=True)
    #=======================================================================


    # Processing
    #==============================
    buffer.process(monitor=monitor)
    df_buffer = buffer.to_pandas().groupby('turn').get_group(n_turns-1).set_index('torus')
    #==============================

    
    gc.collect()
    # os.system('cls||clear')

    residual = []
    for rx,ry,(idx,torus) in zip(rx_vec.flatten(),ry_vec.flatten(),df_buffer.iterrows()):

        # Initial CS-Action
        Jx0 = rx**2/2
        Jy0 = ry**2/2

        # Average CS-Action
        Jx_avg = torus.Jx
        Jy_avg = torus.Jy

        # Relative error
        _err = np.sqrt((Jx_avg-Jx0)**2 + (Jy_avg-Jy0)**2)/np.sqrt(Jx0**2 + Jy0**2)
        
        residual.append(_err)
    residual = np.array(residual)


    df = pd.DataFrame({ 'rx'        :rx_vec.flatten(),
                        'ry'        :ry_vec.flatten(),
                        'r'         :rr.flatten(),
                        'angle'     :tt.flatten(),
                        'residual'  :residual})
    #===============================

    # Saving some metadata
    #===============================
    df.attrs['config'] = config
    if full_configuration is not None:
        df.attrs['config_study-da'] = full_configuration


    # Exporting to parquet
    xPlus.mkdir(config['out_path'])
    df.to_parquet(config['out_path'])

    return df,line,twiss_init,config


# Matplotlib config for possible plot
#============================
import matplotlib.pyplot as plt
import matplotlib.colors as colors
FIGPATH  = './'
FIG_W = 6
FIG_H = 6


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "xtick.labelsize":14,
    "ytick.labelsize":14,
    "axes.labelsize":16,
    "axes.titlesize":16,
    "legend.fontsize":14,
    "legend.title_fontsize":16
})
plt.rc('text.latex', preamble=r'\usepackage{physics}')
for key in plt.rcParams.keys():
    if 'date.auto' in key:
        plt.rcParams[key] = "%H:%M"

def polar_grid(rho_ticks=None,phi_ticks=None,**kwargs):
    _xlim = plt.gca().get_xlim()
    _ylim = plt.gca().get_ylim()
    _default_kwargs = {'alpha':0.5,'color':'grey','linestyle':'-','lw':1}
    _default_kwargs.update(kwargs)


    # Add circle patches
    #====================================
    if rho_ticks is not None:

        lowest_zorder = min([obj.get_zorder() for obj in plt.gca().get_children()])
        _tvec = np.linspace(0,2*np.pi,100)
        for rho in rho_ticks:
            plt.gca().add_patch(plt.Circle((0, 0), rho,fill=False,zorder=lowest_zorder-1, **_default_kwargs)) 

    # Add angle lines
    #====================================
    if phi_ticks is not None:
        
        lowest_zorder = min([obj.get_zorder() for obj in plt.gca().get_children()])
        for phi in phi_ticks:
            _rmax = 2*np.max([np.abs(_xlim),np.abs(_ylim)])
            plt.plot([0,_rmax*np.cos(phi)],[0,_rmax*np.sin(phi)],zorder=lowest_zorder-1, **_default_kwargs)


    plt.xlim(_xlim)
    plt.ylim(_ylim)
#============================



# ==================================================================================================
# --- Script for execution
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-id"  , "--id_job"        , help = "Job ID"               , default = '0')
    aparser.add_argument("-c"   , "--config"        , help = "Config file"          , default = './config_2_nlr.yaml')
    aparser.add_argument("-plt" , "--plot"          , help = "Show results as plot" , action  = "store_true")
    aparser.add_argument("-splt" , "--splot"        , help = "Save results as plot" , action  = "store_true")
    aparser.add_argument("-DA"  , "--critical_DA"   , help = "Set crit. DA for plot", default = 6)
    aparser.add_argument("-CRA"  , "--critical_residual"   , help = "Set crit. DA for plot", default = None)
    args = aparser.parse_args()
    
    
    
    # Main function
    df,line,twiss_init,config = compute_residual(   Jid         = args.id_job,
                                                    config_file = args.config)
    
    # # Exporting to parquet
    # xutils.mkdir('./outputs')
    # df.to_parquet(f'./outputs/OUT_JOB_{str(args.id_job).zfill(4)}.parquet')


    if args.plot or args.splot:

        # critical_res = 1e-3
        # vmin = critical_res/1e1
        # vmax = critical_res*1e1

        critical_res = 5e-3
        vmin = critical_res*(1-0.25)
        vmax = critical_res*(1+0.25)

        plt.figure()
        # plt.suptitle(config['collider_path'])
        plt.scatter(df.rx,df.ry,s=10,c=df.residual,cmap='RdGy_r',norm=colors.LogNorm(vmin=vmin,vmax=vmax))
        cbar = plt.colorbar(pad=-0.1)

        plt.axis('square')
        plt.xlim([0,df.r.max()+0.5])
        plt.ylim([0,df.r.max()+0.5])
        plt.gca().set_facecolor('#fafafa')  # off-white background


        plt.xlabel(r'$r_x$',fontsize=14)
        plt.ylabel(r'$r_y$',fontsize=14)
        cbar.set_label(r'Non-linear Residual')

        polar_grid(rho_ticks=np.arange(0,20+1,1),phi_ticks=np.linspace(0,np.pi/2,21+2)[1:-1],alpha=0.3)

        # plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper right',fontsize=10)
        plt.tight_layout()

        if args.splot:
            xPlus.mkdir(config['out_path'] + '/figures')
            plt.savefig(config['out_path'] + f'/figures/{str(Path(config['collider_path']).parent).split('/')[-1]}.png',dpi=300)
        else:
            plt.show()
    
    #===========================