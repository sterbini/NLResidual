
import pandas as pd
from pathlib import Path
import ruamel.yaml
import json
import numpy as np
import gc


import nafflib
import xtrack as xt
import xpart as xp
import xobjects as xo



# Adding to_pandas method to monitors:
#==============================
def pandas_monitor(self):
    
    extract_columns = ['at_turn','particle_id','x','px','y','py','zeta','pzeta','state']

    _df_tbt = self.data.to_pandas()

    _df_tbt.insert(list(_df_tbt.columns).index('zeta'),'pzeta',_df_tbt['ptau']/_df_tbt['beta0'])
    _df_tbt = _df_tbt[extract_columns].rename(columns={"at_turn": "turn",'particle_id':'particle'})

    return _df_tbt
xt.ParticlesMonitor.to_pandas = pandas_monitor
#==============================

# Adding reset method to monitors:
#==============================
def reset_monitor(self,start_at_turn = None,stop_at_turn = None):
    if start_at_turn is not None:
        self.start_at_turn = start_at_turn
    if stop_at_turn is not None:
        self.stop_at_turn = stop_at_turn
    
    with self.data._bypass_linked_vars():
            for tt, nn in self._ParticlesClass.per_particle_vars:
                getattr(self.data, nn)[:] = 0

xt.ParticlesMonitor.reset = reset_monitor
#==============================


#============================================================
def read_YAML(file="config.yaml"):
    ryaml = ruamel.yaml.YAML()
    # Read configuration for simulations
    with open(file, "r") as fid:
        config = ryaml.load(fid)

    return config
#============================================================

#============================================================
def mkdir(_path):
    if _path is not None:
        if isinstance(_path,Path):
            _path = str(_path)
        for parent in Path(_path+'/_').parents[::-1]:
            if parent.suffix=='':
                if not parent.exists():
                    parent.mkdir()    
#============================================================
                    

def load_collider(path_collider) -> xt.Multiline:
    """
    From C. Drouin's study-DA
    Load a collider configuration from a file using an external path.

    If the file path ends with ".zip", the file is uncompressed locally
    and the collider configuration is loaded from the uncompressed file.
    Otherwise, the collider configuration is loaded directly from the file.

    Returns:
        xt.Multiline: The loaded collider configuration.
    """
    import os
    from zipfile import ZipFile

    # Correct collider file path if it is a zip file
    if os.path.exists(f"{path_collider}.zip") and not path_collider.endswith(".zip"):
        path_collider += ".zip"

    # Load as a json if not zip
    if not path_collider.endswith(".zip"):
        return xt.Multiline.from_json(path_collider)

    # Uncompress file locally
    with ZipFile(path_collider, "r") as zip_ref:
        zip_ref.extractall()
    final_path = os.path.basename(path_collider).replace(".zip", "")
    return xt.Multiline.from_json(final_path)


#===================================================
# BASE CLASS
#===================================================
class Buffer():
    def __init__(self,):
        self.call_ID = None
        # Data dict to store whatever data
        self.data = {}
        # Particle ID to keep track
        self.particle_id = None
        self.complex2tuple = True


    def to_dict(self):
        dct    = {}
        nparts = len(self.particle_id)

        for key,value in self.data.items():

            if len(np.shape(value)) == 1:
                dct[key] = np.repeat(value,nparts)
            elif len(np.shape(value)) == 2:
                dct[key] = np.hstack(value)
            elif len(np.shape(value)) == 3:
                # numpy array for each particle
                if np.issubdtype(value[0].dtype, complex) and self.complex2tuple:
                    # is complex
                    dct[key] = [[(c.real, c.imag) for c in row] for row in np.vstack(value).tolist()]
                else:
                    dct[key] = np.vstack(value).tolist()
            else:
                pass

        return dct
    
    def to_pandas(self):
        return pd.DataFrame(self.to_dict())
    

    def update(self,monitor):
        # Initialize
        #-------------------------
        if self.call_ID is None:
            self.call_ID = 0
        else:
            self.call_ID += 1
        
        if self.particle_id is None:
            self.particle_id = np.arange(monitor.part_id_start,monitor.part_id_end)
        #-------------------------
#===================================================



#===================================================
# TORUS BUFFER
#===================================================
class TORUS_Buffer(Buffer):
    def __init__(self,normalize=True,complex2tuple=True,skip_naff = False):
        super().__init__()  
        self.clean()
        self.normalize      = normalize
        self.complex2tuple  = complex2tuple
        self.skip_naff      = skip_naff

        # To be injected manually!
        #=========================
        self.twiss          = None
        self.nemitt_x       = None
        self.nemitt_y       = None
        self.nemitt_zeta    = None
        #=========================

        # NAFF parameters
        #=========================
        self.n_torus      = None
        self.n_points     = None
        #-------------------------
        self.n_harm       = None
        self.window_order = None
        self.window_type  = None
        self.multiprocesses = None
        #=========================

    def to_dict(self):
        dct    = {}
        for key,value in self.data.items():
            if len(value) == 0:
                continue
            if np.issubdtype(value[0].dtype, complex) and self.complex2tuple:
                # is complex
                dct[key] = [[(c.real, c.imag) for c in row] for row in value]
            else:
                dct[key] = value.tolist()
        return dct
        
    def clean(self,):
        self.data['turn']   = []
        self.data['torus']  = []
        self.data['state']  = []

        self.data['Ax']  = []
        self.data['Qx']  = []
        self.data['Ay']  = []
        self.data['Qy']  = []
        self.data['Azeta']  = []
        self.data['Qzeta']  = []

        self.data['Jx']  = []
        self.data['Jy']  = []
        self.data['Jzeta']  = []

    def process(self,monitor):
        self.update(monitor = monitor)

        assert self.call_ID <= 1, "TORUS_Buffer is not designed to store multiple chunks!"


        # Extracting data
        #-------------------------
        start_at_turn = monitor.start_at_turn
        stop_at_turn  = monitor.stop_at_turn
        self.n_turns  = stop_at_turn-start_at_turn

        x    = monitor.x
        px   = monitor.px
        y    = monitor.y
        py   = monitor.py
        zeta = monitor.zeta
        pzeta = monitor.pzeta

        if self.normalize:
            # Computing normalized coordinates
            #--------------------------
            XX_sig = xt.twiss._W_phys2norm(x,px,y,py,zeta,pzeta, 
                                            W_matrix    = self.twiss.W_matrix,
                                            co_dict     = self.twiss.particle_on_co.copy(_context=xo.context_default).to_dict(), 
                                            nemitt_x    = self.nemitt_x, 
                                            nemitt_y    = self.nemitt_y, 
                                            nemitt_zeta = self.nemitt_zeta)

            x_sig       = XX_sig[0,:,:]
            px_sig      = XX_sig[1,:,:]
            y_sig       = XX_sig[2,:,:]
            py_sig      = XX_sig[3,:,:]
            zeta_sig    = XX_sig[4,:,:]
            pzeta_sig   = XX_sig[5,:,:]
        else:
            x_sig       = x
            px_sig      = px
            y_sig       = y
            py_sig      = py
            zeta_sig    = zeta
            pzeta_sig   = pzeta


        # Reshaping for faster handling
        #========================================
        torus_idx,turn_idx = np.mgrid[:self.n_torus,:self.n_turns]
        torus_idx = torus_idx.reshape(self.n_torus*self.n_turns)
        turn_idx  = turn_idx.reshape(self.n_torus*self.n_turns)
        state_multi = np.all(np.array(np.split(monitor.state.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)==1,axis=1).astype(int)

        x_multi     = np.array(np.split(x_sig.T     , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        px_multi    = np.array(np.split(px_sig.T    , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        y_multi     = np.array(np.split(y_sig.T     , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        py_multi    = np.array(np.split(py_sig.T    , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        zeta_multi  = np.array(np.split(zeta_sig.T  , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        pzeta_multi = np.array(np.split(pzeta_sig.T , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        #========================================
        
        # Computing C-S like invariants
        Jx = 1/2 * np.mean(x_multi**2+px_multi**2,axis=1)
        Jy = 1/2 * np.mean(y_multi**2+py_multi**2,axis=1)
        Jzeta = 1/2 * np.mean(zeta_multi**2+pzeta_multi**2,axis=1)

        if self.skip_naff or (self.n_harm is None) or (self.n_harm == 0):
            # Appending to data
            #-------------------------
            self.data['turn']   = turn_idx
            self.data['torus']  = torus_idx
            self.data['state']  = state_multi
            #----------
            self.data['Jx']  = Jx
            self.data['Jy']  = Jy
            self.data['Jzeta']  = Jzeta
            #-------------------------
        else:
            # Extracting the harmonics
            #--------------------------
            n_harm       = self.n_harm
            window_order = self.window_order
            window_type  = self.window_type

            Ax,Qx       = nafflib.multiparticle_harmonics(x_multi,px_multi      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
            Ay,Qy       = nafflib.multiparticle_harmonics(y_multi,py_multi      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
            Azeta,Qzeta = nafflib.multiparticle_harmonics(zeta_multi,pzeta_multi, num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)



            # Appending to data
            #-------------------------
            self.data['turn']   = turn_idx
            self.data['torus']  = torus_idx
            self.data['state']  = state_multi
            #----------
            self.data['Ax'] = Ax
            self.data['Qx'] = Qx
            self.data['Ay'] = Ay
            self.data['Qy'] = Qy
            self.data['Azeta'] = Azeta
            self.data['Qzeta'] = Qzeta
            self.data['Jx']  = Jx
            self.data['Jy']  = Jy
            self.data['Jzeta']  = Jzeta
            #-------------------------
#===================================================





