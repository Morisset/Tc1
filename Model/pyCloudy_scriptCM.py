#===============================================================
# This is the script used for the Aleman et al. 2019 paper on PN Tc 1
# See at the end of this file the commented part to know how to run it.
#===============================================================

import numpy as np
import matplotlib.pyplot as plt
import pyCloudy as pc
from pyCloudy.utils.astro import conv_arc
from pyCloudy.utils.misc import convert_label, sextract, correc_He1
import pyneb as pn
from pyneb.utils.misc import int_to_roman

pc.config.cloudy_exe = '/usr/local/Cloudy/c17.01/source/cloudy.exe'
dir_ = './outputs/'

#%%
class Model(object):
    
    def __init__(self, model_name, IR_factor=1., instrument='XShooter', 
                 SED='BB', use_slit = True):

        self.dir_ = dir_
        self.model_name = model_name
        self.instrument = instrument
        self.SED = SED
        self.use_slit = use_slit
        
        self.options = ('print last', 'Save lines, intensity, column, emergent, ".lines"')
        
        emis_tab = ['H  1 4861.33A','H  1 6562.81A','H  1 4340.94A','He 1 5875.64A','He 1 4471.49A',
                      'N  2 6583.45A','O  1 6300.30A','BLND 4363.00A','O  3 5006.84A','Ne 3 3868.76A',
                      'BLND 5199.00A','O  2 3728.81A','O  2 3726.03A','BLND 7323.00A','Mg 1 4562.60A',
                      'S  3 6312.06A','Ar 3 5191.82A','S  2 6730.82A','S  2 6716.44A','S  2 4076.35A',
                      'S  2 4068.60A','Fe 3 4658.01A','Fe 3 5270.40A','Fe 3 4701.62A', 'BLND 5755.00A',
                      'BLND 7332.00A', 'Ar 3 7135.79A', 'Ar 3 7751.11A', 'Ne 3 3967.47A', 'BLND 1909.00A',
                      'BLND 2326.00A', 'Cl 3 5517.71A', 'Cl 3 5537.87A', 'Ar 2 6.98337m', 'Ar 3 8.98898m',
                      'Ar 3 21.8253m', 'S  4 10.5076m', 'Ne 2 12.8101m', 'Ne 3 15.5509m', 'S  3 18.7078m',
                      'S  3 33.4704m', 'Ca B 12.3684m', 'Ca B 12.3837m', 'BLND 3726.00A', 
                      'BLND 3729.00A','Ca B 5875.64A','S  3 9068.62A', 'Ca B 4471.49A', 'Ca B 6.94480m',
                      'Ca B 7.09071m']
        

        self.emis_tab_call = [convert_label(label) for label in emis_tab]
        
        # Observed Tc 1 Line Intensities (relative to Hb = 100) for the lines above, in order. Corrected for reddening.
#        Obs_Tc1_old = [100.0, 316.92571, 46.6812, 15.24, 5.3973, 114.80788, 0.10972677, 
#                     0.4629,109.115, 0.59746,0.01575, 149.216, 234.66, 6.10 ,0.07353,
#                     0.57287,0.03006,3.9613816, 2.5649392,0.281553, 0.55302, 0.271188, 
#                     0.134266, 0.083, 1.2, 5.1,8.3, 1.96, 0.175, 27., 45.,0.28, 0.32, 
#                     32.8, 6.5, 0.432, 0.47, 37.5, 1.46, 14., 6.21, 0.92, 0.08, 234, 
#                     149, 15.24, np.nan, 5.40, 32.8, 32.8]
                     
        Obs_Tc1 = [100.0, 302.83, 46.66, 15.26, 5.393, 111.33, 0.087, 
                     0.462,109.143, 0.596,0.016, 148.885, 234.14, 1.434+4.677 ,0.037,
                     0.574,0.030,3.531, 2.306,0.281, 0.552, 0.271, 
                     0.134, 0.083, 1.198, 2.56+2.54,8.292, 1.959, 0.175, 27., 45.,0.28, 0.32, 
                     32.8, 6.5, 0.432, 0.47, 37.5, 1.46, 14., 6.21, 0.92, 0.08, 234, 
                     149, 15.26, np.nan, 5.393, 32.8, 32.8]
 
        Obs_Tc1_wave = [4861.33, 6562.77,4340.47,5875.66,4471.50,6583.50,6300.30,
                          4363.21,5006.84,3868.75,5198.26,3728.82,3726.03,7324.83,
                          4571.10,6312.10,5191.82,6730.82,6716.44,4076.35,4068.60,
                          4658.10,5270.40,4701.62, 5755., 7332, 7135, 7752, 3967, 
                          1909, 2326, 5518, 5538, 69833.7, 89889.8, 218253, 105076, 
                          128101, 155509, 187078, 334704, 123684, 123837, 3726, 
                          3729, 5876, 9069, 4471, 69448.0, 70907.1]
                          
        self.line_ordered = ['H__1_486133A',
                     'H__1_656281A',
                     'H__1_434094A',
#                     'HE_1_587564A',
                     'CA_B_587564A',
                     'CA_B_447149A',
                     'BLND_519900A',
                     'BLND_575500A',
                     'N__2_658345A',
                     'O__1_630030A',
                     'O__2_372881A',
                     'O__2_372603A',
                     'BLND_732300A',
                     'BLND_733200A',
                     'O__3_500684A',
                     'BLND_436300A',
                     'NE_3_386876A',
                     'NE_3_396747A',
                     'CL_3_551771A', 
                     'CL_3_553787A',
                     'S__2_673082A',
                     'S__2_671644A',
                     'S__2_407635A',
                     'S__2_406860A',
                     'S__3_631206A',
                     'S__3_906862A',
                     'AR_3_519182A',
                     'AR_3_713579A',
                     'AR_3_775111A',
                     'MG_1_456260A',
                     'FE_3_465801A',
                     'FE_3_527040A',
                     'FE_3_470162A',
                     'BLND_232600A',
                     'BLND_190900A']
                     
                     
        IR_lines=   ['CA_B_123684M',
#                     'CA_B_123837M',
                     'NE_2_128101M',
                     'NE_3_155509M',
                     'S__3_187078M',
                     'S__3_334704M',
                     'S__4_105076M',
                     'CA_B_694480M',
                     'AR_2_698337M',
                     'CA_B_709071M',
                     'AR_3_898898M',
                     'AR_3_218253M'
                     ]
        self.line_ordered.extend(IR_lines)
        self.Obs_Tc1 = {}        
        self.Obs_Tc1_wave = {}
        self.emis_tab = {}
        for line in self.emis_tab_call:
            self.Obs_Tc1[line] = Obs_Tc1[self.emis_tab_call.index(line)]
            self.Obs_Tc1_wave[line] = Obs_Tc1_wave[self.emis_tab_call.index(line)]
            self.emis_tab[line] = emis_tab[self.emis_tab_call.index(line)]
            if self.Obs_Tc1_wave[line] > 10000:
                self.Obs_Tc1[line] *= IR_factor
        if self.instrument in ('LCO', 'AAT'):
            for k in self.Obs_Tc1:
                if self.Obs_Tc1_wave[k] < 10000. and self.Obs_Tc1_wave[k] > 2500. :
                    self.Obs_Tc1[k] = np.nan
            self.Obs_Tc1['H__1_486133A'] = 100.
            self.Obs_Tc1['HE_1_447149A'] = 1.1
            self.Obs_Tc1['CA_B_447149A'] = 1.1
            self.Obs_Tc1['N__2_658345A'] = 95.4
            self.Obs_Tc1['HE_1_587564A'] = 9.01
            self.Obs_Tc1['CA_B_587564A'] = 9.01
            self.Obs_Tc1['O__3_500684A'] = 124.
            self.Obs_Tc1['O__2_372603A'] = 130.
            self.Obs_Tc1['O__2_372881A'] = 86.
            self.Obs_Tc1['BLND_436300A'] = 0.55
            self.Obs_Tc1['AR_3_519182A'] = 0.031
            self.Obs_Tc1['AR_3_713579A'] = 5.65
            self.Obs_Tc1['AR_3_775111A'] = 1.66
            self.Obs_Tc1['S__2_671644A'] = 2.2
            self.Obs_Tc1['S__2_673082A'] = 3.5
            self.Obs_Tc1['BLND_575500A'] = 1.09
            self.Obs_Tc1['CL_3_551771A'] = 0.285
            self.Obs_Tc1['CL_3_553787A'] = 0.303
            self.Obs_Tc1['BLND_732300A'] = 5.43
            self.Obs_Tc1['BLND_733200A'] = 4.57
            self.Obs_Tc1['S__3_631206A'] = 0.46
            self.Obs_Tc1['S__2_406860A'] = 0.62
            self.Obs_Tc1['S__2_407635A'] = 0.18
            self.Obs_Tc1['BLND_519900A'] = 0.04
            self.Obs_Tc1['O__1_630030A'] = 0.12
            self.Obs_Tc1['S__3_906862A'] = 12.51
        #===============================================================
        # Blackbody parameters - L in erg/s ; T in K
        #===============================================================
        self.luminosity_unit = 'Q(H)'
        self.luminosity= 47.2#  np.log10(3000.0*3.826e33)
        self.Temp = 32000.0
        self.gStel = 4
        self.Zstel = 0
        #===============================================================
        # Gas density (nH) in cm-3
        #===============================================================
        self.densi = np.log10(1800)
        self.ff = 1.0
        #===============================================================
        # Inner radius - in log(cm)
        #===============================================================
        self.inner_radius = np.log10(1.0e15)
        self.outer_radius = None #np.log10(2.1e17)
        #===============================================================
        # Distance in kpc
        #===============================================================
        self.distance = 2.3
        
        #===============================================================
        #Abundances
        #===============================================================
        
        # Model c068mb
        
        self.abund = {'He':np.log10(0.14E0),
                    'C':np.log10(5.13E-4),
                    'N':np.log10(8.90E-5),
                    'O':np.log10(6.00E-4),
                    'Ne':np.log10(1.60E-5),
                    'Mg':np.log10(3.47E-5),
                    'Si':np.log10(9.01E-7),
                    'S':np.log10(4.10E-6),
                    'Cl':np.log10(9.40E-8),
                    'Ar':np.log10(1.80E-6),
                    'Fe':np.log10(5.00E-7)}
        self.dust_type = 'graphite_ism_10.opc'
        self.dust = 1.5
        # A function in form of lambda to transform size in cm into arcsec, for a distance defined above.
        
        self.arcsec = lambda cm: conv_arc(dist=self.distance, dist_proj=cm)
        self.cm = lambda arcsec: conv_arc(dist=self.distance, ang_size=arcsec)

        # An object to correct from redenning in case of necessity
        self.RC = pn.RedCorr(cHbeta=0.4,law='CCM89')

        # Reading the line profiles
        self.profiles = np.genfromtxt('obs_profile.cvs', delimiter=',', names=True)
        self.ypos = 1.17
        
    #===============================================================
    def _make_mask(self,ap_center=[0., 0.], ap_size=[1., 1.]):
        """
        This returns a mask (values between 0. and 1.) to multiply the image to simulate the flux passing through an aperture.
        """
        self.x_arc = self.arcsec(self.M_sphere.cub_coord.x_vec)
        self.y_arc = self.arcsec(self.M_sphere.cub_coord.y_vec)
        try:
            self.i_slit = np.where(self.y_arc > self.ypos)[0][0]
        except:
            self.i_slit = None
        X, Y = np.meshgrid(self.x_arc, self.y_arc)
        bool_mask = ((X >= ap_center[0] - ap_size[0]/2.) & 
                (X <= ap_center[0] + ap_size[0]/2.) & 
                (Y >= ap_center[1] - ap_size[1]/2.) & 
                (Y <= ap_center[1] + ap_size[1]/2.))
        mask = np.zeros_like(X)
        mask[bool_mask] = 1.0
        return mask

    #===============================================================
    def _mk_xshooter(self, width=0.5):
    
        # Extracts:
        # Pixels 1 to 19 (3.8")
        mask1 = self._make_mask(ap_center=[+3.6 - 0.39, 0.0 + self.ypos], ap_size=[3.8, width])
        # Pixels 36 to 55 (4.0")
        mask2 = self._make_mask(ap_center=[+3.5 + 0.39, 0.0 + self.ypos], ap_size=[4.0, width])
        # The sum can reach values of 2.0: this is where the 2 masks overlap, and it is OK.
        maskT = mask1 + mask2
    
        return maskT
    
    def _mk_LCO(self):
        mask1 = self._make_mask(ap_center=[2.7, 1.0], ap_size=[2.0, 2.0])
        return mask1

    def _mk_AAT(self):
        mask1 = self._make_mask(ap_center=[4.5, 2.5], ap_size=[9., 5.])
        return mask1

    def _mk(self, width=0.5):
        if not self.use_slit:
            return 1.0
        if self.instrument == 'LCO':
            return self._mk_LCO()
        elif self.instrument == 'XShooter':
            return self._mk_xshooter(width)
        elif self.instrument == 'AAT':
            return self._mk_AAT()
        else:
            raise ValueError('Error {} not a valid apperture name'.format(self.instrument))
            
    def call_pyCloudy(self):
    
      #===============================================================
      # Verbosity level
      #===============================================================
      pc.log_.level = 1
    
      #===============================================================
      # Cloudy model object
      #===============================================================
      c_input = pc.CloudyInput('{0}{1}'.format(self.dir_, self.model_name))
    
      #===============================================================
      # filling the object
      #===============================================================
    
      # ionizing spectrum, Teff and L*
      if self.SED == 'WMbasic':
          c_input.set_star(SED='table star wmbasic', 
                           SED_params='{} {} {}'.format(self.Temp, self.gStel, self.Zstel),
                           lumi_unit = self.luminosity_unit,
                           lumi_value = self.luminosity)
      elif self.SED == 'BB':
          c_input.set_BB(Teff = self.Temp, lumi_unit = self.luminosity_unit,
                         lumi_value = self.luminosity)
      else:
          raise ValueError('ERROR: SED {} unknown'.format(self.SED))
      #gas density (nH)
      c_input.set_cste_density(self.densi, ff=self.ff)
    
      #Inner radius
      c_input.set_radius(r_in=self.inner_radius, r_out=self.outer_radius)
    
      #Abundances
      c_input.set_abund(ab_dict=self.abund, nograins=True)  
      #c_input.set_grains('Orion graphite 6')
      c_input.set_grains('"{}" {}'.format(self.dust_type, self.dust))
      #Other options
      c_input.set_other(self.options)
      c_input.set_iterate()
      c_input.set_sphere(True)
    
      # Table with te lines to print the emissivities
      # Needed for the 3D model to simulate the maps
    
      c_input.set_emis_tab(self.emis_tab.values())
    
      # distance in kpc
      c_input.set_distance(self.distance)   
    
      c_input.print_input()
    
      c_input.run_cloudy()

    def read_model(self, doplot=False, HeI_cut=None, verbose=True, 
                   cube_size=None, r_out_cut=None):
        
      self.Mod = pc.CloudyModel('{0}{1}'.format(self.dir_, self.model_name), 
                                read_grains=True)
      if "CA_B_587564A"  in self.Mod.emis_labels:
          i_line = np.where(self.Mod.emis_labels == 'CA_B_587564A')[0][0]
          self.Mod.emis_full[i_line] *= correc_He1(self.Mod.te_full, self.Mod.ne_full, 5876)
      if "CA_B_447149A"  in self.Mod.emis_labels:
          i_line = np.where(self.Mod.emis_labels == 'CA_B_447149A')[0][0]
          self.Mod.emis_full[i_line] *= correc_He1(self.Mod.te_full, self.Mod.ne_full, 4471)
          
      self.Mod.distance = self.distance
          
      if HeI_cut is not None:
          if cube_size is None:
              print('Needs a cube_size to compute the HeI cut')
              self.Mod = None
              return
          for radius in self.Mod.radius[::-1]:
              self.Mod.r_out_cut = radius
              self.make_3D(cube_size, doplot=False, verbose=False)
              if verbose:
                  print('R = {:8.2e} HeI dif = {:.2f}'.format(radius, self.difs['CA_B_587564A']))
              if self.difs['CA_B_587564A'] > HeI_cut:
                  break      
      elif r_out_cut is not None:
          self.Mod.r_out_cut = r_out_cut
      
      if verbose:
          self.Mod.print_stats()
    
      if doplot:
          f, ax = plt.subplots(figsize=(10,7))
          ax.plot(self.Mod.radius,self.Mod.te,label='Te')
          ax.legend(loc=3)
    
          #Show on the screen
          #plt.show()
     
          #print to file
          #plt.tight_layout(pad=0.2)
          name = self.dir_+self.model_name+'_Te.png'
          f.savefig(name, dpi=200)
    
    def make_3D(self, cube_size, doplot=False, verbose=True, no_VIS=False):
        
      self.M_sphere = pc.C3D(self.Mod, dims=cube_size, center=False, n_dim=1)    
    
      # Slits for UVB and VIS have different widths
        # integrate inside these masks
    
      self.maskUVB = self._mk(0.5)
      if no_VIS:
          self.maskVIS = np.nan
      else:
          self.maskVIS = self._mk(0.4)

      Hb_label = 'H__1_486133A'
      self.Ib = self.Mod.get_emis_vol(Hb_label, at_earth=True) 
      self.Ib_redenned = self.Ib * self.RC.getCorr(4863)
              
      factor = 4. * np.pi * (self.distance * pc.CST.KPC)**2 
    
      Fhb_UVB = ((self.M_sphere.get_emis(Hb_label)*self.M_sphere.cub_coord.cell_size).sum(1) * 
                 self.maskUVB).sum()/factor
      Fhb_VIS = ((self.M_sphere.get_emis(Hb_label)*self.M_sphere.cub_coord.cell_size).sum(1) * 
                 self.maskVIS).sum()/factor
      Fhb_full = ((self.M_sphere.get_emis(Hb_label)*self.M_sphere.cub_coord.cell_size).sum())/factor

      self.flux_mod = {}
      self.difs = {}
      self.QF = {}

      for line in self.line_ordered:
        if verbose:
            print('Doing line {}'.format(line))
        if self.Obs_Tc1_wave[line] < 2500.:
           mask = 1.0
           Fhb = Fhb_full
        elif self.Obs_Tc1_wave[line] < 5500.:
           mask = self.maskUVB
           Fhb = Fhb_UVB
        elif self.Obs_Tc1_wave[line] < 10000.:
           mask = self.maskVIS
           Fhb = Fhb_VIS
        else:
           mask = 1.0
           Fhb = Fhb_full
           
        fObs = self.Obs_Tc1[line]
        flux = 100.0 * ((self.M_sphere.get_emis(line)*self.M_sphere.cub_coord.cell_size).sum(1) * 
                mask).sum()/factor/Fhb
        self.flux_mod[line] = flux # flux in in erg/cm2/s
        dif = 100.0 * (flux - fObs) / fObs
        self.difs[line] = dif
        if fObs < 1.:
            deltaI = 0.5
        elif fObs < 10.:
            deltaI = 0.3
        else: 
            deltaI = 0.2
        if self.Obs_Tc1_wave[line] < 2500:
            deltaI += 0.15
        if self.Obs_Tc1_wave[line] > 10000:
            deltaI += 0.15
        tol = np.log10(1 + deltaI)
        self.QF[line] = np.log10(flux / fObs) / tol

    def plot_slits(self, *args, **kwargs):
        plot_slits(self, *args, **kwargs)
        
    def print_res(self, *args, **kwargs):
        print_res(self, *args, **kwargs)
        
    def plot_profile(self, *args, **kwargs):
        plot_profile(self, *args, **kwargs)

    def plot_all_profiles(self, *args, **kwargs):
        plot_all_profiles(self, *args, **kwargs)
        
    def plot_dust(self, *args, **kwargs):
        plot_dust(self, *args, **kwargs)
      
    def search_sol(self, *args, **kwargs):
        search_sol(M=self, *args, **kwargs)
        
    def get_ICF(self, *args, **kwargs):
        return get_ICF(self, *args, **kwargs)
        
#%%          
def print_res(M, print_IR=False, verbose=False):
  f = open(M.dir_ + M.model_name + '.res', 'w')
  def print2(s):
      print(s)
      f.write(s+'\n')
  print2('Instrument {}, Using slit {}'.format(M.instrument, M.use_slit))
  tup_elem1 = ('He', 'C', 'N', 'O', 'Ne')
  tup_elem2 = ('Cl', 'Ar', 'S', 'Mg', 'Fe')
  print2('Model name: {}'.format(M.Mod.model_name))
  print2('Distance = {:.2f} kpc, H-density = {:.0f} cm-3, FF = {:.2f}'.format(M.Mod.distance, 
                                                         M.Mod.nH_mean, M.Mod.ff[0]))
  try:
      teff = float(sextract(M.Mod.out['Blackbody'], 'body', '*'))
  except:
      pass
  try:
      teff = float(sextract(M.Mod.out['table star'], 'wmbasic', '3.'))
  except:
      pass
  print2('{}: Temp = {} K, Lumi = {:.0f} Lsol, Q(H) = {:.2f}'.format(M.SED, teff,
        10**float(sextract(M.Mod.out['SED4'], 'Lsun:', 'Abs')), np.log10(M.Mod.Q0)))
  print2('Rin = {:.2f}, Rout = {:.2f}, RStr = {:.2f}'.format(M.arcsec(M.Mod.r_in), 
        np.min((M.arcsec(M.Mod.r_out_cut), M.arcsec(M.Mod.r_out))), 
        M.arcsec(M.Mod.r_out)))
  print2(''.join(['{} {:.2f}, '.format(k, M.Mod.abund[k]) for k in tup_elem1]))
  print2(''.join(['{} {:.2f}, '.format(k, M.Mod.abund[k]) for k in tup_elem2]))
  grains = sextract(M.Mod.out['grains'], 'grains', '*').strip()
  print2('{}, D/G = {:.4f}'.format(grains, M.Mod.gdgrat.sum(0).mean()))
  print2('<logU> = {:.2f}'.format(M.Mod.log_U_mean_ne))
  print2('Hbeta       5.1e-11 {:.1e}'.format(M.Ib))
  for line in M.line_ordered:
      try:
          if np.isfinite(M.flux_mod[line]):
              print2('{} {:6.2f} {:6.2f} {:7.2f}'.format(M.emis_tab[line], M.Obs_Tc1[line], 
                                                        M.flux_mod[line], M.QF[line]))
      except:
          if verbose:
              print2('{} ERROR'.format(M.emis_tab[line]))
  def print_ratio(label1, label2, label3):
      print2('{} {:.2e} {:.2e}'.format(label1, M.Obs_Tc1[label2]/M.Obs_Tc1[label3],
                                       M.flux_mod[label2]/M.flux_mod[label3]))
  sO32_obs = ((M.Obs_Tc1['O__3_500684A'] + M.Obs_Tc1['O__2_372881A'] + M.Obs_Tc1['O__2_372603A'])/
              M.Obs_Tc1['H__1_486133A'])    
  sO32_mod = ((M.flux_mod['O__3_500684A'] + M.flux_mod['O__2_372881A'] + M.flux_mod['O__2_372603A'])/
              M.flux_mod['H__1_486133A'])    
  rO32_obs = (M.Obs_Tc1['O__3_500684A'] / (M.Obs_Tc1['O__2_372881A'] + M.Obs_Tc1['O__2_372603A']))    
  rO32_mod = (M.flux_mod['O__3_500684A'] / (M.flux_mod['O__2_372881A'] + M.flux_mod['O__2_372603A']))    
  print2('[OII] + [OIII] / Hb  {:.2e} {:.2e}'.format(sO32_obs, sO32_mod))
  print2('[OIII] / [OII]       {:.2e} {:.2e}'.format(rO32_obs, rO32_mod))
  print_ratio('N [OII] 3726/3729   ', 'O__2_372603A', 'O__2_372881A')
  print_ratio('N [SII] 6731/6716   ', 'S__2_673082A', 'S__2_671644A')
  print_ratio('N [SII] 4069/4076   ', 'S__2_406860A', 'S__2_407635A')
  print_ratio('N [ClIII] 5538/5518 ', 'CL_3_553787A', 'CL_3_551771A')
  if print_IR:
      print_ratio('N [SIII] 18.7/33.5  ', 'S__3_187078M', 'S__3_334704M')
  print_ratio('T [NII] 5755/6584   ', 'BLND_575500A', 'N__2_658345A')
  print_ratio('T [OIII] 4363/5007  ', 'BLND_436300A', 'O__3_500684A')
  if print_IR:
      print_ratio('T [SIII] 6312/18.7  ', 'S__3_631206A', 'S__3_187078M')
      rNeIII_obs = (M.Obs_Tc1['NE_3_386876A'] + M.Obs_Tc1['NE_3_386876A'])/M.Obs_Tc1['NE_3_155509M']
      rNeIII_mod = (M.flux_mod['NE_3_386876A'] + M.flux_mod['NE_3_386876A'])/M.flux_mod['NE_3_155509M']
      print2('T [NeIII] 3930+/15.6 {:.2e} {:.2e}'.format(rNeIII_obs, rNeIII_mod))
      print_ratio('T [ArIII] 7135/9.0  ', 'AR_3_713579A', 'AR_3_898898M')
  print2('T [ArIII] 5192/7135+ {:.2e} {:.2e}'.format(M.Obs_Tc1['AR_3_519182A']/(M.Obs_Tc1['AR_3_713579A']+M.Obs_Tc1['AR_3_775111A']),
        M.flux_mod['AR_3_519182A']/(M.flux_mod['AR_3_713579A']+M.flux_mod['AR_3_775111A'])))
  rSII_obs = ((M.Obs_Tc1['S__2_406860A'] + M.Obs_Tc1['S__2_407635A'])/
              (M.Obs_Tc1['S__2_673082A'] + M.Obs_Tc1['S__2_671644A']))        
  rSII_mod = ((M.flux_mod['S__2_406860A'] + M.flux_mod['S__2_407635A'])/
              (M.flux_mod['S__2_673082A'] + M.flux_mod['S__2_671644A']))        
  print2('[SII] 4072+/6720+    {:.2e} {:.2e}'.format(rSII_obs, rSII_mod))
  rOII_obs = ((M.Obs_Tc1['O__2_372881A'] + M.Obs_Tc1['O__2_372603A'])/
              (M.Obs_Tc1['BLND_732300A'] + M.Obs_Tc1['BLND_733200A']))    
  rOII_mod = ((M.flux_mod['O__2_372881A'] + M.flux_mod['O__2_372603A'])/
              (M.flux_mod['BLND_732300A'] + M.flux_mod['BLND_733200A']))    
  print2('[OII] 3727+/7320+    {:.2e} {:.2e}'.format(rOII_obs, rOII_mod))
  
  N2 = pn.Atom('N',2)
  O3 = pn.Atom('O',3)
  Ar3 = pn.Atom('Ar',3)          
  TO3_obs = O3.getTemDen(M.Obs_Tc1['BLND_436300A']/M.Obs_Tc1['O__3_500684A'], den=2e3, 
                            wave1 = 4363, wave2 = 5007)
  TO3_mod = O3.getTemDen(M.flux_mod['BLND_436300A']/M.flux_mod['O__3_500684A'], den=2e3, 
                            wave1 = 4363, wave2 = 5007)
  TN2_obs = N2.getTemDen(M.Obs_Tc1['BLND_575500A']/M.Obs_Tc1['N__2_658345A'], den=2e3, 
                            wave1 = 5755, wave2 = 6584)
  TN2_mod = N2.getTemDen(M.flux_mod['BLND_575500A']/M.flux_mod['N__2_658345A'], den=2e3, 
                            wave1 = 5755, wave2 = 6584)
  TAr3_obs = Ar3.getTemDen(M.Obs_Tc1['AR_3_519182A']/(M.Obs_Tc1['AR_3_713579A']+M.Obs_Tc1['AR_3_775111A']), 
                           den=2e3, to_eval=('L(5192)/(L(7136)+L(7751))'))
  TAr3_mod = Ar3.getTemDen(M.flux_mod['AR_3_519182A']/(M.flux_mod['AR_3_713579A']+M.flux_mod['AR_3_775111A']), 
                           den=2e3, to_eval=('L(5192)/(L(7136)+L(7751))'))
  print2('T [NII]               {:.0f}     {:.0f} '.format(TN2_obs, TN2_mod))
  print2('T [OIII]              {:.0f}     {:.0f} '.format(TO3_obs, TO3_mod))
  print2('T [ArIII]             {:.0f}     {:.0f} '.format(TAr3_obs, TAr3_mod))
  f.close()

def plot_profile(M, label='H__1_486133A', ax=None, norm=1.0, close=False, label_str=None):
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = plt.gcf()
    prof = (M.M_sphere.get_emis(label) * M.M_sphere.cub_coord.cell_size).sum(0)[:,M.i_slit]
    ax.plot(M.x_arc, prof / np.max(prof) * norm, linestyle='--', c='b')
    ax.plot(-M.x_arc[::-1], (prof / np.max(prof) * norm)[::-1], linestyle='--', c='b')
    if label in M.profiles.dtype.names:
        ax.plot(M.profiles['x_vis'], M.profiles[label]/np.nanmax(M.profiles[label]), c='r')
    ax.set_xlim((-6, 6))
    ax.set_ylim((0, 1.2))
    if label_str is None:
        label_str = label
    ax.set_title(label_str)
    if close:
        plt.close(f)

def plot_all_profiles(M, close=False):
    f, axes = plt.subplots(2, 3, figsize = (10, 5))
    for ax,label, norm, label_str in zip( axes.ravel(), 
                              ('H__1_486133A', 'CA_B_587564A', 'N__2_658345A', 
                               'O__2_372603A', 'O__3_500684A', 'S__2_673082A'),
                               (1.1,1.1,1.,1.,1.3,1.),
                               (r'H$\beta$', r'HeI 5876$\AA$',r'[NII] 6584$\AA$',
                                r'[OII] 3727$\AA$', r'[OIII] 5007$\AA$', r'[SII] 6730$\AA$')):
        plot_profile(M, label=label, ax=ax, norm=norm, label_str=label_str)
    for ax in (axes[1,:]):
        ax.set_xlabel('Slit Position Offset (arcsec)')
    for ax in (axes[:,0]):
        ax.set_ylabel('Line flux (abitrary units)')
    f.tight_layout()
    name = M.dir_ + M.model_name + '_profs.pdf'
    f.savefig(name)
    if close:
        plt.close(f)
        
def plot_slits(M, label='H__1_486133A', ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(10,7))
    im = ax.imshow(M.M_sphere.get_emis(label).sum(0), interpolation='none',label=label)
    f.colorbar(im, ax=ax)
    ax.contour(M.maskVIS, levels=(0.01, 1.0, 1.99))
    ax.contour(M.maskUVB, levels=(0.01, 1.0, 1.99), linestyles='dashed')
    name = M.dir_ + M.model_name + '_map_' + label + '.png'
    f.savefig(name, dpi=200)
    
def plot_9slits(axes=None):
    MLCO_name='tc1_LCO4'
    DLCO=2.2
    MXS_name='tc1_XS2'
    DXS=2.1
    MLCO = Model(MLCO_name, instrument='LCO', SED='BB', use_slit=True)
    MLCO.distance = DLCO
    MXS = Model(MXS_name, instrument='XShooter', SED='BB', use_slit=True)
    MXS.distance = DXS
    for M in (MLCO, MXS):
        M.read_model(HeI_cut=None, cube_size=50, verbose=False, r_out_cut = M.cm(6))
        M.make_3D(100, doplot=False, verbose=False, no_VIS=False)

    if axes is None:
        f, axes = plt.subplots(3, 3, figsize=(10, 10))
    else:
        f = plt.gcf()

    for ax,label, label_str in zip(axes.ravel(), 
                        ('H__1_486133A', 'CA_B_587564A', 'N__2_658345A', 
                         'O__1_630030A','O__2_372603A', 'O__3_500684A', 
                         'S__2_673082A', 'S__3_631206A','AR_3_519182A'),
                         (r'H$\beta$', r'HeI 5876$\AA$', r'[NII] 6583$\AA$',
                          r'[OI] 6300$\AA$', r'[OII] 3727$\AA$', r'[OIII] 5007$\AA$',
                          r'[SII] 6731$\AA$', r'[SIII] 6312$\AA$', r'[ArIII] 5192$\AA$')):
        im = ax.imshow(MXS.M_sphere.get_emis(label).sum(0),label=label, cmap='binary')
        ax.contour(MXS.maskVIS, levels=(0.5, ), linewidths=3, colors='y')
        ax.contour(MLCO.maskVIS, levels=(0.5,), linewidths=3, colors='y')
        ax.set_title(label_str)
    name = MXS.dir_ + 'map9_' + '.png'
    f.savefig(name, dpi=200)
    
def plot_dust(M, ax=None, close=False, fac_only=False):
    IR_wl = [3.5, 4.7, 9, 9.5, 22, 75, 90]
    IR_fl = [0.8, 1.7, 6, 8, 50, 60, 35]
    x = M.Mod.get_cont_x(unit='mu')
    y = M.Mod.get_cont_y(cont='ntrans', unit='Jy')
    if ax is None: 
        f, ax = plt.subplots(figsize=(5, 4))
    else:
        f = plt.gcf()
    vol_fact = (M.Mod.r_out_cut/M.Mod.r_out)**3
    
    if not fac_only:
        ax.loglog(x, y*5, linestyle='--')
    ax.loglog(x, y*5*vol_fact)
    ax.scatter(IR_wl, IR_fl, marker='+', c='r', s=150)
    ax.set_xlim((2, 200))
    ax.set_ylim((0.5, 500))
    ax.set_xlabel(r'Wavelength [$\mu$m]')
    ax.set_ylabel(r'Flux density [Jy]')
    f.tight_layout()
    name = M.dir_ + M.model_name + '_dust.pdf'
    f.savefig(name)
    if close:
        plt.close(f)
    
def get_Ibeta_coeff(M):
    epsB = M.Mod.vol_mean(M.Mod.get_emis('H__1_486133A')/M.Mod.ne/M.Mod.nH)
    Ne = M.Mod.vol_mean(M.Mod.ne)
    Rarcsec = M.arcsec(M.Mod.radius[-1])
    return (M.Mod.ff.mean() * M.distance/1.9 * 
            epsB/1.5e-25 * (Rarcsec/6)**3 * M.Mod.nH_mean/2500 * Ne/2800)

def get_im_ionic(M, atom='O', ion=1):
    return(M.M_sphere.get_ionic(atom,ion) * M.M_sphere.ne).sum(0) / M.M_sphere.ne.sum(0) 
    
    
def get_ICF(M, atom='O', ion=1):
    im_ionic = get_im_ionic(M, atom, ion)
    mfin = np.isfinite(im_ionic)
    ionic_over_mask = (im_ionic * M._mk(0.5))[mfin].sum() / M._mk(0.5)[mfin].sum()
    return 1./ionic_over_mask
    
#%%
    
def search_sol(teff=None, qh=None, M=None, Ibeta_obs = 5.1e-11, HeI_cut=-20):
    if M is None:
        M = Model('tc1_E_{:.0f}_{:.2f}'.format(teff, qh), instrument='XShooter', 
                  SED='BB', use_slit=True)
        M.densi = np.log10(2500)
        M.ff = 1.0
        M.Temp = teff
        M.luminosity_unit = 'Q(H)'
        M.luminosity= qh
        M.inner_radius = np.log10(1e16)
        M.abund['He'] = -0.98 
        M.abund['C']  = -3.1  
        M.abund['N']  = -4.25 
        M.abund['O']  = -3.25 -0.25
        M.abund['Ne'] = -4.4
        M.abund['Cl'] = -6.95 -0.05
        M.abund['Ar'] = -5.5 -0.4
        M.abund['S']  = -5.7 +0.05
        M.abund['Mg'] = -5.3
        M.abund['Fe'] = -6.4
        M.abund['Si'] = -6.1
        M.dust_type = 'graphite_myism_00005_1500_35_10.opc'
        M.dust = 1.0
    else:
        M.model_name = 'tc1_E_{:.0f}_{:.2f}'.format(M.Temp, M.luminosity)
    print('Model {}'.format(M.model_name))
        
    M.call_pyCloudy()
    distance = 1.2
    Ibeta = None
    while True:
        Ibeta_prev = Ibeta
        M.distance = distance
        M.read_model(HeI_cut=HeI_cut, cube_size=50, verbose=False, r_out_cut=M.cm(6.))
        M.make_3D(100, doplot=False, verbose=False)
        Ibeta = M.Ib / Ibeta_obs
        print('distance={:.1f} Ib/Ibobs={:.2f}'.format(distance, Ibeta))
        if Ibeta_prev is not None:
            if (Ibeta-1) * (Ibeta_prev-1) < 0.:
                break
        distance += 0.2
    if Ibeta < Ibeta_prev:
        fp = [distance, distance-0.2]
        xp = [Ibeta, Ibeta_prev]
    else:
        fp = [distance-0.2, distance]
        xp = [Ibeta_prev, Ibeta]
    distance = np.interp(x=1.0, xp=xp, fp=fp)
    print('choosen distance = {:.2f}'.format(distance))
    M.distance = distance
    M.read_model(HeI_cut=HeI_cut, cube_size=50, verbose=False, r_out_cut=M.cm(6.))
    M.make_3D(100, doplot=False, verbose=False)
    M.print_res()
    M.plot_all_profiles(close=True)
    M.plot_dust(close=True)
    
#%%
def make_Pottash():
    M = Model('tc1_Pottash_BB', instrument='LCO', SED='BB', use_slit=False)
    M.densi = np.log10(2800)
    M.ff = 1.0
    M.Temp = 34700.
    M.luminosity_unit = 'Q(H)'
    M.luminosity= 46.67#    np.log10(5000.0*3.826e33)
    M.inner_radius = np.log10(1e16)
    M.abund['He'] = 10.916 -12 
    M.abund['C'] = 8.674 -12 
    M.abund['N'] = 7.59 - 12 
    M.abund['O'] = 8.431 - 12 
    M.abund['Ne'] = 7.481 - 12
    M.abund['Ar'] = 6.478 - 12 
    M.abund['S'] = 6.2 - 12
    M.abund['Fe'] = -6.4
    M.abund['Si'] = 5.778 - 12
    M.call_pyCloudy()
    M.distance = 1.8
    M.read_model(HeI_cut=None, cube_size=50, verbose=False, r_out_cut = 1.69e17)
    M.make_3D(100, doplot=False)
    M.print_res()
    print('\a')
    return M
#%%
def make_mod_XS2():
    """
    Best XShooter model, without HeI reproduced.
    
    MXS2 = make_mod_XS2()
    MXS2.call_pyCloudy()
    MXS2.read_model(HeI_cut=None, cube_size=50, verbose=False, r_out_cut = MXS2.cm(6.))
    MXS2.make_3D(100, doplot=False, verbose=False)
    MXS2.print_res() 
    
    """
    M = Model('tc1_XS2', instrument='XShooter', SED='BB', use_slit=True)
    M.densi = np.log10(2500)
    M.dust_type = 'graphite_myism_00005_1500_35_10.opc'
    M.dust = 1.3
    M.ff = 1.0
    M.Temp = 32000.
    M.luminosity_unit = 'Q(H)'
    M.luminosity= 46.85#    np.log10(5000.0*3.826e33)
    M.inner_radius = np.log10(1e16)
    M.abund['He'] = -0.98 
    M.abund['C']  = -3.1  
    M.abund['N']  = -4.25 
    M.abund['O']  = -3.25
    M.abund['Ne'] = -4.4
    M.abund['Cl'] = -6.95
    M.abund['Ar'] = -5.6
    M.abund['S']  = -5.8
    M.abund['Mg'] = -5.1
    M.abund['Fe'] = -6.4
    M.abund['Si'] = -6.1
    M.distance = 2.1
    return M    

#%%
def make_mod_LCO4():
    """
    Best LCO model, without HeI reproduced.

    MKP4 = make_mod_LCO4()
    MKP4.call_pyCloudy()
    MKP4.read_model(HeI_cut=None, cube_size=50, verbose=False, r_out_cut = MKP4.cm(6.))
    MKP4.make_3D(100, doplot=False, verbose=False)
    MKP4.print_res() 

    """
    M = Model('tc1_LCO4', instrument='LCO', SED='BB', use_slit=True)
    M.densi = np.log10(2500)
    M.dust_type = 'graphite_myism_00005_1500_35_10.opc'
    M.dust = 1.2
    M.ff = 1.0
    M.Temp = 30000.
    M.luminosity_unit = 'Q(H)'
    M.luminosity= 47.05
    M.inner_radius = np.log10(1e16)
    M.abund['He'] = -0.98 
    M.abund['C']  = -3.25 
    M.abund['N']  = -4.25
    M.abund['O']  = -3.6
    M.abund['Ne'] = -4.4
    M.abund['Cl'] = -7.10
    M.abund['Ar'] = -6.00
    M.abund['S']  = -5.65
    M.abund['Mg'] = -5.1
    M.abund['Fe'] = -6.4
    M.abund['Si'] = -6.1
    M.distance = 2.2
    return M    
#%%

def make_obs_tab():
    lines = np.genfromtxt('lines.dat', delimiter='&', dtype=None, names='ID, lam1, lam2, f, t')
    RC = pn.RedCorr(cHbeta=0.4, R_V=3.1, law='CCM89')
    with open('new_lines.dat', 'w') as f:
        for l in lines:
            towrite = '{} & {:.2f} & {:.2f} & {:5.3f} & {:5.3f} & {}\n'.format(l['ID'], l['lam1'], l['lam2']*1.00029,
                    l['f'], l['f']*RC.getCorrHb(l['lam1']), l['t'])
            towrite = towrite.replace('b\'', '')
            towrite = towrite.replace('\\\\', '\\')
            towrite = towrite.replace('\\t\\t', '')
            towrite = towrite.replace('\\t(', '(')
            towrite = towrite.replace('\\t\\', '\\')
            towrite = towrite.replace('\\t', '**')
            towrite = towrite.replace('**ex', '\\tex')
            towrite = towrite.replace('**', '')
            towrite = towrite.replace('\\\\\\\\', '\\')
            towrite = towrite.replace('\'', ' ')
            f.write(towrite)
        
#%%

def trans_emis_label(label):
    trans_dic = {}
    trans_dic['BLND 5199.00A'] = 'N  1 5199.00A'
    trans_dic['BLND 4363.00A'] = 'O  3 4363.00A'
    trans_dic['BLND 7323.00A'] = 'O  2 7323.00A'
    trans_dic['BLND 5755.00A'] = 'N  2 5755.00A'
    trans_dic['BLND 7332.00A'] = 'O  2 7332.00A'
    trans_dic['BLND 1909.00A'] = 'C  3 1909.00A'
    trans_dic['BLND 2326.00A'] = 'C  2 2326.00A'
    trans_dic['BLND 3726.00A'] = 'O  2 3726.00A'
    trans_dic['BLND 3729.00A'] = 'O  2 3729.00A'
    trans_dic['Ca B 5875.64A'] = 'He 1 5875.64A'
    if label in trans_dic:
        new_label = trans_dic[label]
    else:
        new_label = label
    elem, ion, wlstr = new_label.split()
    if ion == 'B':
        ion = '1'
        elem = 'H'
    wlr = wlstr[-1]
    wl = float(wlstr[:-1])
    if elem in ('H', 'He'):
        if wlr == 'A':
            out_label = '~{}{} {:.0f}\AA'.format(elem, int_to_roman(int(ion)), wl)
        elif wlr == 'm':
            out_label = '~{}{} {:.2f}$\mu$m'.format(elem, int_to_roman(int(ion)), wl)
    else:
        if wlr == 'A':
            out_label = '~[{}{}] {:.0f}\AA'.format(elem, int_to_roman(int(ion)), wl) 
        elif wlr == 'm':
            out_label = '~[{}{}] {:.2f}$\mu$m'.format(elem, int_to_roman(int(ion)), wl) 
    return out_label

#%%
def make_tab_params(MLCO_name='tc1_LCO4', DLCO=2.2, 
             MXS_name='tc1_XS2', DXS=2.1, 
             outfile='tab_params.tex'):
    MLCO = Model(MLCO_name, instrument='LCO', SED='BB', use_slit=True)
    MLCO.distance = DLCO
    MXS = Model(MXS_name, instrument='XShooter', SED='BB', use_slit=True)
    MXS.distance = DXS
    for M in (MLCO, MXS):
        M.read_model(HeI_cut=None, cube_size=50, verbose=False, r_out_cut = M.cm(6))
        M.make_3D(100, doplot=False, verbose=False, no_VIS=False)
    teffLCO = float(sextract(MLCO.Mod.out['Blackbody'], 'body', '*'))
    teffXS = float(sextract(MXS.Mod.out['Blackbody'], 'body', '*'))
    str2write= """\\begin{{table}}
  \\centering
  \\caption{{Tc~1 Photoionization models parameters.}}
  \\label{{tab:models}}
  \\begin{{tabular}}{{lccc}}
	\\hline
	Parameter & LCO Model &  XS Model \\\\
	\\hline
     Distance [kpc] & {} &  {}\\\\    
     T$_{{eff}}$ [kK] & {:.1f}     &  {:.1f}        \\\\
     Q0          [s$^{{-1}}$] & {:.2f} &  {:.2f}    \\\\
     Total luminosity [L$_\odot$] & {:.0f} & {:.0f} \\\\
     n$_H$       [cm$^{{-3}}$] & {:.0f} &  {:.0f}    \\\\
     inner radius [\\arcsec] & {:.1f} &  {:.1f} \\\\
     outer radius [\\arcsec] & {:.1f} &  {:.1f} \\\\
     <log U>   & {:.2f} & {:.2f} \\\\ 
     log He/H  & {:.2f} & {:.2f}\\\\
     log C/H & {:.2f} & {:.2f}\\\\
     log N/H & {:.2f} & {:.2f}\\\\
     log O/H & {:.2f} & {:.2f}\\\\
     log Ne/H & {:.2f} & {:.2f}\\\\
     log Mg/H & {:.2f} & {:.2f}\\\\
     log Si/H & {:.2f} & {:.2f}\\\\
     log S/H & {:.2f} & {:.2f}\\\\
     log Cl/H & {:.2f} & {:.2f}\\\\
     log Ar/H & {:.2f} & {:.2f}\\\\
     log Fe/H & {:.2f} & {:.2f}\\\\
     D/G & {:.4f} & {:.4f} \\\\
     ICF(He$^{{+}}$) & {:.2f} & {:.2f} \\\\
     ICF(C$^{{++}}$) & {:.2f} & {:.2f} \\\\
     ICF(N$^{{+}}$) & {:.2f} & {:.2f} \\\\
     ICF(N$^{{++}}$) & {:.2f} & {:.2f} \\\\
     ICF(O$^{{+}}$) & {:.2f} & {:.2f} \\\\
     ICF(O$^{{++}}$) & {:.2f} & {:.2f} \\\\
     ICF(Ne$^{{++}}$) & {:.2f} & {:.2f} \\\\
     ICF(Ar$^{{++}}$) & {:.2f} & {:.2f} \\\\
     ICF(S$^{{+}}$) & {:.2f} & {:.2f} \\\\
    \\hline
  \\end{{tabular}}
\\end{{table}} 
        """.format(MLCO.distance, MXS.distance, 
                   teffLCO/1e3, teffXS/1e3,
                   np.log10(MLCO.Mod.Q0), np.log10(MXS.Mod.Q0),
                   10**float(sextract(MLCO.Mod.out['SED4'], 'Lsun:', 'Abs')),
                   10**float(sextract(MXS.Mod.out['SED4'], 'Lsun:', 'Abs')),
                   MLCO.Mod.nH_mean, MXS.Mod.nH_mean,
                   MLCO.arcsec(MLCO.Mod.r_in), MXS.arcsec(MXS.Mod.r_in), 
                   np.min((MLCO.arcsec(MLCO.Mod.r_out_cut), MLCO.arcsec(MLCO.Mod.r_out))),
                   np.min((MXS.arcsec(MXS.Mod.r_out_cut), MXS.arcsec(MXS.Mod.r_out))),
                   MLCO.Mod.log_U_mean_ne,MXS.Mod.log_U_mean_ne,
                   MLCO.Mod.abund['He'],MXS.Mod.abund['He'],
                   MLCO.Mod.abund['C'],MXS.Mod.abund['C'],
                   MLCO.Mod.abund['N'],MXS.Mod.abund['N'],
                   MLCO.Mod.abund['O'],MXS.Mod.abund['O'],
                   MLCO.Mod.abund['Ne'],MXS.Mod.abund['Ne'],
                   MLCO.Mod.abund['Mg'],MXS.Mod.abund['Mg'],
                   MLCO.Mod.abund['Si'],MXS.Mod.abund['Si'],
                   MLCO.Mod.abund['S'],MXS.Mod.abund['S'],
                   MLCO.Mod.abund['Cl'],MXS.Mod.abund['Cl'],
                   MLCO.Mod.abund['Ar'],MXS.Mod.abund['Ar'],
                   MLCO.Mod.abund['Fe'],MXS.Mod.abund['Fe'],
                   MLCO.Mod.gdgrat.sum(0).mean(), MXS.Mod.gdgrat.sum(0).mean(),
                   MLCO.get_ICF('He', 1), MXS.get_ICF('He', 1),
                   MLCO.get_ICF('C', 2), MXS.get_ICF('C', 2),
                   MLCO.get_ICF('N', 1), MXS.get_ICF('N', 1),
                   MLCO.get_ICF('N', 2), MXS.get_ICF('N', 2),
                   MLCO.get_ICF('O', 1), MXS.get_ICF('O', 1),
                   MLCO.get_ICF('O', 2), MXS.get_ICF('O', 2),
                   MLCO.get_ICF('Ne', 2), MXS.get_ICF('Ne', 2),
                   MLCO.get_ICF('Ar', 2), MXS.get_ICF('Ar', 2),
                   MLCO.get_ICF('S', 1), MXS.get_ICF('S', 1)
                   )
    print(str2write)
    with open(outfile, 'w') as f:
        f.write(str2write)
        
        
def make_tab_lines(MLCO_name='tc1_LCO4', DLCO=2.2, 
             MXS_name='tc1_XS2', DXS=2.1, 
             outfile='tab_lines.tex'):
    MLCO = Model(MLCO_name, instrument='LCO', SED='BB', use_slit=True)
    MLCO.distance = DLCO
    MXS = Model(MXS_name, instrument='XShooter', SED='BB', use_slit=True)
    MXS.distance = DXS
    for M in (MLCO, MXS):
        M.read_model(HeI_cut=None, cube_size=50, verbose=False, r_out_cut = M.cm(6))
        M.make_3D(100, doplot=False, verbose=False, no_VIS=False)
    str2write =  """\\begin{{table*}}
  \\centering
  \\caption{{Tc~1 Photoionization models results.}}
  \label{{tab:results}}
  \\begin{{tabular}}{{lcccccc}}
	\\hline
         & \\multicolumn{{3}}{{c}}{{LCO model}} & \\multicolumn{{3}}{{c}}{{XS model}} \\
        \\hline
        Line & Observation & model & $\\kappa(O)$ & Observation & Model & $\\kappa(O)$ \\
        \\hline
       ~H$\\beta$ [10$^{{-11}}$ erg/s/cm2] & 5.10 & {:.2f} & & 5.10 & {:.2f}& \\\\  
	\\hline
    """.format(MLCO.Ib/1e-11, MXS.Ib/1e-11)
    for line in MLCO.line_ordered:
        if MLCO.emis_tab[line] in ('BLND 2326.00A', 'Ca B 12.3684m'):
            str_tmp = "\hline \n"
        else:
            str_tmp = ''
        str_tmp += trans_emis_label(MLCO.emis_tab[line])
        str_tmp += " & {:.2f} & ".format(MLCO.Obs_Tc1[line])
        try:
            if np.isfinite(MLCO.flux_mod[line]):
                str_tmp += '{:6.2f} & {:6.2f} &'.format(MLCO.flux_mod[line], MLCO.QF[line])
            else:
                str_tmp += " & & "
        except:
            str_tmp += " & & "
        str_tmp += "{:.2f} & ".format(MXS.Obs_Tc1[line])
        try:
            if np.isfinite(MXS.flux_mod[line]):
                str_tmp += '{:6.2f} & {:6.2f} '.format(MXS.flux_mod[line], MXS.QF[line])
            else:
                str_tmp += " &  "
        except:
            str_tmp += " &  "
        str_tmp += '\\\\ \n'
        str_tmp = str_tmp.replace('nan', '---')
        str2write += str_tmp
    str2write += """
    \\hline
  \\end{{tabular}}
\\end{{table*}} 
        """.format()
    print(str2write)
    with open(outfile, 'w') as f:
        f.write(str2write)
        
def computeHb():
    theta = 12. # arcsec diam
    D = 1.9 # kpc
    Eps = 1.5e-25
    n_p = 2500. # cm-3
    n_e = 2800. # cm-3
    kpc2cm = 3.086e21 
    R = theta/2. / 206265. * D*kpc2cm
    Vol = 4./3 * np.pi * R**3
    Dilut = 4. *np.pi * (D*kpc2cm)**2
    
    Ibeta = Eps * n_p * n_e * Vol / Dilut
    Ibeta = Eps * n_p * n_e * 4./3 * np.pi * R**3 / (4. *np.pi * (D*kpc2cm)**2)
    Ibeta = Eps * n_p * n_e / 3 * (theta/2. / 206265.)**3 * D*kpc2cm
    print(Ibeta)
    
#%%
#%%

#===============================================================
########################## user input ##########################
#===============================================================
# Comment out what is needed in the following:
"""
print('Starting!')

MXS2 = make_mod_XS2()
MXS2.call_pyCloudy()
MXS2.read_model(HeI_cut=None, cube_size=50, verbose=False, r_out_cut = MXS2.cm(6.))
MXS2.make_3D(100, doplot=False, verbose=False)
MXS2.print_res() 

MKP4 = make_mod_LCO4()
MKP4.call_pyCloudy()
MKP4.read_model(HeI_cut=None, cube_size=50, verbose=False, r_out_cut = MKP4.cm(6.))
MKP4.make_3D(100, doplot=False, verbose=False)
MKP4.print_res() 

make_tab_params()
make_tab_lines()

plot_all_profiles(MXS2)

plot_9slits()

plot_dust(MXS2)   

"""

#%%
#===============================================================
# All finished!    
print('=================================')
print('Done!')
#===============================================================


